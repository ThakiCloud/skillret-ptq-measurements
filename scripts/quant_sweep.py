#!/usr/bin/env python3
"""모델·코퍼스에 무관한 양자화 스윕 — 우리 주장이 다른 인코더에서도 서는지 본다.

⛔ 왜 sentence-transformers 로 로드하나: 풀링(CLS/mean/last-token)과 프롬프트 규약이
   모델마다 다르다. 직접 구현하면 모델마다 다른 실수를 하게 되고, 그건 정확히 우리가
   접두어 불일치로 8.84pp 를 날린 실패다. 모델 자신의 규약을 쓰게 한다.

⛔ 모든 팔은 **같은 프롬프트 규약**을 쓴다. 팔마다 다르면 비교가 아니다.
"""
import argparse, json, math, os, sys, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_ft22m import fake_quant_

# ── 부위 판정 (아키텍처 무관하게 이름으로 분류) ─────────────────────────────
def part_of(name):
    n = name.lower()
    if "embed" in n and "position" not in n:
        return "embedding"
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attention", "attn")):
        return "attention"
    if any(k in n for k in ("mlp", "ffn", "intermediate", "output.dense",
                            "gate_proj", "up_proj", "down_proj")):
        return "ffn"
    return "other"


def quantize(st_model, bits, group, parts=None, sym=False):
    """parts=None 이면 전체. sym=True 면 대칭(영점 없음)."""
    tf = st_model[0].auto_model
    n = 0
    for name, p in tf.named_parameters():
        if p.dim() < 2:
            continue
        if parts and part_of(name) not in parts:
            continue
        if sym:
            _sym_quant_(p.data, bits, group)
        else:
            fake_quant_(p.data, bits, group)
        n += p.numel()
    return n


def _sym_quant_(w, bits, group):
    """대칭 양자화 — 영점 없이 absmax 만. 비대칭과의 차이를 재기 위한 팔."""
    flat = w.reshape(-1); nn = flat.numel(); pad = (-nn) % group
    if pad:
        flat = torch.cat([flat, flat[-1:].repeat(pad)])
    g = flat.reshape(-1, group)
    amax = g.abs().max(1, keepdim=True).values
    levels = 2 ** (bits - 1) - 1
    scale = (amax / levels).to(torch.float16).to(g.dtype)
    # ⛔ 스케일을 fp16 으로 내리면 전부 0(또는 극소값)인 그룹에서 0 이 된다.
    #    그대로 나누면 inf → nan 이고, 실측으로 BGE-M3 의 token_type_embeddings 가
    #    여기 걸려 "BGE-M3 는 대칭 INT4 에서 붕괴한다"는 가짜 발견을 만들 뻔했다.
    #    실제 커널(GGUF)은 전부 0인 블록을 scale=0 · 인덱스 0 으로 두고 0 으로 되돌린다.
    dead = scale <= 0
    safe = scale.masked_fill(dead, 1.0)
    q = (g / safe).round().clamp(-levels - 1, levels) * safe
    q = q.masked_fill(dead.expand_as(q), 0.0)
    w.data.copy_(q.reshape(-1)[:nn].reshape(w.shape))


# ── 코퍼스 로더 ───────────────────────────────────────────────────────────
def load_skillret(root):
    ld = lambda f: [json.loads(l) for l in open(f"{root}/{f}", encoding="utf-8") if l.strip()]
    sk = ld("data_skills_test.jsonl"); qs = ld("data_queries_test.jsonl")
    qr = ld("data_qrels_test.jsonl")
    ids = [s["id"] for s in sk]
    # ⛔ 정본 템플릿(quant_ft22m.py:106)과 글자 단위로 같아야 기존 원장과 비교된다.
    #    줄바꿈으로 바꿨더니 같은 모델이 75.26 → 75.17 로 움직였다.
    docs = [f"{r['name']} | {r['description']} | {r.get('body','')[:512]}" for r in sk]
    rel = {}
    for r in qr:
        rel.setdefault(r["query_id"], set()).add(r["skill_id"])
    qs = [q for q in qs if q["id"] in rel]
    return docs, ids, [q["query"] for q in qs], [rel[q["id"]] for q in qs]


def load_beir(name):
    """⛔ mteb/<name> 은 corpus.jsonl · queries.jsonl · qrels/test.jsonl 세 파일이다.
       `mteb/<name>-qrels` 라는 저장소는 존재하지 않는다(스모크에서 잡혔다)."""
    import json as _j
    from huggingface_hub import hf_hub_download
    g = lambda f: hf_hub_download(f"mteb/{name}", f, repo_type="dataset")
    rd = lambda f: [_j.loads(l) for l in open(g(f), encoding="utf-8") if l.strip()]
    c, q, qr = rd("corpus.jsonl"), rd("queries.jsonl"), rd("qrels/test.jsonl")
    ids = [str(x["_id"]) for x in c]
    docs = [(x.get("title", "") + "\n" + x.get("text", "")).strip() for x in c]
    rel = {}
    for r in qr:
        if int(r.get("score", 0)) > 0:
            rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    qtexts, qrels = [], []
    for x in q:
        if str(x["_id"]) in rel:
            qtexts.append(x["text"]); qrels.append(rel[str(x["_id"])])
    return docs, ids, qtexts, qrels


def turnover(D0, Q0, D1, Q1, ids, qrels, k=10):
    """교체된 문서가 정답인가 아닌가 — 추론이 아니라 측정으로.

    ⛔ 리뷰 지적(2026-09-11): "NDCG 가 안 떨어졌으니 움직인 건 비정답"은 논리적으로
       필연이 아니다. 한 질의에 정답이 여럿이면 A 가 빠지고 C 가 들어와도 NDCG 는 그대로다.
       질의별 상쇄도 평균을 0으로 만든다. qrels 가 있으니 직접 센다.
    """
    D0, Q0, D1, Q1 = (np.asarray(x, dtype=np.float64) for x in (D0, Q0, D1, Q1))
    ev_rel = ev_tot = ent_rel = ent_tot = 0
    rec0 = rec1 = n = 0
    exact = 0
    rel_ov_num = rel_ov_den = 0
    for i in range(0, len(Q0), 256):
        S0 = Q0[i:i + 256] @ D0.T
        S1 = Q1[i:i + 256] @ D1.T
        t0 = np.argpartition(-S0, k, axis=1)[:, :k]
        t1 = np.argpartition(-S1, k, axis=1)[:, :k]
        for r in range(S0.shape[0]):
            R = qrels[i + r]
            a, b = set(t0[r].tolist()), set(t1[r].tolist())
            evicted, entered = a - b, b - a
            ev_tot += len(evicted); ent_tot += len(entered)
            ev_rel += sum(1 for j in evicted if ids[j] in R)
            ent_rel += sum(1 for j in entered if ids[j] in R)
            g0 = sum(1 for j in a if ids[j] in R); g1 = sum(1 for j in b if ids[j] in R)
            rec0 += g0 / max(len(R), 1); rec1 += g1 / max(len(R), 1); n += 1
            exact += (a == b)
            ra = [j for j in t0[r] if ids[j] in R]
            if ra:
                rel_ov_num += sum(1 for j in ra if j in b); rel_ov_den += len(ra)
    return {"evicted_total": ev_tot, "evicted_relevant": ev_rel,
            "evicted_relevant_frac": round(ev_rel / max(ev_tot, 1), 4),
            "entered_total": ent_tot, "entered_relevant": ent_rel,
            "entered_relevant_frac": round(ent_rel / max(ent_tot, 1), 4),
            "recall@10_fp16": round(100 * rec0 / max(n, 1), 3),
            "recall@10_quant": round(100 * rec1 / max(n, 1), 3),
            "delta_recall@10": round(100 * (rec1 - rec0) / max(n, 1), 3),
            "frac_queries_identical_top10": round(exact / max(n, 1), 4),
            "relevant_only_top10_overlap": round(rel_ov_num / max(rel_ov_den, 1), 4)}


def rank_preservation(D0, Q0, D1, Q1, k=10, deciles=10):
    """순위 보존 — NDCG 로는 안 보이는 축.

    ⛔ 논문이 top-k 경계 마진을 얘기하는데 NDCG 만 재면 그 주장이 측정 밖에 있다.
       (a) top-k 겹침 (b) 겹치는 부분의 Kendall tau (c) 교사 경계 마진 십분위별 뒤집힘률.
       (c) 가 핵심이다 — 마진이 작을수록 더 뒤집히면 경계 가설이 실측된 것이다.
    """
    D0, Q0, D1, Q1 = (np.asarray(x, dtype=np.float64) for x in (D0, Q0, D1, Q1))
    ov, taus, margins, flips = [], [], [], []
    for i in range(0, len(Q0), 256):
        S0 = Q0[i:i + 256] @ D0.T
        S1 = Q1[i:i + 256] @ D1.T
        t0 = np.argpartition(-S0, k, axis=1)[:, :k]
        t1 = np.argpartition(-S1, k, axis=1)[:, :k]
        for r in range(S0.shape[0]):
            a_ = t0[r][np.argsort(-S0[r, t0[r]])]
            b_ = set(t1[r].tolist())
            inter = [x for x in a_ if x in b_]
            ov.append(len(inter) / k)
            if len(inter) >= 2:
                pos = {d: j for j, d in enumerate(t1[r][np.argsort(-S1[r, t1[r]])])}
                order = [pos[d] for d in inter]
                conc = dis = 0
                for x in range(len(order)):
                    for y in range(x + 1, len(order)):
                        conc += order[x] < order[y]; dis += order[x] > order[y]
                taus.append((conc - dis) / max(conc + dis, 1))
            top = np.argpartition(-S0[r], k + 1)[:k + 1]
            srt = np.sort(S0[r, top])[::-1]
            margins.append(float(srt[k - 1] - srt[k]))
            flips.append(0.0 if a_[k - 1] in b_ else 1.0)
    margins, flips = np.array(margins), np.array(flips)
    qs = np.quantile(margins, np.linspace(0, 1, deciles + 1))
    by = []
    for j in range(deciles):
        m = ((margins >= qs[j]) & (margins <= qs[j + 1])) if j == deciles - 1 else \
            ((margins >= qs[j]) & (margins < qs[j + 1]))
        if m.sum():
            by.append({"decile": j + 1, "margin_lo": round(float(qs[j]), 5),
                       "n": int(m.sum()), "flip_rate": round(float(flips[m].mean()), 4)})
    return {"top10_overlap": round(float(np.mean(ov)), 4),
            "kendall_tau": round(float(np.mean(taus)), 4) if taus else None,
            "boundary_flip_rate": round(float(flips.mean()), 4),
            "flip_by_margin_decile": by}


def ndcg10(D, Q, ids, qrels):
    D = D.astype(np.float64); Q = Q.astype(np.float64)
    per = []
    for i in range(0, len(Q), 256):
        S = Q[i:i + 256] @ D.T
        idx = np.argpartition(-S, 10, axis=1)[:, :10]
        for r_, o in enumerate(idx):
            o = o[np.argsort(-S[r_, o])]
            R = qrels[i + r_]
            h = [1.0 if ids[j] in R else 0.0 for j in o]
            dcg = sum(v / math.log2(k + 2) for k, v in enumerate(h))
            idcg = sum(1 / math.log2(k + 2) for k in range(min(len(R), 10)))
            per.append(dcg / idcg if idcg else 0.0)
    a = np.array(per)
    return 100 * a.mean(), 100 * a.std(ddof=1) / math.sqrt(len(a)), per


ARMS = [
    ("fp16",            dict(bits=16)),
    ("int8/g16",        dict(bits=8,  group=16)),
    ("int4/g16-asym",   dict(bits=4,  group=16)),
    ("int4/g16-sym",    dict(bits=4,  group=16, sym=True)),
    ("int4/g128-asym",  dict(bits=4,  group=128)),
    ("int3/g16",        dict(bits=3,  group=16)),
    ("int2/g16",        dict(bits=2,  group=16)),
    ("ternary/g16",     dict(bits=2,  group=16, sym=True)),
    ("int4-embed-only", dict(bits=4,  group=16, parts={"embedding"})),
    ("int4-attn-only",  dict(bits=4,  group=16, parts={"attention"})),
    ("int4-ffn-only",   dict(bits=4,  group=16, parts={"ffn"})),
    # int4/g16 에서는 세 부위 모두 손실이 SE 안이라 배분할 여지가 없다(실측 2026-09-11).
    # 민감도는 모델이 이미 무너지기 시작하는 비트폭에서만 드러난다 — int3 을 같이 잰다.
    ("int3-embed-only", dict(bits=3,  group=16, parts={"embedding"})),
    ("int3-attn-only",  dict(bits=3,  group=16, parts={"attention"})),
    ("int3-ffn-only",   dict(bits=3,  group=16, parts={"ffn"})),
    ("int3/g32",        dict(bits=3,  group=32)),
    ("int3g32-ffn-only",   dict(bits=3, group=32, parts={"ffn"})),
    ("int3g32-embed-only", dict(bits=3, group=32, parts={"embedding"})),
    # ⛔ 리뷰 지적(2026-09-11): int3/g32 에 어텐션 단독 팔이 없어 Σparts 가 세 부위의 합이
    #    아니었다. 그 상태로 joint/Σparts 비율을 핵심 증거로 썼다 — 표 숫자를 계산하면 드러난다.
    ("int3g32-attn-only", dict(bits=3, group=32, parts={"attention"})),
    # ⛔ 2026-09-12 사용자 지적: 절벽(INT2)에서 부위별을 한 번도 재지 않았다. 그래서
    #    "임베딩 테이블은 어디서나 거의 공짜"가 절벽 이전 구간에서만 측정된 주장이었다.
    #    어휘가 큰 모델이 무너지는 경향이 있어(E5 30k 생존 / BGE 250k·Qwen 151k 붕괴)
    #    토크나이저·어휘가 원인이라는 가설이 서는데, 그 가설이 맞으면 임베딩 단독
    #    INT2 가 붕괴를 재현해야 한다. 이 세 팔이 그 검정이다.
    ("int2-embed-only", dict(bits=2, group=16, parts={"embedding"})),
    ("int2-attn-only",  dict(bits=2, group=16, parts={"attention"})),
    ("int2-ffn-only",   dict(bits=2, group=16, parts={"ffn"})),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="skillret:<root> | beir:<name>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--arms", default="")
    ap.add_argument("--pooling", choices=["mean", "cls", "lasttoken"], default=None,
                    help="풀링을 강제로 바꾼다 — 풀링이 INT2 생존의 원인인지 조작 검증")
    ap.add_argument("--rank-metrics", action="store_true",
                    help="top-k 겹침·Kendall tau·경계 마진 십분위별 뒤집힘률")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    kind, _, arg = a.corpus.partition(":")
    docs, ids, qs, qrels = load_skillret(arg) if kind == "skillret" else load_beir(arg)
    if a.max_docs and len(docs) > a.max_docs:      # 스모크용
        keep = set()
        for R in qrels[:20]:
            keep |= R
        extra = [i for i, x in enumerate(ids) if x not in keep][: a.max_docs]
        sel = sorted(set(extra) | {i for i, x in enumerate(ids) if x in keep})
        docs = [docs[i] for i in sel]; ids = [ids[i] for i in sel]
        qs, qrels = qs[:20], qrels[:20]
    print(f"corpus={a.corpus} docs={len(docs)} queries={len(qs)}", flush=True)

    want = set(a.arms.split(",")) if a.arms else None
    ref = None
    rows = []
    base = SentenceTransformer(a.model, trust_remote_code=True)
    prompts = getattr(base, "prompts", {}) or {}
    qprompt = prompts.get("query") or prompts.get("s2p_query") or None
    print(f"풀링={type(base[1]).__name__ if len(base)>1 else '?'} 질의프롬프트={qprompt!r}", flush=True)

    if a.pooling:
        # ⛔ 관찰(mean 풀링 모델만 INT2 생존)은 백본 계열과 교란돼 있다.
        #    같은 체크포인트의 풀링만 바꿔 생존이 따라오는지 본다.
        pool = None
        for mod in base:
            if type(mod).__name__ == "Pooling":
                pool = mod; break
        if pool is None:
            raise SystemExit("⛔ Pooling 모듈이 없다 — 풀링 조작 불가")
        # ⛔ sentence-transformers 6.x 는 불리언이 아니라 pooling_mode 문자열 하나다.
        #    구버전 API(pooling_mode_*_tokens)로 쓰면 조용히 안 바뀌는 게 아니라 죽는다.
        if hasattr(pool, "pooling_mode"):
            pool.pooling_mode = a.pooling
        else:                                   # 구버전 호환
            for attr in ("pooling_mode_mean_tokens", "pooling_mode_cls_token",
                         "pooling_mode_lasttoken", "pooling_mode_max_tokens",
                         "pooling_mode_mean_sqrt_len_tokens"):
                if hasattr(pool, attr):
                    setattr(pool, attr, False)
            tgt = {"mean": "pooling_mode_mean_tokens", "cls": "pooling_mode_cls_token",
                   "lasttoken": "pooling_mode_lasttoken"}[a.pooling]
            setattr(pool, tgt, True)
        assert getattr(pool, "pooling_mode", a.pooling) == a.pooling, "풀링이 안 바뀌었다"
        print(f"풀링 강제 변경 → {getattr(pool, 'pooling_mode', a.pooling)}", flush=True)

    sd = copy.deepcopy(base[0].auto_model.state_dict())
    for tag, kw in ARMS:
        if want and tag not in want:
            continue
        base[0].auto_model.load_state_dict(sd)     # ⛔ 팔마다 원본에서 다시 시작
        nq = 0 if kw["bits"] >= 16 else quantize(
            base, kw["bits"], kw.get("group", 16), kw.get("parts"), kw.get("sym", False))
        D = base.encode(docs, batch_size=a.batch, normalize_embeddings=True,
                        show_progress_bar=False)
        Q = base.encode(qs, batch_size=a.batch, normalize_embeddings=True,
                        show_progress_bar=False, prompt=qprompt)
        n, se, per = ndcg10(np.asarray(D), np.asarray(Q), ids, qrels)
        row = {"arm": tag, "ndcg@10": round(n, 2), "se": round(se, 2),
               "quantized_params": nq, "per_query": per}
        if tag == "fp16":
            ref = (np.asarray(D), np.asarray(Q))
        elif a.rank_metrics and ref is not None:
            row.update(rank_preservation(ref[0], ref[1], np.asarray(D), np.asarray(Q)))
            row.update(turnover(ref[0], ref[1], np.asarray(D), np.asarray(Q), ids, qrels))
        rows.append(row)
        print(f"  {tag:18s} NDCG@10 {n:6.2f} ± {se:.2f}   (양자화 {nq/1e6:.1f}M)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model": a.model, "corpus": a.corpus, "n_docs": len(docs),
               "n_queries": len(qs), "query_prompt": qprompt, "rows": rows},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("→", a.out)


if __name__ == "__main__":
    main()
