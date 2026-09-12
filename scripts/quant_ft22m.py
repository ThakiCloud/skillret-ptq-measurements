#!/usr/bin/env python3
"""파인튜닝된 22M 을 실제 배포 비트폭으로 양자화해 손상을 잰다.

⛔ Linear 만 양자화하면 이 모델의 절반을 안 건드린 것이다 — arctic-embed-xs 는
   vocab 30522×384 = 11.7M 으로 **전체 22.7M 의 51.6%** 가 임베딩이다.
   그래서 임베딩 테이블까지 같은 비트폭으로 재고, 크기도 그렇게 계산한다.
"""
import argparse, json, math, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix import resolve as resolve_prefix   # ⛔ 접두어는 단일출처(prefix.py)에서만 온다
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

C = os.path.expanduser("~/data/skillret")


ld = lambda f: [json.loads(l) for l in open(f"{C}/{f}", encoding="utf-8") if l.strip()]


def fake_quant_(w, bits, group=128, scale_dtype=torch.float16):
    """그룹별 비대칭 min/max 양자화 — 배포 커널(GGUF Q4_K/Q8_0)과 같은 형태.

    ⛔ scale·zero 는 **fp16 으로 저장된다**(크기 산식도 그룹당 32비트로 그렇게 센다).
       float32 scale 로 역양자화하면 디스크에 존재하지 않는 정밀도로 점수를 내게 된다 —
       실측 차이가 가중치 RMSE 의 28% 였다. 저장 왕복을 여기서 반영한다.
       `scale_dtype=None` 으로 옛 동작(fp32 scale)을 재현할 수 있다(대조용).
    """
    if bits >= 16:
        return 0.0
    orig = w.detach().clone()
    flat = w.reshape(-1)
    n = flat.numel(); pad = (-n) % group
    if pad:  # ⛔ 0 으로 패딩하면 min/max 가 오염된다 — 마지막 실값으로 채운다
        flat = torch.cat([flat, flat[-1:].repeat(pad)])
    g = flat.reshape(-1, group)
    lo, hi = g.min(1, keepdim=True).values, g.max(1, keepdim=True).values
    levels = 2 ** bits - 1
    scale = ((hi - lo) / levels).clamp(min=1e-12)
    idx = ((g - lo) / scale).round().clamp(0, levels)
    if scale_dtype is not None:            # 디스크 저장 정밀도로 내림
        scale = scale.to(scale_dtype).to(g.dtype)
        lo = lo.to(scale_dtype).to(g.dtype)
    q = idx * scale + lo
    out = q.reshape(-1)[:n].reshape(w.shape)
    w.data.copy_(out)
    return float((orig - out).pow(2).mean().sqrt())


def size_mb(model, bits, group=128):
    """비트폭 + 그룹당 scale/zero(fp16 2개) 포함한 실제 파일 크기."""
    tot = 0.0
    for p in model.parameters():
        n = p.numel()
        # ⛔ fp16 은 그룹 scale·zero 가 없다 — 오버헤드를 더하면 46.14MB 가 51.10MB 로 부풀어
        #    파레토 표에서 fp16 팔이 실제보다 크게 보인다.
        if n < group or p.dim() < 2 or bits >= 16:
            tot += n * 16
        else:
            tot += n * bits + math.ceil(n / group) * 32
    return tot / 8 / 1e6


def evaluate(model, tok, sk, qs, rel, ids, dt, dev, qp=""):
    def enc(ts, bs=64):
        o = []
        for i in range(0, len(ts), bs):
            e = tok(ts[i:i+bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(dev)
            with torch.no_grad():
                h = model(**e).last_hidden_state[:, 0]
            o.append(F.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(o).astype(np.float64)
    D, Q = enc(dt), enc([qp + q["query"] for q in qs])
    nd = rc = 0.0; per = []
    for i in range(0, len(Q), 256):
        S = Q[i:i+256] @ D.T
        idx = np.argpartition(-S, 10, axis=1)[:, :10]
        for r_, o in enumerate(idx):
            o = o[np.argsort(-S[r_, o])]
            R = rel[qs[i+r_]["id"]]
            h = [1.0 if ids[j] in R else 0.0 for j in o]
            dcg = sum(v / math.log2(k+2) for k, v in enumerate(h))
            idcg = sum(1 / math.log2(k+2) for k in range(min(len(R), 10)))
            x = dcg / idcg if idcg else 0.0
            nd += x; rc += sum(h) / len(R); per.append(x)
    n = len(qs)
    return (nd/n*100, float(np.std(per, ddof=1)/np.sqrt(n))*100, rc/n*100, per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/skillret-ft22m"))
    ap.add_argument("--bits", default="16,8,4,3,2")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--out", default="ledger/control/2026-09-10-ft22m-quant.json")
    a = ap.parse_args()

    sk, qs = ld("data_skills_test.jsonl"), ld("data_queries_test.jsonl")
    rel = {}
    for r in ld("data_qrels_test.jsonl"):
        if r.get("relevance", 1) > 0:
            rel.setdefault(r["query_id"], set()).add(r["skill_id"])
    qs = [q for q in qs if q["id"] in rel]
    ids = [s["id"] for s in sk]
    dt = [f"{r['name']} | {r['description']} | {r.get('body','')[:512]}" for r in sk]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)

    rows = []; per_all = {}
    for bits in [float(b) for b in a.bits.split(",")]:
        m = AutoModel.from_pretrained(a.model).eval().to(dev)
        errs = []
        if bits < 16:
            with torch.no_grad():
                for p in m.parameters():
                    if p.dim() >= 2 and p.numel() >= a.group:
                        errs.append(fake_quant_(p, int(bits), a.group))
        nd, se, rcl, per = evaluate(m, tok, sk, qs, rel, ids, dt, dev,
                                    resolve_prefix(a.model, a.query_prefix))
        mb = size_mb(m, bits, a.group)
        label = f"{'fp16' if bits >= 16 else f'int{int(bits)}'} (g{a.group})"
        rows.append({"bits": bits, "arm": label, "size_mb": round(mb, 2),
                     "ndcg@10": nd, "se": se, "recall@10": rcl,
                     "weight_rmse": round(float(np.mean(errs)), 6) if errs else 0.0})
        per_all[label] = per
        print(f"{label:<16} {mb:6.2f}MB  NDCG@10 {nd:6.2f} ± {se:.2f} · R@10 {rcl:6.2f}",
              flush=True)
        del m
    base = rows[0]["ndcg@10"]
    for r in rows:
        r["delta_vs_fp16"] = round(r["ndcg@10"] - base, 2)
    json.dump({"model": a.model, "n_queries": len(qs), "n_docs": len(sk),
               "group": a.group, "rows": rows, "per_query": per_all},
              open(a.out, "w"), ensure_ascii=False, indent=2)
    print("→", a.out)


if __name__ == "__main__":
    raise SystemExit(main())
