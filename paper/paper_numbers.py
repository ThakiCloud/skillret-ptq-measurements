#!/usr/bin/env python3
"""단일 출처 — 원장 JSON 에서만 수치를 읽는다. 그림·표·본문이 전부 여기서만 읽는다.

⛔ 파일명을 numbers.py 로 두면 표준 `numbers` 모듈을 가려 decimal 임포트가 죽는다
   ([[paper-figure-standard]]).
"""
import json, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "ledger/sweep")
LEDGER = os.path.join(ROOT, "ledger")


def _status():
    """원장 항목의 신뢰 상태. canonical 이 아닌 것은 표에 싣지 않는다.

    ⛔ 옛 판본을 지우는 대신 상태를 붙이기로 했으므로(STATUS.yaml), 그 상태를
       실제로 읽는 쪽이 없으면 라벨은 장식일 뿐이다.
    """
    p = os.path.join(LEDGER, "STATUS.yaml")
    if not os.path.exists(p):
        return {}
    try:
        import yaml
    except ImportError:      # yaml 이 없으면 필터를 못 건다 — 조용히 통과시키지 않는다
        raise SystemExit("⛔ STATUS.yaml 이 있는데 pyyaml 이 없다. pip install pyyaml")
    d = yaml.safe_load(open(p, encoding="utf-8")) or {}
    return {k: (v or {}).get("status", "canonical")
            for k, v in (d.get("entries") or {}).items()}


STATUS = _status()


def is_canonical(path):
    rel = os.path.relpath(os.path.abspath(path), LEDGER)
    return STATUS.get(rel, "canonical") == "canonical"

MODEL_LABEL = {"qwen": "Qwen3-Emb-0.6B", "gemma": "EmbeddingGemma-300M",
               "bgem3": "BGE-M3", "skillret06": "SkillRet-0.6B (finetuned)",
               "e5": "E5-base-v2"}
CORPUS_LABEL = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "skillret": "SkillRet"}
UNIFORM = ["int8/g16", "int4/g16-asym", "int4/g16-sym", "int4/g128-asym",
           "int3/g16", "int3/g32", "int2/g16", "ternary/g16"]
MODULES_I4 = ["int4-embed-only", "int4-attn-only", "int4-ffn-only"]
MODULES_I3 = ["int3-embed-only", "int3-attn-only", "int3-ffn-only"]


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(SWEEP, "*.json"))):
        # ⛔ rank-*/pool-* 는 접두어가 하나 더 붙어 있어 이 파서로는 모델명이 "rank" 가 된다.
        #    실측: protocol 표에 유령 모델·유령 코퍼스 12개가 생겨 페이지를 501pt 넘었다.
        #    이 함수는 균일/부위별 스윕만 읽는다. attn-*/student-* 도 같은 이유로 제외한다 —
    #    attn 은 본 파일에 병합됐고, student 는 별도 원장(2026-09-12-student-ood.json)이다.
        if os.path.basename(f).startswith(("rank-", "pool-", "attn-", "student-", "turn-")):
            continue
        if not is_canonical(f):      # superseded/retracted/exploratory 는 제외
            continue
        d = json.load(open(f))
        # ⛔ 이 디렉터리는 팔별 스윕 원장 전용이 아니다 — 다른 형태(집계본·부위별 격리 등)가
        #    섞여 들어오면 접두어 목록은 매번 뒤늦게 고쳐야 한다. 형태로 판정한다.
        if not isinstance(d, dict) or not isinstance(d.get("rows"), list):
            continue
        tag, _, corpus = os.path.basename(f)[:-5].partition("-")
        # ⛔ retracted 플래그가 붙은 팔은 아예 싣지 않는다. 표에 안 나와야 인용도 못 한다.
        out[(tag, corpus.replace("beir-", ""))] = {
            r["arm"]: {"ndcg": r["ndcg@10"], "se": r.get("se"),
                       "per_query": r.get("per_query")}
            for r in d["rows"] if not r.get("retracted")}
    return out


def retention(rows, model, arm):
    """fp16 대비 보존율(%) — 코퍼스 평균. 없으면 None."""
    vals = []
    for (m, c), r in rows.items():
        if m != model or arm not in r or "fp16" not in r:
            continue
        vals.append(100 * r[arm]["ndcg"] / r["fp16"]["ndcg"])
    return sum(vals) / len(vals) if vals else None


def recon():
    # ⛔ 4모델·17팔 판이 정본. 2모델·10팔 구판은 균일 사다리에 치우쳐 상관이 부풀려졌다.
    p = os.path.join(ROOT, "ledger/control/2026-09-11-recon-vs-rank-4models.json")
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    rows = load()
    models = sorted({m for m, _ in rows})
    print("모델", models, "· 코퍼스", sorted({c for _, c in rows}))
    for m in models:
        print(f"{MODEL_LABEL.get(m,m):26s}" +
              "".join(f"{(retention(rows,m,a) or float('nan')):8.1f}%" for a in UNIFORM))
    print("팔:", UNIFORM)


def rank_rows():
    """순위 보존 원장 — 없으면 None."""
    p = os.path.join(ROOT, "ledger/control/2026-09-11-rank-preservation.json")
    return json.load(open(p)) if os.path.exists(p) else None


def flip_curves(arm="int3/g16"):
    """(모델 → 마진 십분위별 뒤집힘률). 코퍼스는 질의수로 가중 평균한다."""
    import collections
    agg = collections.defaultdict(lambda: [[] for _ in range(10)])
    wts = collections.defaultdict(lambda: [[] for _ in range(10)])
    for f in sorted(glob.glob(os.path.join(SWEEP, "rank-*.json"))):
        d = json.load(open(f))
        tag = os.path.basename(f)[5:-5]
        m = tag.partition("-")[0]
        for r in d["rows"]:
            if r.get("arm") != arm or "flip_by_margin_decile" not in r:
                continue
            for x in r["flip_by_margin_decile"]:
                i = x["decile"] - 1
                agg[m][i].append(x["flip_rate"]); wts[m][i].append(x["n"])
    out = {}
    for m, dec in agg.items():
        ys = []
        for i, vals in enumerate(dec):
            w = wts[m][i]
            ys.append(sum(v * n for v, n in zip(vals, w)) / sum(w) if w else None)
        if all(y is not None for y in ys):
            out[m] = ys
    return out
