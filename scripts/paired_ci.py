#!/usr/bin/env python3
"""질의 단위 페어드 부트스트랩 CI — "SE 보다 작다"를 실제 숫자로 만든다.

⛔ 리뷰 지적(2026-09-11): 초록이 "every isolated module cost is less than the standard error"
   라고 쓰는데 표에 SE 도 CI 도 없었다. 같은 질의가 두 팔을 모두 통과하므로 페어드가 맞다 —
   절대 SE 는 질의 난이도 분산까지 세어 과대평가한다.
"""
import argparse, json, glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def boot(delta, n=10000, seed=20260911):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "ledger/sweep/*.json"))):
        b = os.path.basename(f)
        if b.startswith(("rank-", "pool-")):
            continue
        d = json.load(open(f))
        model, _, corpus = b[:-5].partition("-")
        by = {r["arm"]: r for r in d["rows"] if not r.get("retracted")}
        if "fp16" not in by or not by["fp16"].get("per_query"):
            continue
        fp = np.array(by["fp16"]["per_query"], dtype=float)
        for arm in a.arms:
            r = by.get(arm)
            if not r or not r.get("per_query"):
                continue
            q = np.array(r["per_query"], dtype=float)
            if len(q) != len(fp):
                print(f"⛔ {b} {arm}: 길이 불일치 {len(q)} vs {len(fp)} — 건너뜀"); continue
            dl = 100.0 * (q - fp)                       # 질의별 ΔNDCG (포인트)
            m, lo, hi = boot(dl, a.n_boot)
            rows.append({"model": model, "corpus": corpus.replace("beir-", ""), "arm": arm,
                         "n_queries": len(dl), "delta_pp": round(m, 3),
                         "ci95": [round(lo, 3), round(hi, 3)],
                         "covers_zero": bool(lo <= 0 <= hi),
                         "paired_se": round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 3)})
    cov = sum(r["covers_zero"] for r in rows)
    json.dump({"method": f"질의 단위 페어드 부트스트랩 {a.n_boot}회, 백분위 95% CI",
               "note": "ΔNDCG_q = NDCG_quant(q) - NDCG_fp16(q). 같은 질의가 두 팔을 모두 통과한다",
               "n_cells": len(rows), "n_covering_zero": cov, "rows": rows},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(rows)}셀 · 0을 포함하는 CI {cov}개")
    for r in rows:
        mark = "0포함" if r["covers_zero"] else "⚠ 0제외"
        print(f"  {r['model']:11s}{r['corpus']:9s}{r['arm']:20s} "
              f"Δ{r['delta_pp']:+7.3f} [{r['ci95'][0]:+7.3f},{r['ci95'][1]:+7.3f}] {mark}")


if __name__ == "__main__":
    sys.exit(main())
