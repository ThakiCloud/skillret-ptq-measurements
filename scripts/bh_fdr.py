#!/usr/bin/env python3
"""INT4 부위별 45셀의 다중비교 보정 — 논문 §Module sensitivity 의 "BH 보정 뒤 6개" 를 원시
per_query 에서 재계산한다. ⛔ 이 수치는 그동안 control/2026-09-12-robustness.json 에 손으로만
적혀 있었다(감사 2026-09-14). 값은 코드가 낸다([[sonnet-format-determinism]]).

방법: paired_ci.py 와 같은 부트스트랩(질의 단위 재표집 10,000회, seed 20260911). 양측 p 는
재표집 평균이 0 의 양쪽에 떨어지는 경험 비율의 2배(최소 1/n). BH 는 q=0.05, 45셀."""
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
from paper_numbers import sweep_files  # noqa: E402
ARMS = ["int4-embed-only", "int4-attn-only", "int4-ffn-only"]


def cells(n_boot, seed, arms=ARMS):
    """(model, corpus, arm) 셀별 질의 단위 페어드 부트스트랩. 원장 파일 집합은 paper_numbers.sweep_files()
    와 같다 — 여기만 따로 glob 하면 STATUS 로 superseded 된 행(2026-09-14 prompt-contract)이 되살아난다."""
    out = []
    for f, model, corpus, d in sweep_files():
        b = os.path.basename(f)
        by = {r["arm"]: r for r in d["rows"] if not r.get("retracted")}
        if "fp16" not in by or not by["fp16"].get("per_query"):
            continue
        fp = np.asarray(by["fp16"]["per_query"], float)
        for arm in arms:
            r = by.get(arm)
            if not r or not r.get("per_query"):
                continue
            q = np.asarray(r["per_query"], float)
            assert len(q) == len(fp), (b, arm)
            delta = 100 * (q - fp)                      # NDCG points, per query
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
            means = delta[idx].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            p = 2 * min((means <= 0).mean(), (means >= 0).mean())
            p = max(p, 1.0 / n_boot)
            out.append({"model": model, "corpus": corpus, "arm": arm, "n_queries": int(len(delta)),
                        "delta_pp": round(float(delta.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                        "ci_excludes_zero": bool(lo > 0 or hi < 0), "p_two_sided": float(p),
                        "paired_se": round(float(delta.std(ddof=1) / np.sqrt(len(delta))), 3)})
    return out


def bh(rows, q):
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i]["p_two_sided"])
    thresh = 0.0
    for rank, i in enumerate(order, 1):
        if rows[i]["p_two_sided"] <= q * rank / m:
            thresh = rows[i]["p_two_sided"]
    for r in rows:
        r["bh_significant"] = r["p_two_sided"] <= thresh and thresh > 0
    return sum(r["bh_significant"] for r in rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, "ledger/control/2026-09-14-bh-fdr-int4.json"))
    ap.add_argument("--ci-out", default=os.path.join(ROOT, "ledger/control/2026-09-14-paired-ci-int4.json"),
                    help="Table~ci 가 읽는 60셀(부위 45 + 균일 int4/g16-asym 15) 페어드 CI 원장. 2026-09-11 판은 산출 스크립트가 없었다.")
    a = ap.parse_args()
    rows = cells(a.n_boot, a.seed)
    ci_rows = [{k: v for k, v in r.items() if k not in ("ci_excludes_zero", "p_two_sided")}
               | {"covers_zero": not r["ci_excludes_zero"]} for r in cells(a.n_boot, a.seed, ARMS + ["int4/g16-asym"])]
    json.dump({"date": "2026-09-14", "method": "질의 단위 페어드 부트스트랩 10000회, 백분위 95% CI",
               "note": "ΔNDCG_q = NDCG_quant(q) - NDCG_fp16(q). 같은 질의가 두 팔을 모두 통과한다. E5·Gemma 셀은 sweep-promptfix/ 행(프롬프트 계약 수정 후).",
               "n_cells": len(ci_rows), "n_covering_zero": sum(r["covers_zero"] for r in ci_rows), "rows": ci_rows},
              open(a.ci_out, "w"), ensure_ascii=False, indent=1)
    print(f"paired-ci cells={len(ci_rows)} covering_zero={sum(r['covers_zero'] for r in ci_rows)} → {a.ci_out}")
    n_sig = bh(rows, a.q)
    n_ci = sum(r["ci_excludes_zero"] for r in rows)
    n_pos = sum(r["ci_excludes_zero"] and r["delta_pp"] > 0 for r in rows)
    largest = max(rows, key=lambda r: abs(r["delta_pp"]))
    res = {"date": "2026-09-14", "subject": "INT4/g16 부위별 45셀 — 페어드 부트스트랩 양측 p + BH(q=0.05) 재계산",
           "method": f"per-query paired bootstrap n={a.n_boot} seed={a.seed}; p=2*min(empirical tails), floor 1/n; BH step-up q={a.q}",
           "n_cells": len(rows), "ci_excludes_zero": n_ci, "ci_excludes_zero_positive": n_pos,
           "bh_significant_q05": n_sig, "largest_abs_delta_pp": largest["delta_pp"],
           "largest_cell": f"{largest['model']}/{largest['corpus']}/{largest['arm']}", "rows": rows}
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"cells={len(rows)} ci_excludes_zero={n_ci} (positive {n_pos}) bh_significant={n_sig} "
          f"largest={largest['delta_pp']:+.2f} @ {res['largest_cell']} → {a.out}")
