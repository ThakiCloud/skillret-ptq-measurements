#!/usr/bin/env python3
"""srpromptfix(2026-09-14) 재측정을 정본으로 승격한 뒤, 그 행을 읽는 **집계 원장 두 개**를 다시 만든다.

배경 — 감사 (ii)/(iii): 평가기가 프롬프트를 질의에만 적용해 E5-base-v2 는 query:/passage: 없이,
EmbeddingGemma 는 문서 접두어 없이 측정됐다. sweep-promptfix/ 가 계약대로 다시 잰 행이고
STATUS.yaml 이 옛 행을 superseded 로 돌렸다. 그런데 두 집계 원장이 옛 행에서 파생돼 있었다:

  1. sweep/2026-09-12-int2-module-isolation.json  — INT2 부위별 셀 (e5·gemma 셀만 다시 만든다)
  2. control/2026-09-11-recon-vs-rank-4models.json — recon 오차 ↔ 손실. recon 은 **가중치만의 양**이라
     promptfix 잡이 낸 값과 소수점까지 같다(실측 Δ=0). 손실(ndcg_drop_pct)만 다시 계산한다.

⛔ 회귀 가드: 무영향 3모델(qwen·skillret06·bgem3)의 값은 옛 원장과 반올림 안에서 같아야 한다.
   같지 않으면 재계산 코드가 옛 원장과 다른 정의를 쓰는 것이므로 여기서 죽는다.
⛔ 분석 블록(pooled·per_model·within-bitwidth·robustness)은 **옛 행으로 먼저 돌려** 옛 analysis 와
   대조한다(self-check). 부트스트랩 CI 만 시드가 기록돼 있지 않아 대조 대상이 아니다.
"""
import json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
from paper_numbers import load, INT2_ISO, RECON_PATH, LEDGER  # noqa: E402

OLD_ISO = os.path.join(LEDGER, "sweep/2026-09-12-int2-module-isolation.json")
OLD_RECON = os.path.join(LEDGER, "control/2026-09-11-recon-vs-rank-4models.json")
REMEASURED = ("e5", "gemma")
CORPUS_LABEL = {"scifact": "beir:scifact", "nfcorpus": "beir:nfcorpus", "skillret": "skillret:public-test"}
UNIFORM = {"int8/g16", "int4/g16-asym", "int4/g16-sym", "int4/g128-asym", "int3/g16", "int3/g32", "int2/g16", "ternary/g16"}
BIT_OF = {"int4-": "INT4/g16", "int3-": "INT3/g16", "int3g32-": "INT3/g32", "int2-": "INT2/g16"}
I2ARMS = ["int2/g16", "int2-embed-only", "int2-attn-only", "int2-ffn-only"]
SEED = 20260914


def pearson(x, y):
    if len(x) < 3:
        return None
    r = float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])
    return None if math.isnan(r) else round(r, 3)


def spearman(x, y):
    if len(x) < 3:
        return None
    from scipy.stats import rankdata
    return pearson(rankdata(x), rankdata(y))


def boot_ci(x, y, n=10000, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        vals.append(np.corrcoef(x[idx], y[idx])[0, 1])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def rebuild_int2_iso(rows):
    old = json.load(open(OLD_ISO))
    cells = {}
    for m, cs in old["cells"].items():
        if m not in REMEASURED:
            cells[m] = cs
            continue
        cells[m] = []
        for c in ("scifact", "skillret", "nfcorpus"):          # 옛 원장의 코퍼스 순서를 유지
            r = rows[(m, c)]
            fp = r["fp16"]["ndcg"]
            cells[m].append({"model": m, "corpus": CORPUS_LABEL[c], "fp16_ndcg@10": fp,
                             "rows": [{"arm": a, "ndcg@10": r[a]["ndcg"],
                                       "retained_pct": round(100 * r[a]["ndcg"] / fp, 2)} for a in I2ARMS]})
    out = {k: v for k, v in old.items() if k != "cells"}
    out["date"] = "2026-09-14"
    out["experiment"] = old.get("experiment", "int2-module-isolation") + " (promptfix merge)"
    out["provenance"] = {"unchanged_from": os.path.relpath(OLD_ISO, LEDGER), "unchanged_models": [m for m in old["cells"] if m not in REMEASURED],
                         "rebuilt_models": list(REMEASURED), "rebuilt_from": "sweep-promptfix/ (job srpromptfix, H200 tk-ai-wkld-wk-gpu-003, 2026-09-14)",
                         "why": "audit item (ii): the 2026-09-12 run applied prompts to queries only; E5 had no query:/passage:, Gemma no document prefix"}
    out["honest_caveats"] = list(old.get("honest_caveats", [])) + [
        "e5·gemma 셀은 2026-09-14 재측정(h200)이고 나머지 3모델은 2026-09-12 측정(b200)이다 — RTN 은 결정론적이라 체크포인트 잡음은 없고, 평가 수치 차이는 fp16 행렬곱 순서 수준이다"]
    out["cells"] = cells
    json.dump(out, open(INT2_ISO, "w"), ensure_ascii=False, indent=1)
    return out


def drop_pct(rows, iso, model, arm):
    """세 코퍼스 평균 NDCG 손실(%). INT2 부위 팔은 격리 원장에서(옛 4models 와 같은 출처 규칙)."""
    if arm.startswith("int2-") and arm.endswith("-only"):
        v = [100 - r["retained_pct"] for c in iso["cells"][model] for r in c["rows"] if r["arm"] == arm]
    else:
        v = [100 * (r["fp16"]["ndcg"] - r[arm]["ndcg"]) / r["fp16"]["ndcg"] for (m, _), r in rows.items() if m == model and arm in r]
    return round(float(np.mean(v)), 3), len(v)


def analysis(rows):
    live = [r for r in rows if r["ndcg_drop_pct"] < 50]
    models = sorted({r["model"] for r in rows})
    X = lambda rs: [r["recon_rel_err"] for r in rs]
    Y = lambda rs: [r["ndcg_drop_pct"] for r in rs]
    A = {"pooled_pearson": pearson(X(rows), Y(rows)),
         "per_model": {m: pearson(X([r for r in rows if r["model"] == m]), Y([r for r in rows if r["model"] == m])) for m in models},
         "excluding_collapsed_pooled": pearson(X(live), Y(live)),
         "excluding_collapsed_per_model": {m: pearson(X([r for r in live if r["model"] == m]), Y([r for r in live if r["model"] == m])) for m in models},
         "note": "붕괴 팔(손실 50% 초과)을 넣으면 상관이 부풀려진다 — 오차도 크고 손실도 크니까. 배분이 실제로 필요한 구간은 '살아 있는' 팔이다."}
    fam = {}
    for m in models:
        u = [r for r in live if r["model"] == m and r["arm"] in UNIFORM]
        md = [r for r in live if r["model"] == m and r["arm"] not in UNIFORM]
        al = [r for r in live if r["model"] == m]
        fam[m] = {"uniform": pearson(X(u), Y(u)), "module": pearson(X(md), Y(md)), "all": pearson(X(al), Y(al))}
    A["by_arm_family"] = fam
    u = [r for r in live if r["arm"] in UNIFORM]; md = [r for r in live if r["arm"] not in UNIFORM]
    A["by_arm_family_pooled"] = {"uniform": pearson(X(u), Y(u)), "module": pearson(X(md), Y(md)), "all": A["excluding_collapsed_pooled"]}
    rob = {}
    for m in models:
        def blk(rs):
            return {"pearson": pearson(X(rs), Y(rs)), "spearman": spearman(X(rs), Y(rs)), "ci95": boot_ci(X(rs), Y(rs)), "n": len(rs)}
        rob[m] = {"uniform_excl_collapse": blk([r for r in live if r["model"] == m and r["arm"] in UNIFORM]),
                  "uniform_incl_collapse": blk([r for r in rows if r["model"] == m and r["arm"] in UNIFORM]),
                  "module_excl_collapse": blk([r for r in live if r["model"] == m and r["arm"] not in UNIFORM]),
                  "all_incl_collapse": blk([r for r in rows if r["model"] == m])}
    A["robustness"] = rob
    A["robustness_note"] = ("리뷰 지적(2026-09-11): 붕괴 팔을 빼면 비선형 구간이 제거돼 상관이 올라간 것 아니냐는 질문이 가능하다. "
                            f"포함/제외 둘 다, Pearson·Spearman 둘 다 싣는다. ci95 = 팔 단위 백분위 부트스트랩 10000회, seed {SEED}.")
    within = {}
    for pre, lab in BIT_OF.items():
        pts = [r for r in md if r["arm"].startswith(pre)]
        within[lab] = {"n": len(pts), "pearson": pearson(X(pts), Y(pts))}
    A["module_axis_within_bitwidth"] = within
    A["module_axis_within_bitwidth_note"] = ("⛔ 비트폭을 섞어 module 축 상관을 내면 '얼마나 세게 눌렀나'(균일 축)가 섞여 들어와 범위 확장으로 "
                                             "상관이 부풀려진다. 배분기가 푸는 문제는 '같은 비트폭에서 어느 부위를 보호할까'이므로 비트폭 안 상관이 그 주장에 맞는 통계다.")
    i2 = [r for r in rows if r["arm"].startswith("int2-") and r["arm"].endswith("-only")]
    A["int2_module_all15"] = {"n": len(i2), "pearson": pearson(X(i2), Y(i2)), "note": "사전규칙(손실 50% 초과 제외)을 풀면 이 값이 된다. 본표는 규칙을 유지한다."}
    return A


def _is_i2mod(r):
    return r["arm"].startswith("int2-") and r["arm"].endswith("-only")


def _is_g32attn(r):
    return r["arm"] == "int3g32-attn-only"


def selfcheck(old):
    """분석 코드가 옛 원장의 analysis 를 재현하는가.

    옛 원장(2026-09-11-recon-vs-rank-4models.json)의 analysis 는 **세 시점의 행 집합**이 섞여 있다(실측):
      72행 (09-11 원본: INT2 부위 팔·int3g32-attn-only 없음)   → per_model · robustness
      87행 (09-12 INT2 부위 팔 15개 추가, g32-attn 은 아직 없음) → pooled · excluding_collapsed · by_arm_family · pooled_family · int2_all15
      92행 (09-12 g32-attn recon CPU 백필 뒤)                     → module_axis_within_bitwidth
    행을 더할 때 앞 블록을 다시 내지 않았다. 그래서 본문의 "INT2 부위 팔을 넣으면 pooled module 0.658" 은
    INT2 팔은 들어 있지만 int3g32-attn-only 5행이 빠진 값이고, 같은 옛 손실로 92행 전부면 0.651 이다.
    여기서는 각 블록을 그 블록이 실제로 쓴 행 집합으로 재현해 코드가 옛 코드와 같음을 보이고(20개 robustness
    블록의 pearson·spearman·n 까지 일치), 새 원장은 92행 전부로 모든 블록을 다시 낸다."""
    O = old["analysis"]
    A72 = analysis([r for r in old["rows"] if not _is_i2mod(r) and not _is_g32attn(r)])
    A87 = analysis([r for r in old["rows"] if not _is_g32attn(r)])
    A92 = analysis(old["rows"])
    bad = []
    if A72["per_model"] != O["per_model"]: bad.append(("per_model@72", A72["per_model"], O["per_model"]))
    for m, blocks in O["robustness"].items():
        for b, v in blocks.items():
            a = A72["robustness"][m][b]
            if (a["pearson"], a["spearman"], a["n"]) != (v["pearson"], v["spearman"], v["n"]):
                bad.append(("robustness@72", m, b, (a["pearson"], a["spearman"], a["n"]), (v["pearson"], v["spearman"], v["n"])))
    for k in ("pooled_pearson", "excluding_collapsed_pooled", "excluding_collapsed_per_model", "by_arm_family", "by_arm_family_pooled"):
        if A87[k] != O[k]: bad.append((k + "@87", A87[k], O[k]))
    if A87["int2_module_all15"]["pearson"] != O["int2_module_all15"]["pearson"]: bad.append(("all15@87", A87["int2_module_all15"], O["int2_module_all15"]))
    for lab, v in O["module_axis_within_bitwidth"].items():
        if (A92["module_axis_within_bitwidth"][lab]["n"], A92["module_axis_within_bitwidth"][lab]["pearson"]) != (v["n"], v["pearson"]):
            bad.append(("within@92", lab, A92["module_axis_within_bitwidth"][lab], v))
    return bad, A72, A87, A92


def main():
    rows = load()
    iso = rebuild_int2_iso(rows)
    old = json.load(open(OLD_RECON))
    bad, A72, A87, A_all92 = selfcheck(old)
    if bad:
        print("⛔ self-check: 분석 코드가 옛 analysis 를 재현하지 못한다"); [print("  ", b) for b in bad]; sys.exit(1)
    print("self-check OK — 옛 analysis 의 모든 블록을 각자의 행 집합(72/87/92)으로 정확히 재현")
    print(f"  (옛 원장 자체의 불일치: pooled module 0.658 은 87행 값; 같은 옛 손실로 92행 전부면 {A_all92['by_arm_family_pooled']['module']})")
    new_rows, deltas, guard = [], [], []
    for r in old["rows"]:
        d, n = drop_pct(rows, iso, r["model"], r["arm"])
        nr = {**r, "ndcg_drop_pct": d, "n_corpora": n}
        if r["model"] in REMEASURED:
            deltas.append((r["model"], r["arm"], r["ndcg_drop_pct"], d))
        elif abs(d - r["ndcg_drop_pct"]) > 0.0015:
            guard.append((r["model"], r["arm"], r["ndcg_drop_pct"], d, r["n_corpora"], n))
        elif n != r["n_corpora"]:
            # 옛 원장의 int3g32-attn-only 는 n_corpora=6 으로 적혀 있다(백필 때 attn-* 파일과 본 파일이 같은 셀을
            # 두 번 세었다). 값은 소수점까지 같으므로 개수 필드만 고친다 — 여기 기록해 둔다.
            nr["n_corpora_note"] = f"old ledger recorded n_corpora={r['n_corpora']} (double-counted attn-* merge); value identical"
        new_rows.append(nr)
    if guard:
        print("⛔ 회귀 가드: 무영향 모델의 손실이 옛 원장과 다르다"); [print("  ", g) for g in guard]; sys.exit(1)
    print(f"회귀 가드 OK — 무영향 3모델 {sum(1 for r in old['rows'] if r['model'] not in REMEASURED)} 행 일치")
    A = analysis(new_rows)
    A["retraction_note"] = old["analysis"].get("retraction_note")
    A["g32_attn_backfill_note"] = old["analysis"].get("g32_attn_backfill_note")
    out = {"date": "2026-09-14", "subject": "재구성 오차 ↔ NDCG 손실 (5 체크포인트·17팔) — E5·Gemma 손실을 prompt-contract 재측정으로 갱신",
           "provenance": {"recon_rel_err": f"{os.path.relpath(OLD_RECON, LEDGER)} 그대로 (가중치만의 양; promptfix 잡의 in-run 재계산과 max|Δ|=0)",
                          "ndcg_drop_pct": "paper_numbers.load() (STATUS canonical 행) + " + os.path.relpath(INT2_ISO, LEDGER),
                          "remeasured_models": list(REMEASURED), "unchanged_models_guard": "abs Δ ≤ 0.0015 on every row"},
           "rows": new_rows, "analysis": A, "int2_module_note": old.get("int2_module_note"),
           "analysis_rowset_note": ("모든 analysis 블록을 같은 92행(INT2 부위 팔 15·int3g32-attn-only 5 포함)으로 냈다. 옛 원장은 per_model/robustness 가 72행, "
                                    "pooled/excluding_collapsed/by_arm_family/int2_all15 가 87행(g32-attn 백필 전), within-bitwidth 만 92행 값이었다 — "
                                    f"본문의 'INT2 팔을 넣은 pooled module 0.658' 은 int3g32-attn-only 5행이 빠진 값이고, 같은 옛 손실로 92행이면 {A_all92['by_arm_family_pooled']['module']} 이다."),
           "old_rowset_analysis_92rows": {"by_arm_family_pooled": A_all92["by_arm_family_pooled"], "excluding_collapsed_pooled": A_all92["excluding_collapsed_pooled"]},
           "remeasurement_deltas": [{"model": m, "arm": a, "old_drop_pct": o, "new_drop_pct": d} for m, a, o, d in deltas]}
    json.dump(out, open(RECON_PATH, "w"), ensure_ascii=False, indent=1)
    O = old["analysis"]
    print("\n분석 옛 → 새")
    print(f"  pooled (excl collapse) uniform {O['by_arm_family_pooled']['uniform']} → {A['by_arm_family_pooled']['uniform']} · "
          f"module {O['by_arm_family_pooled']['module']} → {A['by_arm_family_pooled']['module']} · all {O['excluding_collapsed_pooled']} → {A['excluding_collapsed_pooled']}")
    for lab in BIT_OF.values():
        print(f"  within {lab:8s} {O['module_axis_within_bitwidth'][lab]['pearson']:+.3f} → {A['module_axis_within_bitwidth'][lab]['pearson']:+.3f} (n={A['module_axis_within_bitwidth'][lab]['n']})")
    print(f"  int2 all15 {O['int2_module_all15']['pearson']} → {A['int2_module_all15']['pearson']}")
    for m in REMEASURED:
        print(f"  {m}: family uniform/module/all {O['by_arm_family'][m]} → {A['by_arm_family'][m]}")
        print(f"      robustness uniform incl r={A['robustness'][m]['uniform_incl_collapse']['pearson']} ci={A['robustness'][m]['uniform_incl_collapse']['ci95']} "
              f"excl r={A['robustness'][m]['uniform_excl_collapse']['pearson']} ci={A['robustness'][m]['uniform_excl_collapse']['ci95']}")
    print("\n재측정 행 손실 변화 (>0.5pp 만):")
    for m, a, o, d in deltas:
        if abs(d - o) > 0.5:
            print(f"  {m:6s} {a:20s} {o:7.3f} → {d:7.3f}")
    print("→", INT2_ISO, "\n→", RECON_PATH)


if __name__ == "__main__":
    main()
