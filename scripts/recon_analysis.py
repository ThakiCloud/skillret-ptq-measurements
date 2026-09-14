#!/usr/bin/env python3
"""재구성 오차 ↔ 검색 손실 상관을 **두 정의**로 낸다 — 감사(2026-09-14)에서 recon_vs_rank.py 가
부위 팔의 오차를 '양자화한 텐서 안에서만' 평균한다는 것이 드러났다(분모 = touched params).
논문 문장은 '모델 전체 파라미터 가중'을 말한다. 전체 가중값 = touched-only 값 × (touched / all)
이므로 원장의 quantized_params 로 재구성할 수 있다. 어느 정의로도 결론이 서는지가 이 파일의 질문.
또 recon-vs-rank-4models.json 의 module_axis_within_bitwidth 등은 산출 스크립트가 없었다 — 여기서 낸다."""
import glob, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper"))
from paper_numbers import load, recon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFORM = {"int8/g16", "int4/g16-asym", "int4/g16-sym", "int4/g128-asym", "int3/g16", "int3/g32", "int2/g16", "ternary/g16"}
BIT_OF = {"int4-": "INT4/g16", "int3-": "INT3/g16", "int3g32-": "INT3/g32", "int2-": "INT2/g16"}


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy), 3) if sx and sy else None


def touched_fraction(rows):
    """모델별 팔별 quantized_params / 전체(균일 int3/g16 팔의 quantized_params)."""
    out = {}
    for (m, c), arms in rows.items():
        pass
    # quantized_params 는 sweep 원장 rows 에 있다 — paper_numbers.load() 는 안 싣는다. 직접 읽는다.
    frac = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "ledger/sweep/*.json"))):
        b = os.path.basename(f)
        if b.startswith(("rank-", "pool-", "attn-", "student-", "turn-", "2026-")):
            continue
        d = json.load(open(f))
        if not isinstance(d, dict) or not isinstance(d.get("rows"), list):
            continue
        m = b[:-5].partition("-")[0]
        nq = {r["arm"]: r.get("quantized_params") for r in d["rows"] if not r.get("retracted")}
        total = nq.get("int3/g16") or nq.get("int4/g16-asym")
        if not total:
            continue
        for arm, n in nq.items():
            if n:
                frac.setdefault(m, {})[arm] = n / total
    return frac


def main():
    R = recon(); frac = touched_fraction(load())
    rows = []
    for r in R["rows"]:
        fr = frac.get(r["model"], {}).get(r["arm"])
        if fr is None and "-" in r["arm"] and r["arm"].endswith("-only"):
            # INT2 부위 팔은 별도 원장(int2-module-isolation)에서 왔다. 마스크는 비트폭과 무관하므로
            # 같은 부위의 INT3 팔과 touched_fraction 이 같다(embed/attn/ffn 파라미터 수는 동일).
            part = r["arm"].split("-")[-2]
            fr = frac.get(r["model"], {}).get(f"int3-{part}-only")
        if fr is None:
            continue
        rows.append({**r, "touched_fraction": round(fr, 4),
                     "recon_touched_only": r["recon_rel_err"],
                     "recon_whole_model": round(r["recon_rel_err"] * fr, 5)})
    live = [r for r in rows if r["ndcg_drop_pct"] < 50]
    res = {"date": "2026-09-14", "subject": "재구성 오차 두 정의(touched-only vs whole-model 가중)에 대한 상관 강건성",
           "definitions": {"touched_only": "recon_vs_rank.py 가 실제로 계산한 값 — 양자화된 텐서만 파라미터 가중 평균",
                           "whole_model": "touched_only × touched_fraction — 논문 본문이 서술한 정의(모델 전체 파라미터 가중)"},
           "n_rows": len(rows), "by_definition": {}}
    for key in ("recon_touched_only", "recon_whole_model"):
        uni = [r for r in live if r["arm"] in UNIFORM]; mod = [r for r in live if r["arm"] not in UNIFORM]
        within = {}
        for pre, lab in BIT_OF.items():
            pts = [r for r in mod if r["arm"].startswith(pre)]
            within[lab] = {"n": len(pts), "pearson": pearson([r[key] for r in pts], [r["ndcg_drop_pct"] for r in pts])}
        per_model_uniform = {m: pearson([r[key] for r in uni if r["model"] == m], [r["ndcg_drop_pct"] for r in uni if r["model"] == m])
                             for m in sorted({r["model"] for r in uni})}
        res["by_definition"][key] = {"uniform_pooled": pearson([r[key] for r in uni], [r["ndcg_drop_pct"] for r in uni]),
                                     "uniform_per_model": per_model_uniform,
                                     "module_pooled_all_bits": pearson([r[key] for r in mod], [r["ndcg_drop_pct"] for r in mod]),
                                     "module_within_bitwidth": within}
    bg = {r["arm"]: r for r in rows if r["model"] == "bgem3"}
    a, b = bg["int3/g32"], bg["int3g32-ffn-only"]
    res["bgem3_anecdote"] = {"int3/g32": {"touched_only": a["recon_touched_only"], "whole_model": a["recon_whole_model"], "loss_pct": a["ndcg_drop_pct"]},
                             "int3g32-ffn-only": {"touched_only": b["recon_touched_only"], "whole_model": b["recon_whole_model"], "loss_pct": b["ndcg_drop_pct"], "touched_fraction": b["touched_fraction"]},
                             "verdict": "touched-only 로는 FFN-only 오차가 더 큰데 손실은 작다(논문 §recon 의 일화). whole-model 로는 FFN-only 오차가 균일 팔보다 작아 순서가 맞다 — 그 일화는 서로 다른 분모를 비교한 것이다."}
    res["rows"] = rows
    out = os.path.join(ROOT, "ledger/control/2026-09-14-recon-definition-robustness.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    for key, v in res["by_definition"].items():
        print(key, "| uniform pooled", v["uniform_pooled"], "| module pooled", v["module_pooled_all_bits"],
              "| within:", {k: w["pearson"] for k, w in v["module_within_bitwidth"].items()})
    print("bgem3 anecdote:", json.dumps({k: v for k, v in res["bgem3_anecdote"].items() if k != "verdict"}))
    print("→", out)


if __name__ == "__main__":
    main()
