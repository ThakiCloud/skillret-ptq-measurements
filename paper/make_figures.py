#!/usr/bin/env python3
"""원장 → 벡터 PDF. 정본 모듈 paper_figure.py 를 쓴다([[paper-figure-standard]])."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_numbers import load, retention, recon, MODEL_LABEL, flip_curves
from paper_figure import profile_lines, line_ci

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
ORDER = ["qwen", "skillret06", "gemma", "bgem3", "e5"]
# ⛔ 라벨은 평문 — matplotlib 은 LaTeX 이 아니라 \% 를 그대로 찍는다
# ⛔ 대칭·삼진 팔은 fp16 스케일 언더플로 버그로 철회됐다(재측정 중) — 사다리에서 뺀다.
LADDER = [("int8/g16", "W8 g16"), ("int4/g16-asym", "W4 g16"),
          ("int4/g128-asym", "W4 g128"), ("int3/g16", "W3 g16"),
          ("int3/g32", "W3 g32"), ("int2/g16", "W2 g16")]


def fig_ladder(rows):
    cats = [lbl for _, lbl in LADDER]
    series = []
    for m in ORDER:
        if not any(mm == m for mm, _ in rows):
            continue
        ys = [retention(rows, m, a) for a, _ in LADDER]
        if any(y is None for y in ys):
            continue
        series.append((MODEL_LABEL[m].split(" (")[0], ys))
    if not series:
        print("  ladder: 데이터 부족"); return
    # ⛔ 정본 모듈 기본 폭은 3.3in(2단 조판)이다. 이 논문은 단단 6.5in 이라
    #    그대로 쓰면 긴 y축 라벨이 축을 0폭으로 밀어낸다(실측: 그림이 통째로 깨졌다).
    profile_lines(os.path.join(FIG, "fig1_ladder.pdf"), cats, series,
                  ylabel="Retained NDCG@10 (percent)",
                  xlabel="Weight quantization setting", width=5.8, height=3.0)
    print("  fig1_ladder.pdf", len(series), "계열")


def fig_recon():
    d = recon()
    if not d:
        print("  recon: 원장 없음"); return
    import matplotlib.pyplot as plt
    import paper_figure as pf
    pf._style()
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    UNIFORM = {"int8/g16", "int4/g16-asym", "int4/g16-sym", "int4/g128-asym",
               "int3/g16", "int3/g32", "int2/g16", "ternary/g16"}
    n = 0
    for i, m in enumerate([x for x in ORDER if any(r["model"] == x for r in d["rows"])]):
        col = pf.OKABE_ITO[0] if i == 0 else pf.OKABE_ITO[i + 1]
        mk = pf.MARKER[i % len(pf.MARKER)]
        for fam, filled in (("uniform", True), ("module", False)):
            pts = [r for r in d["rows"] if r["model"] == m
                   and ((r["arm"] in UNIFORM) == (fam == "uniform"))]
            if not pts:
                continue
            ax.scatter([r["recon_rel_err"] for r in pts],
                       [r["ndcg_drop_pct"] for r in pts],
                       marker=mk, s=22, linewidths=0.9, color=col,
                       facecolors=col if filled else "none",
                       label=MODEL_LABEL[m].split(" (")[0] if fam == "uniform" else None)
            n += 1
    ax.set_xlabel("Relative weight reconstruction error")
    ax.set_ylabel("NDCG@10 loss (percent)")
    ax.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    print("  fig2_recon.pdf", n, "군")
    return pf._finish(fig, ax, os.path.join(FIG, "fig2_recon.pdf"),
                      n_series=n, distinct="marker+fill+color")


def fig_flip(arm="int3/g16"):
    """Fig 3 — 교사 경계 마진 십분위 대 10위 뒤집힘률."""
    cur = flip_curves(arm)
    if not cur:
        print("  flip: 순위 지표 없음"); return
    series = []
    for m in ORDER:
        if m in cur:
            series.append((MODEL_LABEL[m].split(" (")[0], [100 * y for y in cur[m]], None))
    if not series:
        print("  flip: 모델 매칭 없음"); return
    line_ci(os.path.join(FIG, "fig3_flip.pdf"), list(range(1, 11)), series,
            xlabel="Full-precision boundary-margin decile (1 = smallest margin)",
            ylabel="Rank-10 flip rate (percent)", width=5.8, height=3.0)
    print("  fig3_flip.pdf", len(series), "계열")


if __name__ == "__main__":
    rows = load()
    fig_ladder(rows); fig_recon(); fig_flip()
