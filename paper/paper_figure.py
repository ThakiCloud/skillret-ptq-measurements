#!/usr/bin/env python3
"""학회 논문용 그림 정본 — 데이터 → Matplotlib → **벡터 PDF**.

왜 이 모듈이 있나
    같은 청색 계열 막대를 여러 계열에 쓰면 흑백 인쇄와 색각 이상 환경에서 계열이
    구분되지 않는다(2026-09-10 리뷰 지적). 색을 없애는 게 아니라 **색이 없어도
    읽히게** 만드는 것이 원칙이다: 색 + hatch + marker + linestyle 을 동시에 쓴다.

    포맷·글자크기·팔레트·hatch 순서를 **코드가 소유**한다([[sonnet-format-determinism]]).
    모델은 어떤 데이터를 어떤 종류로 그릴지만 정한다.

원칙 (전부 코드가 강제한다)
    1. 벡터 PDF 로만 저장한다. 래스터는 사진에만 쓴다.
    2. 계열 구분은 색 하나에 기대지 않는다 — hatch(막대) 또는 marker+linestyle(선).
    3. 그림 안에 제목을 넣지 않는다. 설명은 LaTeX caption 이 한다.
    4. 글자 최소 7pt (축라벨 8 · 눈금 7 · 범례 7).
    5. 실데이터/참조는 검정 계열, 합성은 유채색, 결합된 지표는 hatch 로 표시한다.
    6. 팔레트는 Okabe-Ito (색각 안전). rainbow/jet 금지.

쓰는 법
    from paper_figure import grouped_bar, line_ci, profile_lines
    grouped_bar(out="fig.pdf", categories=[...], series=[("name", [...]), ...],
                ylabel="Recovery (%)")

정본 게이트: scripts/skills/paper_figure_gate.py (래스터·글자크기·회색조 구분 검사)
"""
from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

# Okabe-Ito — 색각 안전 8색. 첫 자리는 실데이터/참조용 검정 계열이다.
OKABE_ITO = ["#000000", "#0072B2", "#D55E00", "#009E73",
             "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
# hatch 는 색이 사라져도 남는다. 빈 문자열(민무늬)을 첫 계열에 두어 대비를 만든다.
HATCH = ["", "///", "...", "xxx", "\\\\\\", "++", "oo"]
MARKER = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLE = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]

# 2단 논문 1칼럼 폭 ≈ 3.3in. 본문 폭이면 6.5in.
COL_W, COL_H = 3.3, 2.2
FULL_W, FULL_H = 6.5, 2.6


_TEX_LEAK = ("\\ ", "\\%", "\\textbf", "\\emph", "\\,", "--")


def _check_label(*labels):
    """⛔ matplotlib 은 LaTeX 이 아니다. `vs.\\ `, `\\%`, `--` 같은 표기를 그대로 찍는다
    (2026-09-10 실측: y축에 `vs.\\ standard input` 이 그대로 나갔다). 평문만 받는다."""
    for t in labels:
        if not t:
            continue
        for bad in _TEX_LEAK:
            if bad in t:
                raise ValueError(f"라벨에 LaTeX 표기가 있다 ({bad!r}): {t!r} — 평문으로 넘겨라")


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        "mathtext.fontset": "stix",
        # ⛔ 그림은 \includegraphics[width=...] 로 **축소돼서** 배치된다. 그리는 시점의
        #    7pt 가 배치 후 5pt 가 된다(2026-09-10 arXiv 프리플라이트가 잡았다).
        #    바닥을 9/10 으로 올려 0.7배까지 축소돼도 7pt 를 지킨다.
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,          # TrueType 임베드 — 학회 제출 요건
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.bbox": "standard",   # ⛔ tight 는 범례를 포함해 figsize 를 넘긴다
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    })



# x축 라벨 겹침 검사 결과 — _finish 가 PDF 메타데이터에 실어 게이트(V6)가 읽는다
TICKFIT = {"v": "n/a"}

def _finish(fig, ax, out: str | pathlib.Path, legend_below=True, ncol=None,
            n_series=0, distinct="none"):
    """⛔ 그림 안 제목 금지 · 벡터 PDF 저장 · 범례는 축 밖 아래."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("")                                   # caption 이 설명한다
    h, l = ax.get_legend_handles_labels()
    if l and legend_below:
        lg = ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                       ncol=ncol or min(len(l), 3), frameon=False, handlelength=2.0)
        # ⛔ 범례가 그림 폭을 넘으면 constrained_layout 이 **말없이 잘라낸다**
        #    (2026-09-10 실측: "independent referenc" 로 끝났다). 줄 수를 줄여 보고,
        #    그래도 안 들어가면 죽인다 — 잘린 범례를 내보내지 않는다.
        fig.canvas.draw()
        fw = fig.get_size_inches()[0] * fig.dpi
        for nc in (ncol or min(len(l), 3), 2, 1):
            lg.remove()
            lg = ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                           ncol=nc, frameon=False, handlelength=2.0)
            fig.canvas.draw()
            if lg.get_window_extent().width <= fw * 1.02:
                break
        else:
            raise ValueError(
                f"범례가 그림 폭을 넘는다 ({lg.get_window_extent().width:.0f}px > {fw:.0f}px) "
                f"— 라벨을 줄이거나 width 를 키워라. 라벨: {l}")
    p = pathlib.Path(out)
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"논문 그림은 벡터 PDF 로만 저장한다 (받은 값: {p.suffix})")
    p.parent.mkdir(parents=True, exist_ok=True)
    # ⛔ 게이트가 읽을 증거를 그림 자신에 박는다. hatch/marker 는 벡터 경로로 그려져
    #    PDF 연산자만 봐서는 "색 말고 다른 구분 수단이 있나"를 확실히 알 수 없다
    #    (2026-09-10: 헐거운 패턴 매치가 오탐해 검사가 통과만 했다).
    # ⛔ CreationDate 를 비우지 않으면 같은 입력이 매번 다른 바이트를 낸다 — 배포본의
    #    SHA256SUMS 가 그림을 다시 그리는 순간 깨진다(2026-09-13 clean clone 에서 실측).
    fig.savefig(p, metadata={"Subject": f"paper_figure:v1 series={n_series} distinct={distinct} tickfit={TICKFIT['v']}",
                             "Creator": "scripts/skills/paper_figure.py",
                             "CreationDate": None})
    plt.close(fig)
    return p


def grouped_bar(out, categories: Sequence[str], series: Sequence[tuple[str, Sequence[float]]],
                ylabel: str, horizontal: bool | None = None, width=None, height=None,
                highlight_first_black=True, log: bool = False):
    """그룹 막대. **계열마다 hatch 가 다르다** — 회색조에서도 구분된다.

    horizontal=None 이면 라벨 길이로 자동 판단한다(긴 라벨은 가로 막대가 읽기 쉽다).
    """
    _style()
    _check_label(ylabel, *[n for n, _ in series])
    n = len(series)
    if horizontal is None:
        horizontal = max(len(c) for c in categories) > 12 or len(categories) > 6
    fig, ax = plt.subplots(figsize=(width or COL_W, height or COL_H))
    span = 0.8 / n
    pos = list(range(len(categories)))
    for i, (name, vals) in enumerate(series):
        off = i * span - 0.4 + span / 2
        color = OKABE_ITO[0] if (i == 0 and highlight_first_black) else OKABE_ITO[i + 1]
        kw = dict(edgecolor="black", linewidth=0.6, hatch=HATCH[i % len(HATCH)],
                  facecolor=color if i else ("white" if highlight_first_black else color),
                  label=name)
        if horizontal:
            ax.barh([p + off for p in pos], list(vals), span, **kw)
        else:
            ax.bar([p + off for p in pos], list(vals), span, **kw)
    if horizontal:
        ax.set_yticks(pos); ax.set_yticklabels(categories); ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.xaxis.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    else:
        ax.set_xticks(pos); ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    if log:
        # ⛔ A log value axis must be announced in the caller's caption. Bars whose
        # values span two orders of magnitude are illegible on a linear axis, but
        # an unannounced log axis makes a small effect look large.
        axis_obj = ax.xaxis if horizontal else ax.yaxis
        (ax.set_xscale if horizontal else ax.set_yscale)("log")
        # ⛔ Default log ticks are powers of ten typeset with a SUPERSCRIPT, and the
        # superscript lands around 0.7x the tick size — under the 7pt floor even
        # when the tick itself is fine. Measured: a 8pt tick produced 5.6pt glyphs
        # and failed the font gate. Plain separated integers also read better for
        # a count axis.
        from matplotlib.ticker import FuncFormatter, LogLocator
        axis_obj.set_major_locator(LogLocator(base=10.0))
        axis_obj.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axis_obj.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    _fit_xticklabels(fig, ax)
    return _finish(fig, ax, out, n_series=n, distinct="hatch+color")


def _fit_xticklabels(fig, ax, pad_pt: float = 2.0) -> None:
    """⛔ x축 눈금 라벨이 서로 겹치면 **회전시킨다**.

    2026-09-11 실측: 'dialectness / region match / compr.' 이 3.4in 폭에서 겹쳐 찍혀
    `dialectn+region+matc+compr.` 처럼 보였다. **텍스트 레이어는 멀쩡해서** pdftotext 로도,
    글자크기 게이트로도 안 잡혔다 — 픽셀에서만 드러난다([[evaluator-must-act]]).
    렌더러가 보고한 실제 폭으로 판단한다(글자수 휴리스틱이 아니라).
    """
    fig.canvas.draw()
    labs = [t for t in ax.get_xticklabels() if t.get_text()]
    if len(labs) < 2:
        TICKFIT["v"] = "n/a"; return
    for rot, ha in ((0, "center"), (20, "right"), (35, "right"), (50, "right")):
        if rot:
            for t in labs:
                t.set_rotation(rot); t.set_ha(ha); t.set_rotation_mode("anchor")
        fig.canvas.draw()
        bb = sorted((t.get_window_extent() for t in labs), key=lambda b: b.x0)
        if all(bb[i].x1 + pad_pt <= bb[i + 1].x0 for i in range(len(bb) - 1)):
            TICKFIT["v"] = f"ok-rot{rot}"; return
    raise ValueError("x축 라벨이 어떤 회전에서도 겹친다 — 라벨을 줄이거나 폭을 늘려라: "
                     + " / ".join(t.get_text() for t in labs))



REF_LABEL_PT = 8.0


def _plain_log_axis(axis_obj, set_scale, base: float = 10.0) -> None:
    """로그축 눈금을 평문 정수로 찍는다.

    ⛔ 기본 로그 눈금은 **위첨자**로 지수를 쓰고, 위첨자는 눈금 크기의 0.7배쯤으로
    떨어진다. 눈금 자체가 8pt 여도 글자는 5.6pt 가 되어 글자크기 게이트에 걸린다
    (2026-09-13 실측: fig7 이 정확히 이걸로 V2 실패). `grouped_bar` 가 이미 같은
    처리를 갖고 있었는데 선 계열에는 없어서 되풀이됐다 — 한 곳으로 합친다.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator

    set_scale("log", base=base)
    axis_obj.set_major_locator(LogLocator(base=base))
    axis_obj.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis_obj.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _draw_refs(ax, vrefs, hrefs):
    """기준선. ⛔ 색이 아니라 **점선 + 주석**으로 구분한다 — 회색조에서도 남아야 하고,
    선만 그어두면 독자가 그 값이 무엇인지 알 방법이 없다([[paper-figure-standard]] 2).
    계열 색을 쓰지 않아 범례를 오염시키지도 않는다."""
    for ref in (vrefs or []):
        val, label = ref[0], ref[1]
        # 3번째 원소로 축분수 높이를 줄 수 있다. ⛔ 기본 위치(맨 위)가 계열 선과
        # 겹치는 그림이 있었다(fig6: 대조군 두 계열이 상단에 평평하게 깔린다).
        # 겹친 라벨은 게이트가 못 잡는다 — 픽셀에서만 보인다([[evaluator-must-act]]).
        frac = ref[2] if len(ref) > 2 else 1.0
        va = "top" if frac >= 0.99 else "center"
        ha, dx = ("left", 2) if len(ref) <= 3 else (ref[3], -3 if ref[3] == "right" else 3)
        ax.axvline(val, color="black", linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
        ax.annotate(label, xy=(val, frac), xycoords=("data", "axes fraction"),
                    xytext=(dx, -2 if va == "top" else 0), textcoords="offset points",
                    # ⛔ 8pt 이상. 게이트가 0.8배 축소 배치를 가정해 6pt 를 떨어뜨린다
                    # (실측 2026-09-13: 6pt 로 두 그림 모두 V2 실패).
                    ha=ha, va=va, fontsize=REF_LABEL_PT)
    for ref in (hrefs or []):
        val, label = ref[0], ref[1]
        frac = ref[2] if len(ref) > 2 else 1.0
        ax.axhline(val, color="black", linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
        ax.annotate(label, xy=(frac, val), xycoords=("axes fraction", "data"),
                    xytext=(-2 if frac >= 0.99 else 2, 2), textcoords="offset points",
                    ha="right" if frac >= 0.99 else "left", va="bottom",
                    fontsize=REF_LABEL_PT)


def line_ci(out, x: Sequence[float], series: Sequence[tuple[str, Sequence[float], Sequence[tuple[float, float]] | None]],
            xlabel: str, ylabel: str, width=None, height=None, zero_line=False,
            vrefs: Sequence[tuple[float, str]] | None = None,
            hrefs: Sequence[tuple[float, str]] | None = None,
            log: bool = False):
    """순서형/연속 변수용 선+점. CI 가 있으면 오차막대로 그린다.

    ⛔ MCC 처럼 **순서가 있는 연속 변수**는 막대가 아니라 이 함수를 쓴다.
    """
    _style()
    _check_label(xlabel, ylabel, *[it[0] for it in series])
    fig, ax = plt.subplots(figsize=(width or COL_W, height or COL_H))
    for i, item in enumerate(series):
        name, ys = item[0], list(item[1])
        cis = item[2] if len(item) > 2 else None
        color = OKABE_ITO[0] if i == 0 else OKABE_ITO[i + 1]
        yerr = None
        if cis:
            yerr = [[y - lo for y, (lo, _) in zip(ys, cis)],
                    [hi - y for y, (_, hi) in zip(ys, cis)]]
        ax.errorbar(list(x), ys, yerr=yerr, label=name, color=color,
                    marker=MARKER[i % len(MARKER)], markersize=3.5,
                    markerfacecolor="white" if i else color,
                    markeredgewidth=0.8, linestyle=LINESTYLE[i % len(LINESTYLE)],
                    capsize=2, elinewidth=0.7)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":", zorder=0)
    if log:
        _plain_log_axis(ax.xaxis, ax.set_xscale, base=2)
    _draw_refs(ax, vrefs, hrefs)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    _fit_xticklabels(fig, ax)
    return _finish(fig, ax, out, n_series=len(series), distinct="marker+linestyle+color")


def multi_x_lines(out, series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
                  xlabel: str, ylabel: str, width=None, height=None,
                  vrefs: Sequence[tuple[float, str]] | None = None,
                  hrefs: Sequence[tuple[float, str]] | None = None,
                  log_x: bool = False):
    """계열마다 **자기 x 격자**를 갖는 선+점. `(name, xs, ys)` 를 받는다.

    ⛔ `line_ci` 는 모든 계열이 같은 x 를 쓴다고 가정한다. 팔마다 측정 지점이 다른
    실험을 거기 억지로 넣으려면 공통 격자로 맞춰야 하고, 그건 재지 않은 점을
    **앞의 값으로 채워 넣는 것**(forward-fill)이라 데이터를 만들어내는 짓이다.
    각 계열을 자기가 실제로 측정한 x 에 그대로 찍는 게 정직하다.
    """
    _style()
    _check_label(xlabel, ylabel, *[it[0] for it in series])
    fig, ax = plt.subplots(figsize=(width or COL_W, height or COL_H))
    for i, (name, xs, ys) in enumerate(series):
        color = OKABE_ITO[0] if i == 0 else OKABE_ITO[i + 1]
        ax.plot(list(xs), list(ys), label=name, color=color,
                marker=MARKER[i % len(MARKER)], markersize=3.5,
                markerfacecolor="white" if i else color, markeredgewidth=0.8,
                linestyle=LINESTYLE[i % len(LINESTYLE)], linewidth=1.0)
    if log_x:
        _plain_log_axis(ax.xaxis, ax.set_xscale, base=2)
    _draw_refs(ax, vrefs, hrefs)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    _fit_xticklabels(fig, ax)
    return _finish(fig, ax, out, n_series=len(series),
                   distinct="marker+linestyle+color")


def profile_lines(out, categories: Sequence[str],
                  series: Sequence[tuple[str, Sequence[float]]],
                  ylabel: str, xlabel: str = "", width=None, height=None,
                  vrefs: Sequence[tuple[float, str]] | None = None,
                  hrefs: Sequence[tuple[float, str]] | None = None):
    """범주별 프로파일 비교 — 모양(shape)이 관심사일 때 막대보다 선이 낫다."""
    _style()
    _check_label(ylabel, xlabel, *[n for n, _ in series])
    fig, ax = plt.subplots(figsize=(width or COL_W, height or COL_H))
    xs = list(range(len(categories)))
    for i, (name, vals) in enumerate(series):
        color = OKABE_ITO[0] if i == 0 else OKABE_ITO[i + 1]
        ax.plot(xs, list(vals), label=name, color=color,
                marker=MARKER[i % len(MARKER)], markersize=4,
                markerfacecolor="white" if i else color, markeredgewidth=0.8,
                linestyle=LINESTYLE[i % len(LINESTYLE)])
    ax.set_xticks(xs); ax.set_xticklabels(categories)
    _draw_refs(ax, vrefs, hrefs)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    _fit_xticklabels(fig, ax)
    return _finish(fig, ax, out, n_series=len(series), distinct="marker+linestyle+color")
