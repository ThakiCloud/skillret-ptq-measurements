#!/usr/bin/env python3
"""원장 → booktabs .tex. ⛔ 표를 손으로 치지 않는다([[paper-figure-standard]])."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_numbers import (load, retention, recon, MODEL_LABEL, CORPUS_LABEL,
                           UNIFORM, MODULES_I4, MODULES_I3)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT, exist_ok=True)
ARM_TEX = {"int8/g16": "W8 g16", "int4/g16-asym": "W4 g16 asym",
           "int4/g16-sym": "W4 g16 sym", "int4/g128-asym": "W4 g128 asym",
           "int3/g16": "W3 g16", "int3/g32": "W3 g32",
           "int2/g16": "W2 g16", "ternary/g16": "Ternary"}
ORDER = ["qwen", "skillret06", "gemma", "bgem3", "e5"]


def w(name, s):
    open(os.path.join(OUT, name), "w").write(s)
    print("  wrote", name)


def t_protocol(rows):
    models = [m for m in ORDER if any(mm == m for mm, _ in rows)]
    corpora = sorted({c for _, c in rows})
    body = ""
    for m in models:
        cells = []
        for c in corpora:
            r = rows.get((m, c))
            cells.append(f"{r['fp16']['ndcg']:.2f}" if r and "fp16" in r else "--")
        body += f"{MODEL_LABEL.get(m,m)} & " + " & ".join(cells) + r" \\" + "\n"
    hdr = " & ".join(CORPUS_LABEL.get(c, c) for c in corpora)
    w("protocol.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Full-precision (FP32 weights) NDCG@10 for every model and corpus. These are the
baselines every retention figure in this paper is relative to; absolute values differ across
models partly because SkillRet is in-domain for the finetuned checkpoint only.}}
\\label{{tab:protocol}}
\\begin{{tabular}}{{l{'r'*len(corpora)}}}
\\toprule
Model & {hdr} \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")


def t_uniform(rows):
    models = [m for m in ORDER if any(mm == m for mm, _ in rows)]
    arms = [a for a in UNIFORM if any(a in r for r in rows.values())]
    body = ""
    for m in models:
        cells = []
        for a in arms:
            v = retention(rows, m, a)
            cells.append("--" if v is None else
                         (f"\\textbf{{{v:.1f}}}" if v < 50 else f"{v:.1f}"))
        body += f"{MODEL_LABEL.get(m,m)} & " + " & ".join(cells) + r" \\" + "\n"
    hdr = " & ".join(ARM_TEX.get(a, a) for a in arms)
    w("uniform.tex", f"""\\begin{{table}}[t]\\centering\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Retained NDCG@10 as a percentage of the full-precision score, averaged over the
three corpora. Weight-only quantization; group-wise affine unless marked \\emph{{sym}}.
Bold marks arms that lost more than half their retrieval quality. W8/g16 and W4/g16 are
approximately NDCG-neutral for every model; widening to g128 already costs E5 5.9 points of
retention and the fine-tuned checkpoint 3.3 --- which is not the same as harmless; see
Table~\\ref{{tab:turnover}}. The models separate at W3 and diverge at W2: EmbeddingGemma retains roughly
two thirds of its quality where the two decoder-lineage and XLM-R-lineage models retain
almost none. Ternary is reported for E5 only: the ternary and symmetric arms for the other
four checkpoints were measured but retracted after a defect was found in the symmetric
quantizer (Section~\\ref{{sec:repro}}), and were not re-measured. One surviving ternary arm
is not evidence about the other four.}}
\\label{{tab:uniform}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{l{'r'*len(arms)}}}
\\toprule
Model & {hdr} \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}}}\\end{{table}}""")


def t_modules(rows):
    """세 동작점 × 4모델 부위별 절제 + 상호작용 갭."""
    models = [m for m in ORDER if any(mm == m for mm, _ in rows)]
    groups = [("INT4 / g16", "int4/g16-asym",
               [("Embed.", "int4-embed-only"), ("Attn.", "int4-attn-only"),
                ("FFN", "int4-ffn-only")]),
              ("INT3 / g16", "int3/g16",
               [("Embed.", "int3-embed-only"), ("Attn.", "int3-attn-only"),
                ("FFN", "int3-ffn-only")]),
              ("INT3 / g32", "int3/g32",
               [("Embed.", "int3g32-embed-only"), ("Attn.", "int3g32-attn-only"),
                ("FFN", "int3g32-ffn-only")])]

    def delta(m, arm):
        if arm is None:
            return None
        v = [r[arm]["ndcg"] - r["fp16"]["ndcg"]
             for (mm, _), r in rows.items() if mm == m and arm in r]
        return sum(v) / len(v) if v else None

    body = ""
    for label, allarm, mods in groups:
        if not any(delta(m, a) is not None for m in models for _, a in mods):
            continue
        body += f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}} \\\\\n"
        for m in models:
            cells = []
            for _, a in mods:
                d_ = delta(m, a)
                cells.append("--" if d_ is None else f"{d_:+.2f}")
            tot = delta(m, allarm)
            parts = [delta(m, a) for _, a in mods if delta(m, a) is not None]
            # ⛔ 비율은 분모가 0 근처에서 폭발한다. 잔차 I = joint - Σparts 를 쓴다.
            #    그리고 세 부위가 다 있을 때만 계산한다 — 어텐션이 빠진 합은 Σparts 가 아니다.
            full = len(parts) == len(mods)
            resid = (tot - sum(parts)) if (tot is not None and full) else None
            cells.append("--" if tot is None else f"{tot:+.2f}")
            cells.append("--" if resid is None else f"{resid:+.2f}")
            body += (f"\\quad {MODEL_LABEL.get(m,m)} & " + " & ".join(cells)
                     + r" \\" + "\n")
    w("modules.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Change in NDCG@10 (points, relative to full precision, averaged over three
corpora) when \\emph{{only}} the named module is quantized, against quantizing everything.
Three things vary. At INT4/g16 isolated-module effects stay small --- at most 0.59 in the
averages shown here, and 1.01 on any individual model--corpus pair --- leaving little
sensitivity differential to allocate against. At INT3 sensitivity appears, but the ordering
is not shared: at g16 the feed-forward block is the most costly module for Qwen3, its
fine-tuned derivative and EmbeddingGemma, BGE-M3 reverses the ordering with a free FFN and
attention as the bottleneck, and E5 is tied to within 0.01 before becoming attention-dominant
at g32. And in every model
the parts do not add up to the whole. The last column is the interaction residual
$I=\\Delta_{{\\text{{joint}}}}-\\sum\\Delta_{{\\text{{module}}}}$, shown only where all three
single-module arms were measured; negative means joint quantization hurts more than the parts
predict. At INT4/g16 $I$ is small ($-0.15$ to $+0.24$); at INT3 it is substantial in every
model and at g16 it changes sign between them, so a bit allocator that
treats per-module sensitivity as an independent cost is wrong by a model-dependent amount in
a model-dependent direction. The
embedding table is nearly free everywhere, at every operating point.}}
\\label{{tab:modules}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Model & Embed.\\ only & Attn.\\ only & FFN only & All & $I$ \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")


def t_recon():
    d = recon()
    if not d:
        w("recon.tex", "% recon ledger 없음\n"); return
    A = d["analysis"]
    fam = A.get("by_arm_family", {})
    body = ""
    for k in ORDER:
        if k not in fam:
            continue
        f = fam[k]
        body += (f"{MODEL_LABEL.get(k,k)} & {f.get('uniform')} & {f.get('module')} & "
                 f"{f.get('all')} \\\\\n")
    wbw = A.get("module_axis_within_bitwidth", {})
    wrow = "".join(f"{k} & {v['n']} & {v['pearson']} \\\\\n" for k, v in sorted(wbw.items()))
    pooled = A.get("by_arm_family_pooled", {})
    i2 = [r for r in d["rows"] if r["arm"] == "int2/g16"]
    ex = "".join(f"{MODEL_LABEL.get(r['model'],r['model'])} & {r['recon_rel_err']:.4f} & "
                 f"{r['ndcg_drop_pct']:.1f} \\\\\n"
                 for r in sorted(i2, key=lambda x: x["model"]))
    w("recon.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Pearson correlation between parameter-weighted relative weight reconstruction
error and NDCG@10 loss, excluding arms that lost more than half their quality. Left: computed
within each model and split by arm family. Reconstruction error tracks damage well along the
\\emph{{uniform}} axis --- how hard the whole model was quantized --- and poorly along the
\\emph{{module}} axis --- which part was quantized, the axis a mixed-precision allocator
actually searches. Middle: the module axis at fixed operating points, which
removes bit-width severity as a confound and asks whether reconstruction error tracks module
damage consistently across the evaluated checkpoints. Model-specific evidence for the same
point is the ordering inversion in Table~\\ref{{tab:modules}}. Pooling module arms across bit widths
inflates the correlation to 0.658 by range extension, because severity then varies along with
module choice; within a bit width it never exceeds 0.42, and at INT4 it is approximately zero
($r=-0.008$). The INT2 module arms added in
Section~\\ref{{sec:int2mod}} sit in the same weak band as INT3 rather than improving it.
Right: the INT2 arm per model, where nearly equal reconstruction error produces very
different retrieval outcomes. $^{{\\ddagger}}$Pooled over all bit widths and inflated for the
reason above; the middle table is the figure to read. Every column drops arms losing more than
half their quality, which at INT2 removes three of fifteen module arms; relaxing that rule
moves the INT2 figure to 0.294 rather than improving it.}}
\\label{{tab:recon}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Model & Uniform arms & Module arms$^{{\\ddagger}}$ & All \\\\
\\midrule
{body}\\midrule
Pooled & {pooled.get('uniform')} & {pooled.get('module')} & {A.get('excluding_collapsed_pooled')} \\\\
\\bottomrule
\\end{{tabular}}\\quad
\\begin{{tabular}}{{lrr}}
\\toprule
Module axis & $n$ & $r$ \\\\
\\midrule
{wrow}\\bottomrule
\\end{{tabular}}\\quad
\\begin{{tabular}}{{lrr}}
\\toprule
INT2/g16 & Recon.\\ err. & NDCG loss (\\%) \\\\
\\midrule
{ex}\\bottomrule
\\end{{tabular}}\\end{{table}}""")


def t_paired(rows):
    """paired control — 파인튜닝 효과. ⛔ 판정은 OOD 코퍼스에서만."""
    beir = [c for c in ("scifact", "nfcorpus") if ("qwen", c) in rows]
    if not beir or ("skillret06", beir[0]) not in rows:
        w("paired.tex", "% paired control 데이터 없음\n"); return
    arms = [a for a in UNIFORM if all(a in rows[(m, c)] for m in ("qwen", "skillret06")
                                      for c in beir)]
    def ret(m, a):
        return sum(100 * rows[(m, c)][a]["ndcg"] / rows[(m, c)]["fp16"]["ndcg"]
                   for c in beir) / len(beir)
    body = ""
    for lab, m in [("Qwen3-Emb-0.6B (base)", "qwen"),
                   ("\\quad + SkillRet fine-tune", "skillret06")]:
        body += lab + " & " + " & ".join(f"{ret(m,a):.1f}" for a in arms) + r" \\" + "\n"
    body += r"\midrule" + "\n" + "Difference & " + " & ".join(
        f"{ret('skillret06',a)-ret('qwen',a):+.1f}" for a in arms) + r" \\" + "\n"
    hdr = " & ".join(ARM_TEX.get(a, a) for a in arms)
    w("paired.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Paired control. Retained NDCG@10 (\\% of full precision) for the base checkpoint and
for the same checkpoint fine-tuned on 127{{,}}190 in-domain pairs, averaged over SciFact and
NFCorpus --- both out-of-domain for both models, so the comparison is like-for-like. Task
fine-tuning moves quantization tolerance by at most 1.0 point in the two-corpus averages
shown here, and by at most 2.28 points on any individual model--corpus arm. In the averages
the movement is always toward greater robustness, but that is partly cancellation: at
INT3/g32 the fine-tune gains 2.28 points on SciFact and loses 2.08 on NFCorpus, which the
average reports as $+0.1$. Either way the base model already shows both the INT3 decline and
the INT2 collapse, so fine-tuning did not create them.}}
\\label{{tab:paired}}
\\begin{{tabular}}{{l{'r'*len(arms)}}}
\\toprule
Checkpoint & {hdr} \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")


def t_rank(rows=None):
    """순위 보존 — NDCG 가 무손실이라고 말하는 지점에서 순위가 얼마나 바뀌나."""
    import glob as _g
    files = sorted(_g.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ledger/sweep/rank-*.json")))
    if not files:
        w("rank.tex", "% 순위 지표 없음\n"); return
    ARMS = ["int8/g16", "int4/g16-asym", "int3/g16", "int2/g16"]
    agg = {}
    for f in files:
        d = json.load(open(f))
        tag = os.path.basename(f)[5:-5]
        m = tag.partition("-")[0]
        fp = [r for r in d["rows"] if r["arm"] == "fp16"][0]["ndcg@10"]
        for r in d["rows"]:
            if r["arm"] not in ARMS or "top10_overlap" not in r:
                continue
            k = (m, r["arm"])
            agg.setdefault(k, []).append(
                (100 * r["ndcg@10"] / fp, r["top10_overlap"], r["boundary_flip_rate"]))
    body = ""
    for m in ORDER:
        if not any(mm == m for mm, _ in agg):
            continue
        body += f"\\multicolumn{{4}}{{l}}{{\\emph{{{MODEL_LABEL.get(m,m)}}}}} \\\\\n"
        for a in ARMS:
            v = agg.get((m, a))
            if not v:
                continue
            n = len(v)
            body += (f"\\quad {ARM_TEX.get(a,a)} & {sum(x[0] for x in v)/n:.1f} & "
                     f"{sum(x[1] for x in v)/n:.3f} & {sum(x[2] for x in v)/n:.3f}"
                     + r" \\" + "\n")
    w("rank.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Retrieval quality against ranking stability, averaged over the corpora measured for
each model. Retained NDCG@10 is the percentage of the full-precision score; top-10 overlap is
the fraction of the full-precision top-10 still present; the flip rate is the fraction of
queries whose rank-10 document left the top-10 entirely. At INT4/g16 the metric barely moves
--- in two cells it reports a gain --- while 13\% to 22\% of the top-10 has been replaced
(11.6\% to 25.2\% across individual model--corpus cells) and the boundary document moves for
roughly half the queries. E5-base-v2 is absent from this table because the rank-preservation
arm was never measured for it; its eviction composition, which was
measured, is in Table~\\ref{{tab:turnover}} --- where what is displaced is counted against the
relevance judgements rather than inferred from the metric here.}}
\\label{{tab:rank}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Arm & Retained NDCG@10 (\\%) & Top-10 overlap & Rank-10 flip rate \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")


def t_ci(rows=None):
    """질의 단위 페어드 부트스트랩 CI — '크기가 작다'를 숫자로 보인다."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "ledger/control/2026-09-11-paired-ci-int4.json")
    if not os.path.exists(p):
        w("ci.tex", "% paired CI 원장 없음\n"); return
    d = json.load(open(p))
    ARMS = ["int4-embed-only", "int4-attn-only", "int4-ffn-only", "int4/g16-asym"]
    LBL = {"int4-embed-only": "Embed.\\ only", "int4-attn-only": "Attn.\\ only",
           "int4-ffn-only": "FFN only", "int4/g16-asym": "All"}
    by = {}
    for r in d["rows"]:
        by.setdefault((r["model"], r["arm"]), []).append(r)
    body = ""
    for m in ORDER:
        if not any(mm == m for mm, _ in by):
            continue
        body += f"\\multicolumn{{4}}{{l}}{{\\emph{{{MODEL_LABEL.get(m,m)}}}}} \\\\\n"
        for a in ARMS:
            v = by.get((m, a))
            if not v:
                continue
            worst = max(v, key=lambda r: abs(r["delta_pp"]))
            ncz = sum(1 for r in v if not r["covers_zero"])
            body += (f"\\quad {LBL[a]} & {worst['delta_pp']:+.2f} & "
                     f"$[{worst['ci95'][0]:+.2f}, {worst['ci95'][1]:+.2f}]$ & "
                     f"{ncz}/{len(v)}" + r" \\" + "\n")
    w("ci.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Per-query paired bootstrap on the INT4 arms. For each model we show the corpus with
the largest absolute effect, its 95\\% interval over 10{{,}}000 resamples of
$\\Delta\\text{{NDCG}}_q$, and how many of that model's three corpora have an interval
excluding zero. No single-module effect exceeds 0.59 in the three-corpus averages, or 1.01
on any individual model--corpus pair. Ten of the forty-five
module cells exclude zero, which is what 4{{,}}392 queries buys in resolution rather than
evidence of a usable sensitivity gap --- and three of those ten are positive.}}
\\label{{tab:ci}}
\\begin{{tabular}}{{lrcr}}
\\toprule
Arm & Worst $\\Delta$ (pts) & 95\\% CI & CI excludes 0 \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")


def t_students(rows=None):
    """증류 학생 vs 일반 모델 — 도메인 안/밖."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "ledger/control/2026-09-12-student-ood.json")
    if not os.path.exists(p):
        w("students.tex", "% student 원장 없음\n"); return
    d = json.load(open(p))
    CORP = ["skillret", "scifact", "nfcorpus"]
    LBL = {"edge22m": "22M student", "edge109m": "109M student"}
    body = ""
    for m in ("edge22m", "edge109m"):
        st = d["students"].get(m)
        if not st:
            continue
        for a, nm in (("fp16", "FP16"), ("int4/g16-asym", "INT4/g16"), ("int3/g16", "INT3/g16")):
            mb = d["sizes_mb"].get(m, {}).get(a)
            cells = " & ".join(f"{st[c][a]:.2f}" if c in st and a in st[c] else "--" for c in CORP)
            body += (f"{LBL[m]}, {nm} & {mb:.1f} & " if mb else f"{LBL[m]}, {nm} & -- & ") + cells + r" \\" + "\n"
    body += r"\midrule" + "\n"
    # ⛔ equal-byte 주장을 표에서 재현할 수 있어야 한다: 교사를 극단 PTQ 로 눌렀을 때의
    #    packed 크기와 점수를 같은 표에 노출한다(2026-09-12 리뷰).
    tsz = d.get("sizes_mb", {}).get("skillret06", {}).get("int3/g32")
    if rows is not None and tsz:
        tc = []
        for c in CORP:
            r = rows.get(("skillret06", c))
            tc.append(f"{r['int3/g32']['ndcg']:.2f}" if r and "int3/g32" in r else "--")
        if any(x != "--" for x in tc):
            body += (f"SkillRet-0.6B teacher, INT3/g32 & {tsz:.1f} & " + " & ".join(tc)
                     + r" \\" + "\n")
    # 교사는 Table 1 과 같은 출처(rows)에서 읽는다 — 손으로 적지 않는다.
    if rows:
        tcell = []
        for c in CORP:
            r = rows.get(("skillret06", c))
            tcell.append(f"{r['fp16']['ndcg']:.2f}" if r and "fp16" in r else "--")
        if any(x != "--" for x in tcell):
            body += ("SkillRet-0.6B teacher$^{\\dagger}$ & 1191.6 & " + " & ".join(tcell)
                     + r" \\" + "\n")
    for m, nm in (("qwen", "Qwen3-Emb-0.6B"), ("gemma", "EmbeddingGemma-300M"),
                  ("bgem3", "BGE-M3"), ("e5", "E5-base-v2")):
        b = d["general_baselines_fp16"].get(m)
        if not b:
            continue
        body += (f"{nm}, FP32 & -- & "
                 + " & ".join(f"{b[c]:.2f}" if c in b else "--" for c in CORP) + r" \\" + "\n")
    # ⛔ 캡션 크기도 원장에서 읽는다 — 손으로 적었더니 표만 갱신되고 캡션은 옛 값으로 남았다.
    s109 = d["sizes_mb"]["edge109m"]["int3/g16"]
    s06 = d["sizes_mb"]["skillret06"]["int3/g32"]
    w("students.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Distilled students against general-purpose embedders, same gate and same corpora.
SkillRet is the domain the students were distilled on; SciFact and NFCorpus are not. Inside
the domain the 109M student at INT3 holds 78.04 in {s109:.1f}\\,MB. Outside it the same checkpoint
scores 7.79 on NFCorpus, against 31 to 39 for models that were never specialised. The distilled student strictly dominates the extreme-PTQ arm on the
size--quality frontier --- {s109:.1f}\\,MB at 78.04 against {s06:.1f}\\,MB at 64.46 --- but only within
the task it was trained for. Three precisions are distinct here and should not be conflated: every score in
this paper is computed in FP32 arithmetic; the teacher's reference storage artifact is its
1191.6\,MB BF16 checkpoint as distributed; and student sizes are measured on-disk artifacts:
serialized FP16 for the FP16 rows, packed quantized files for the INT rows. The lower block holds the 0.6B teacher, which is specialised for SkillRet, and
four general-purpose references, which are not; size is left blank for the four, which are
included for their scores rather than as size comparisons. $^{{\\dagger}}$BF16 storage
artifact as distributed; evaluation uses FP32 arithmetic like every other row.}}
\\label{{tab:students}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Model & Size (MB) & SkillRet & SciFact & NFCorpus \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")



def t_int2mod(rows=None):
    """절벽에서의 부위별 격리 — 어휘/토크나이저 가설의 직접 검정."""
    import statistics as _st
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "ledger/sweep/2026-09-12-int2-module-isolation.json")
    if not os.path.exists(p):
        w("int2mod.tex", "% int2 부위별 원장 없음\n"); return
    d = json.load(open(p))
    ARMS = ["int2/g16", "int2-embed-only", "int2-attn-only", "int2-ffn-only"]
    body = ""
    for m in ORDER:
        cells = d["cells"].get(m)
        if not cells:
            continue
        avg = {}
        for a in ARMS:
            v = [r["retained_pct"] for c in cells for r in c["rows"] if r["arm"] == a]
            if v:
                avg[a] = _st.mean(v)
        # ⛔ All 열은 Table 2 와 같은 팔이다. 두 표가 한 팔에 두 값을 실으면 리뷰어가 먼저
        #    본다 — 이 열은 Table 2 와 같은 출처(rows)에서 읽어 값이 하나만 존재하게 한다.
        #    (이 원장의 재측정값과는 0.00~0.05pp 차이였고, 그 사실은 캡션이 적는다.)
        if rows is not None:
            rr = retention(rows, m, "int2/g16")
            if rr is not None:
                avg["int2/g16"] = rr
        body += (f"{MODEL_LABEL.get(m, m)} & "
                 + " & ".join(f"{avg[a]:.1f}" if a in avg else "--" for a in ARMS)
                 + r" \\" + "\n")
    w("int2mod.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Retained NDCG@10 (\\% of full precision, averaged over the three corpora) when only
the named module is quantized to INT2/g16, against quantizing everything. Vocabulary size
directly determines the number of embedding rows, so this arm tests whether fragility of the
vocabulary-dependent embedding table explains the cliff; it does not test tokenization
behaviour itself. It does not explain it: embedding-only INT2 leaves 98.0\\% to 100.5\\% in the
per-model averages (95.6\\% to 101.3\\% across cells), including in the three checkpoints that
retain under 2\\% when everything is quantized, and BGE-M3 reads 100.5\\%. The decomposition of
the failure is itself family-dependent: BGE-M3 is close to a single-module bottleneck, since
quantizing its attention alone leaves almost what quantizing everything leaves; the Qwen
lineage is feed-forward-dominant with a large residual on top; and for EmbeddingGemma and
E5-base-v2 no isolated module comes near the joint loss at all. The All column is the same arm as in
Table~\\ref{{tab:uniform}} and is read from the same source; the run reported here
re-measured it and agreed to within 0.05 points of retained quality.}}
\\label{{tab:int2mod}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Model & All & Embed.\\ only & Attn.\\ only & FFN only \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")

def t_turnover(rows=None):
    """qrels 로 센 교체 구성 — 추론이 아니라 측정."""
    import glob as _g
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(_g.glob(os.path.join(root, "ledger/sweep/turn-*.json")))
    if not files:
        w("turnover.tex", "% turnover 원장 없음\n"); return
    ARMS = ["int8/g16", "int4/g16-asym", "int3/g16", "int2/g16"]
    agg = {}
    for f in files:
        d = json.load(open(f))
        m = os.path.basename(f)[5:-5].partition("-")[0]
        fp = [r for r in d["rows"] if r["arm"] == "fp16"][0]["ndcg@10"]
        for r in d["rows"]:
            if r["arm"] not in ARMS or "evicted_total" not in r:
                continue
            agg.setdefault((m, r["arm"]), []).append(
                (100 * r["ndcg@10"] / fp, r["evicted_total"], r["evicted_relevant"],
                 r["delta_recall@10"], r["relevant_only_top10_overlap"]))
    body = ""
    for m in ORDER:
        if not any(mm == m for mm, _ in agg):
            continue
        body += f"\\multicolumn{{5}}{{l}}{{\\emph{{{MODEL_LABEL.get(m,m)}}}}} \\\\\n"
        for a in ARMS:
            v = agg.get((m, a))
            if not v:
                continue
            n = len(v)
            ev = sum(x[1] for x in v); evr = sum(x[2] for x in v)
            body += (f"\\quad {ARM_TEX.get(a,a)} & {sum(x[0] for x in v)/n:.1f} & "
                     f"{ev} & {100*evr/max(ev,1):.1f}\\% & "
                     f"{sum(x[3] for x in v)/n:+.2f} & {sum(x[4] for x in v)/n:.3f}"
                     + r" \\" + "\n")
    w("turnover.tex", f"""\\begin{{table}}[t]\\centering\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{What quantization actually displaces, counted against the relevance judgements
rather than inferred from NDCG. ``Evicted'' counts documents leaving the full-precision
top-10, summed over each model's corpora; the next column is the relevant share of those.
\\textbf{{Read the last column, not the eviction share.}} On a hard corpus everything churns,
so the relevant share rises for healthy and broken models alike --- at INT2 on NFCorpus it is
22\\% for EmbeddingGemma, which keeps two thirds of its NDCG, and 23--26\\% for BGE-M3 and
Qwen, which keep none. What separates them is how much of the gold set survives: a
relevant-only overlap of 0.500 against 0.000. At INT4/g16 that overlap is 0.944 to 0.970 in the per-model averages shown here, and 0.899
to 1.000 across individual model--corpus cells
and recall moves by less than a point, which is the sense in which INT4 leaves the gold set
alone while replacing a sixth of the list.}}
\\label{{tab:turnover}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Arm & Ret.\\ NDCG (\\%) & Evicted & \\ldots relevant & $\\Delta$Recall@10 & Rel.-only overlap \\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}\\end{{table}}""")


if __name__ == "__main__":
    rows = load()
    t_protocol(rows); t_uniform(rows); t_modules(rows); t_recon(); t_paired(rows); t_rank(); t_ci(); t_students(rows); t_turnover(); t_int2mod(rows)
    print("표 생성 완료 —", len(rows), "개 (모델,코퍼스) 조합")
