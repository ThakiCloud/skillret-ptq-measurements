# Reproducing a number

Every number in the paper is read from `ledger/`. Nothing is typed into the text.

## Regenerate every table and figure

```bash
pip install -r requirements.txt
python paper/make_tables.py          # every table, regenerated from the ledger
python paper/make_figures.py         # every figure, same source
python paper/duplicate_arm_gate.py   # fails if one arm carries two different values,
                                     #   table-to-table or table-to-prose
python paper/latex_escape_gate.py    # fails on escapes mangled by the generators
python paper/stale_prose_gate.py     # fails if a canonical ledger's prose still carries
                                     #   a superseded size or the old byte-matched framing
```

## Re-measure one arm from scratch

First materialise the pinned dataset revision into the three JSONL files the sweep
expects — the Hub head is mutable, so this step is what makes the run reproducible:

```bash
python scripts/materialize_skillret.py --dst /tmp/skillret
#   data_queries_test.jsonl    4392 lines
#   data_skills_test.jsonl     6006 lines
#   data_qrels_test.jsonl      7187 lines
```

```bash
python scripts/quant_sweep.py \
    --model Qwen/Qwen3-Embedding-0.6B \
    --corpus skillret:/tmp/skillret \
    --arms "fp16,int3/g16,int3-attn-only" \
    --out /tmp/rerun.json
```

Pooling and the query prefix come from the checkpoint's own config; see
`manifest/models.yaml` for what each one resolves to. The evaluator refuses to run
without a resolved prefix, because a mismatch between the training prefix and the
evaluation prefix once produced a result in which quantization beat full precision.

`--corpus` takes `skillret:<root>` or `beir:<name>`. `--arms` is a comma-separated
subset of the names in `ARMS` at the top of the script; omit it to run all of them.

Module isolation arms (`int2-embed-only`, `int2-attn-only`, `int2-ffn-only` and
their INT3 and INT4 counterparts) select tensors by name through `part_of()` in the
same script; nothing else about the pipeline changes between a joint arm and an
isolated one.

## Pack a checkpoint for real

Sizes in the paper are file sizes, not arithmetic. `build_release.py` writes the
packed artifact and reports predicted against measured:

```bash
python scripts/build_release.py --src <checkpoint> --dst /tmp/packed --bits 3 --group 32
```

`artifacts.csv` is the byte provenance for every size in the paper: the file name, its byte
count, its SHA-256 and the model revision it was packed from. It is regenerated, not typed:

```bash
SKILLRET_EDGE22M=... SKILLRET_EDGE109M=... SKILLRET_06B=... \
    python scripts/artifact_ledger.py --out artifacts.csv
```

`size_bytes` is canonical there; `size_mb_decimal` is derived from it. The arithmetic
prediction runs 0.04 to 0.44% above the packed file, so the two are close but not
interchangeable — the measured byte count is what the paper reports.

Sizes in `artifacts.csv` are decimal MB (bytes / 1e6). `du -sm` reports MiB and
will not match.

## Ledger status

Not every file under `ledger/` is citable. Two mechanisms keep the withdrawn ones out of
the tables. File-level: `ledger/STATUS.yaml` marks a whole file `canonical`, `superseded`
or `exploratory`, and `paper_numbers.py` loads only canonical files. Row-level: an
individual arm inside a canonical file carries `"retracted": true` (plus a reason), and the
loader drops that row. The records stay in place — see `RETRACTIONS.md` for why each one
was withdrawn and `grep -rl '"retracted": true' ledger/` to enumerate them.

Two sweep directories feed the same loader: `ledger/sweep/` (the 2026-09-10/12 runs) and
`ledger/sweep-promptfix/` (E5-base-v2 and EmbeddingGemma re-measured on 2026-09-14 under their own
prompt contracts, see below). `paper_numbers.sweep_files()` reads both with the same STATUS filter;
the pre-fix E5/Gemma files in `sweep/` are `superseded(prompt-contract)`, and the three aggregates
derived from them (`2026-09-12-int2-module-isolation`, `2026-09-11-recon-vs-rank-4models`,
`2026-09-11-paired-ci-int4`) are `superseded(prompt-contract-partial)` with 2026-09-14 rebuilds.
Every consumer (`recon_analysis.py`, `bh_fdr.py`, the table generators) goes through that one
function — a private glob would resurrect superseded rows.

## Multiple-comparison check (INT4 module cells)

The sentence "ten of forty-five cells exclude zero, four of them positive, eight survive
Benjamini--Hochberg at q=0.05, and the largest effect is unchanged at 1.01" is recomputed from the
raw per-query scores:

```bash
python scripts/bh_fdr.py     # prints cells=45 ci_excludes_zero=10 (positive 4) bh_significant=8 largest=-1.01
```

It writes `ledger/control/2026-09-14-bh-fdr-int4.json` with per-cell two-sided bootstrap p-values
and `ledger/control/2026-09-14-paired-ci-int4.json` (the 60-cell paired-CI table, 46 covering zero).
On the pre-fix E5/Gemma rows the same script printed `bh_significant=6` and three positive cells;
the arXiv v1 text carries those values.

## Arithmetic size prediction vs packed bytes

```bash
python scripts/size_gap.py --artifacts artifacts.csv   # shapes from Hub safetensors metadata, no weight download
```

Recomputed on 2026-09-14: the prediction runs +0.04% to +0.45% above the packed file over the
seven quantized artifacts (the arXiv v1 text says 0.44%; the 109M int3/g32 cell is 0.447%).
Output: `ledger/control/2026-09-14-size-prediction-gap.json`.

## Reconstruction-error denominator (audit 2026-09-14)

`scripts/recon_vs_rank.py` averages the per-tensor relative error over the **quantized tensors
only** (`den` accumulates touched parameters). For a module arm that is the module's own
error, not a whole-model figure; the arXiv v1 text describes whole-model parameter weighting.
`scripts/recon_analysis.py` recomputes every correlation under both definitions
(whole-model = touched-only x touched fraction, from `quantized_params` in the sweep rows):

```bash
python scripts/recon_analysis.py   # writes ledger/control/2026-09-14-recon-definition-robustness.json
```

Both definitions leave the module axis weak (within-bit Pearson r: touched-only
-0.043 / 0.350 / 0.415 / 0.412; whole-model -0.272 / -0.151 / -0.150 / -0.061 at
INT4/g16, INT3/g16, INT3/g32, INT2/g16) and the uniform axis strong (0.875 either way).
On the pre-fix rows (arXiv v1) the same four read -0.008 / 0.367 / 0.408 / 0.415 and
-0.346 / -0.088 / -0.089 / -0.078. The
v1 anecdote comparing BGE-M3 `int3/g32` (0.159) with `int3g32-ffn-only` (0.174) compares two
denominators; whole-model the FFN-only arm reads 0.062 and the ordering is unremarkable. That
sentence is withdrawn in the errata.

Also from the same audit: `quantize()` walks only the transformer; EmbeddingGemma's two
sentence-transformers `Dense` layers (4.72M parameters) stayed FP32 in every v1 arm. The
uniform arm quantizes `position_embeddings` (BGE-M3 1.48%, E5 0.36% of parameters) that no
module arm touches, so the interaction residual for those two models includes that leftover.
E5-base-v2 ran with no query/document prefix and EmbeddingGemma without its document prefix
(`quant_sweep.py` now resolves both and refuses unknown prompt contracts). NFCorpus qrels were
binarized (576 of 12,334 judgments are graded 2). The prompt and Dense cases were re-measured;
see the next section.

## Prompt-contract re-measurement (E5-base-v2, EmbeddingGemma — 2026-09-14)

`ledger/sweep-promptfix/` holds the 21-arm sweep of both checkpoints on all three corpora under
their own prompt contracts (`query: `/`passage: ` for E5; `task: search result | query: ` and
`title: none | text: ` for EmbeddingGemma). Each file records `query_prompt`, `doc_prompt`,
`prompt_source` and `include_dense`. `ledger/control/2026-09-14-promptfix-gemma-dense-*.json` is
the control with EmbeddingGemma's two sentence-transformers Dense layers quantized as well:
INT3/g16 retention moves 0.1 points and INT2 retention 65.9% -> 64.8%, so the FP32 output path
does not explain the INT2 survival.

```bash
python scripts/promote_promptfix.py   # rebuilds the INT2-isolation and recon-vs-rank aggregates
python scripts/recon_analysis.py      # both denominators on the rebuilt rows
python scripts/bh_fdr.py              # BH + paired CI on the canonical per-query scores
```

`promote_promptfix.py` keeps the reconstruction errors of the v1 ledger (a weights-only quantity;
the in-job recomputation in `control/2026-09-14-recon-promptfix-inrun.json` agrees to every
decimal), recomputes only the retrieval loss, and refuses to write if any row of the three
unaffected checkpoints changes. Its self-check first reproduces every analysis block of the
old ledger on the row set that block was actually computed from: the old ledger mixed three
generations (72 rows for per-model/robustness, 87 for the pooled and per-family figures,
92 for the within-bit-width figures), so its pooled module correlation of 0.658 omitted five
`int3g32-attn-only` rows. Every block is now computed on the same 92 rows.

What moved: E5 INT2 retention 41.9% -> 44.8%, EmbeddingGemma 65.7% -> 65.9%; E5's uniform-axis
Pearson r 0.56 -> 0.80 (the v1 "apparent exception" was the missing prompt); at INT3/g16
EmbeddingGemma's costliest module is now attention (0.71 vs 0.56 points) and E5, reported as
tied in v1, is attention-dominant (0.98 vs 0.36). Not re-measured (pre-fix values remain):
`sweep/turn-e5-*`, `sweep/turn-gemma-*`, `sweep/rank-gemma-*`, `ledger/pooling/`.

## Manuscript claims gate

`paper/claims_manifest.json` lists every number or phrasing the manuscript withdrew during
review, each with a regex. `paper/claims_gate.py` fails if any of them reappears:

```bash
python paper/claims_gate.py --manifest paper/claims_manifest.json paper/main.tex   # exit 1 = withdrawn claim is back
python paper/claims_gate.py --manifest paper/claims_manifest.json --self-test        # the gate's own negative controls
```

## Dataset and model revisions

The dataset is pinned to `ThakiCloud/SKILLRET` revision `a050ad2`, the two BEIR corpora and
all five evaluated checkpoints to their own commit SHAs (`manifest/`). The Hub is mutable;
results taken across revisions are not comparable.
