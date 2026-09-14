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

## Multiple-comparison check (INT4 module cells)

The sentence "ten of forty-five cells exclude zero, six survive Benjamini--Hochberg at q=0.05,
and the largest effect is unchanged at 1.01" is recomputed from the raw per-query scores:

```bash
python scripts/bh_fdr.py     # prints cells=45 ci_excludes_zero=10 bh_significant=6 largest=-1.01
```

It writes `ledger/control/2026-09-14-bh-fdr-int4.json` with per-cell two-sided bootstrap p-values.

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
-0.008 / 0.367 / 0.408 / 0.415; whole-model -0.346 / -0.088 / -0.089 / -0.078 at
INT4/g16, INT3/g16, INT3/g32, INT2/g16) and the uniform axis strong (0.856 either way). The
v1 anecdote comparing BGE-M3 `int3/g32` (0.159) with `int3g32-ffn-only` (0.174) compares two
denominators; whole-model the FFN-only arm reads 0.062 and the ordering is unremarkable. That
sentence is withdrawn in the errata.

Also from the same audit: `quantize()` walks only the transformer; EmbeddingGemma's two
sentence-transformers `Dense` layers (4.72M parameters) stayed FP32 in every v1 arm. The
uniform arm quantizes `position_embeddings` (BGE-M3 1.48%, E5 0.36% of parameters) that no
module arm touches, so the interaction residual for those two models includes that leftover.
E5-base-v2 ran with no query/document prefix and EmbeddingGemma without its document prefix
(`quant_sweep.py` now resolves both and refuses unknown prompt contracts). NFCorpus qrels were
binarized (576 of 12,334 judgments are graded 2). None of these change which module is
costliest within a model or the INT2 retention ordering; they are disclosed here and in the
errata, and the prompt/Dense cases are queued for re-measurement.

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
