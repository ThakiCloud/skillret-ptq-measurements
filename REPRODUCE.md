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

Not every file under `ledger/` is citable. `ledger/STATUS.yaml` marks each one
`canonical`, `superseded`, `retracted` or `exploratory`, and `paper_numbers.py` drops
anything that is not canonical, so a withdrawn or outdated arm cannot reach a table. The
records stay in place — see `RETRACTIONS.md` for why each one was withdrawn.

## Dataset and model revisions

The dataset is pinned to `ThakiCloud/SKILLRET` revision `a050ad2`, the two BEIR corpora and
all five evaluated checkpoints to their own commit SHAs (`manifest/`). The Hub is mutable;
results taken across revisions are not comparable.
