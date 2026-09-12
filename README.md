# Where Post-Training Quantization Breaks Text Embedders

Measurements, analysis code and the paper source for *Where Post-Training Quantization
Breaks Text Embedders: A Measured Map Across Four Embedder Families*.

## What is here

- `ledger/` — every arm, with per-query scores. Five checkpoints, four families, three
  corpora, a grid over bit width and group size. Retracted arms carry a `retracted` flag and
  are excluded by the loaders rather than deleted.
- `paper/` — LaTeX source, the table and figure generators, and the gates. Every table and
  figure is generated from `ledger/` by `paper_numbers.py`, and repeated headline values in
  the prose are checked against registered ledger claims by the manuscript gates.
- `scripts/` — the sweep harness, the reconstruction-error analysis, and `build_release.py`,
  which packs a checkpoint for real so reported sizes are file sizes rather than arithmetic.
- `manifest/` — the pinned dataset revision, the five evaluated checkpoints with their
  pooling and query prefixes, and the recipe provenance for the two distilled students.
- `artifacts.csv` — the byte provenance behind every size in the paper, in decimal MB, with
  packed artifacts distinguished from serialized full-precision ones.
- `SHA256SUMS` — checksums for every file here.
- `RETRACTIONS.md` — which arms were withdrawn, why, and whether they were replaced.
- `REPRODUCE.md` — the exact commands, including re-measuring a single arm from scratch.

## Reproducing a number

```bash
python paper/make_tables.py           # regenerate every table from the ledger
python paper/duplicate_arm_gate.py    # fail if one arm carries two different values
```

`REPRODUCE.md` has the rest, including how to re-evaluate one arm end to end and how to
pack a checkpoint so its size is a measured file rather than an estimate.

Evaluation used the public `ThakiCloud/SKILLRET` dataset at revision `a050ad2`. The Hub is
mutable; that revision is what these numbers are pinned to.

## Checkpoints

Public at <https://huggingface.co/ThakiCloud>.

## Note on paths

Internal path prefixes were sanitized; public mappings are recorded in `manifest/` where
available. Some ledger entries still carry a sanitized local path rather than a public model
identifier, which is deliberate — the path is what that run actually used. Numbers are
untouched.

## Tooling note

AI-assisted tools were used during code and manuscript editing. Experimental execution,
verification, interpretation, and the final claims are the responsibility of the author.
