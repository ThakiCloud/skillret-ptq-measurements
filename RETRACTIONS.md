# Retracted arms

Retracted measurements are flagged in place, never deleted. The loaders in
`paper/paper_numbers.py` drop any row carrying `"retracted": true`, so a retracted
number cannot reach a table, but the record of it having been taken survives.

## Symmetric quantization, four checkpoints (2026-09-11)

The symmetric quantizer divided by an fp16 scale that underflows to zero on
all-zero groups, producing NaN embeddings. Retrieval then scored near zero, which
read as a collapse rather than as a defect in our own code. It nearly became a
headline finding about BGE-M3.

Retracted: every `*-sym` and `ternary/g16` arm for Qwen3-Embedding-0.6B,
SKILLRET-Embedding-0.6B, EmbeddingGemma-300M and BGE-M3. E5-base-v2 was
re-measured after the fix, and its symmetric and ternary arms stand.

Replacement: partial. Ternary is reported for one checkpoint only, and the
uniform-quantization table says so rather than generalising from the single
surviving arm.

## What a retraction looks like in the ledger

```json
{"arm": "int2-sym/g16", "retracted": true,
 "retraction_reason": "fp16 scale underflow on all-zero groups -> NaN"}
```

Grep for `retracted` under `ledger/` to enumerate them.
