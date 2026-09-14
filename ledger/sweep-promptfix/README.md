# sweep-promptfix — E5-base-v2 · EmbeddingGemma 재측정 (prompt contract 수정 후)

- 잡 `srpromptfix` (`job/promptfix_spec.json`, `job/promptfix_job.sh`), H200 `tk-ai-wkld-wk-gpu-003`, 2026-09-14, 1 GPU.
- 왜: 감사 (ii) — `quant_sweep.py` 가 프롬프트를 질의에만 적용했다. E5-base-v2 는 sentence-transformers
  프롬프트 설정이 없어 `query:`/`passage:` 없이, EmbeddingGemma 는 문서 접두어 `title: none | text: ` 없이
  측정됐다. `resolve_prompts()` 가 CLI > ST config(프롬프트가 실제로 있을 때만) > `PROMPT_CONTRACT` 로 해석하고
  문서 접두어를 인코딩 시 적용한다. 각 파일이 `query_prompt`·`doc_prompt`·`prompt_source` 를 기록한다.
- 무영향: Qwen3-Embedding·fine-tune(문서 프롬프트 없음)·BGE-M3(프롬프트 없음)는 재측정하지 않았다.
- 정본 규칙: `ledger/STATUS.yaml` 이 `sweep/e5-*.json`·`sweep/gemma-*.json` 을 `superseded(prompt-contract)` 로
  돌렸고, `paper/paper_numbers.py:sweep_files()` 가 `sweep/`·`sweep-promptfix/` 를 같은 규칙으로 읽는다.
- 파생 원장 재생성: `scripts/promote_promptfix.py` (INT2 격리 병합본 · recon-vs-rank 손실 갱신) →
  `scripts/recon_analysis.py` (두 분모 정의) → `scripts/bh_fdr.py` (BH + 60셀 paired CI).
- 여기 없는 것: Gemma 의 ST Dense 두 층까지 양자화한 대조군(`control/2026-09-14-promptfix-gemma-dense-*.json`,
  감사 (iii) 혼입 검정) · 잡 안에서 계산한 recon 요약(`control/2026-09-14-recon-promptfix-inrun.json`).
- 재측정되지 않은 것(논문 §audit 공개): `turn-e5/gemma-*`(top-10 교체율)·`rank-gemma-*`(뒤집힘 곡선)·
  `pooling` 개입은 수정 전 프롬프트로 측정된 그대로다.
