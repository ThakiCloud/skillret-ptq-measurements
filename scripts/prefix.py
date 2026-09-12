#!/usr/bin/env python3
"""질의 접두어의 단일 출처.

⛔ 왜 모듈인가(2026-09-10 실측): 접두어가 12개 파일에 각자 상수로 박혀 있었고, 학습기는
   접두어를 안 붙이는데 평가기들은 붙이는 상태로 갈라졌다. 모델이 한 번도 본 적 없는
   20토큰으로 채점돼 **-8.84pp**(63.10 vs 71.94). 상수를 복붙하는 한 또 갈라진다.

계약: 학습기가 `query_prefix.json` 을 체크포인트 옆에 남기고, 평가기는 그걸 읽는다.
기록이 없으면 추측하지 않고 죽는다 — 조용히 틀린 접두어를 쓰는 게 멈추는 것보다 나쁘다.
"""
import json
import os

SKILLRET = ("Instruct: Given a skill search query, retrieve relevant skills that match "
            "the query\nQuery: ")
ARCTIC = "Represent this sentence for searching relevant passages: "
PREFIX = {"none": "", "skillret": SKILLRET, "arctic": ARCTIC}


def resolve(model_path, override=None):
    """모델이 학습 때 쓴 접두어를 돌려준다. override 는 외부 사전학습 모델용."""
    if override is not None:
        return PREFIX.get(override, override)
    f = os.path.join(str(model_path), "query_prefix.json")
    if os.path.exists(f):
        return json.load(open(f))["resolved"]
    raise SystemExit(
        f"⛔ {model_path} 에 query_prefix.json 이 없다 — 학습 때 쓴 접두어를 알 수 없다. "
        f"--query-prefix 로 명시해라(none|skillret|arctic).")


def stamp(out_dir, name):
    """학습기가 체크포인트 옆에 접두어를 기록한다."""
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"query_prefix": name, "resolved": PREFIX.get(name, name)},
              open(os.path.join(out_dir, "query_prefix.json"), "w"), ensure_ascii=False)
