#!/usr/bin/env python3
"""⛔ S3 재동기화가 철회 플래그를 지운다 — 플래그를 사람이 아니라 코드가 소유하게 한다.

실측 2026-09-11: 원장에 retracted 를 손으로 박았다가, S3 에서 다시 받으면서 통째로
날아갔다. 표에 안 나오던 값이 조용히 되살아났다. 동기화 뒤에는 항상 이걸 돌린다.
"""
import json, glob, os, sys

# (파일 접두어가 이 목록에 없으면) 이 팔들은 수정 전 양자화기로 잰 것이라 철회한다
BROKEN_ARMS = {"int4/g16-sym", "ternary/g16"}
FIXED_PREFIXES = ("rank-", "e5-")
REASON = ("대칭 양자화기의 fp16 스케일 언더플로 — 전부-0 그룹에서 scale 이 0 이 되어 "
          "g/0 → nan. BGE-M3 token_type_embeddings 가 해당. 수정 후 재측정본은 "
          "rank-* / e5-* 파일에 있다.")


def main(root="ledger/sweep"):
    n = 0
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        if os.path.basename(f).startswith(FIXED_PREFIXES):
            continue
        d = json.load(open(f)); ch = False
        for r in d.get("rows", []):
            if r["arm"] in BROKEN_ARMS and not r.get("retracted"):
                r["retracted"] = True; r["retracted_reason"] = REASON; ch = True
        if ch:
            json.dump(d, open(f, "w"), ensure_ascii=False, indent=1); n += 1
    print(f"철회 적용: {n}개 파일 갱신")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
