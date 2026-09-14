#!/usr/bin/env python3
"""철회된 주장이 산출물에 다시 담기지 않았는지 코드가 판정한다.

한 라운드에서 내린 판정이 다음 라운드의 산문 속에서 조용히 복원되는 것을 막는 게이트다.
정본은 claims manifest 이고, 이 스크립트는 그 manifest 의 `withdrawn` 항목에서 유래한
숫자·표현이 대상 파일에 남아 있으면 exit 1 로 떨어진다.

⛔ 모든 금지 패턴은 음성 대조(must_pass)를 함께 갖는다. 실패하는 입력에서만 검증된 패턴은
   "무엇을 통과시키는지" 한 번도 보이지 않은 패턴이고, 실제로 그런 패턴들이 무해한 산문
   (부사 하나, 더 큰 숫자 안의 부분문자열)을 잡아 문장을 망가뜨렸다.

⛔ manifest 자신도 대상에 포함해서 돌려라. 게이트가 본문만 읽으면 정본 파일이 자기가 금지하는
   값을 들고 있어도 통과한다 — 실측 4회 재발.

사용:
    claims_gate.py --manifest m.json FILE...      # exit 0 통과 · 1 위반 · 2 사용법 오류
    claims_gate.py --manifest m.json --self-test  # 패턴의 음성 대조까지 검사
    claims_gate.py --manifest m.json --list       # 활성 금지 패턴 목록
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

EXIT_OK, EXIT_VIOLATION, EXIT_USAGE = 0, 1, 2


def load_manifest(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.stderr.write(f"⛔ manifest 없음: {path}\n")
        raise SystemExit(EXIT_USAGE)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"⛔ manifest 파싱 실패: {path}: {exc}\n")
        raise SystemExit(EXIT_USAGE)


def iter_withdrawn(man: dict):
    """withdrawn 을 (id, entry) 로 훑는다.

    실전 매니페스트는 id 를 키로 한 dict 인 경우가 많고 템플릿은 list 다. 첫 실전 대조에서
    list 만 가정한 초판이 KeyError 로 죽었다 — 두 형태를 모두 받는다.
    """
    w = man.get("withdrawn", [])
    if isinstance(w, dict):
        for k, v in w.items():
            if isinstance(v, dict):
                yield k, v
    else:
        for v in w:
            if isinstance(v, dict):
                yield v.get("id", "?"), v


def compile_bans(man: dict) -> list[tuple[re.Pattern, str, str]]:
    """withdrawn 항목 -> (컴파일된 정규식, 사유, 대체 키)."""
    out = []
    for eid, entry in iter_withdrawn(man):
        pat = entry.get("pattern")
        if not pat:
            continue
        try:
            rx = re.compile(pat)
        except re.error as exc:
            sys.stderr.write(f"⛔ 정규식 오류 [{eid}]: {exc}\n")
            raise SystemExit(EXIT_USAGE)
        # ⛔ exempt_context: 금지어가 **단독으로** 나올 때만 막고 싶을 때 쓴다.
        #    줄 단위 스캔은 표의 다음 줄에 있는 해명을 못 본다(2026-09-10 실측: 올바르게
        #    고친 모델카드가 3건 오탐됐다). 파일 전체에서 매치 주변 창을 본다.
        ex = entry.get("exempt_context")
        try:
            exrx = re.compile(ex) if ex else None
        except re.error as exc:
            sys.stderr.write(f"⛔ exempt_context 정규식 오류 [{eid}]: {exc}\n")
            raise SystemExit(EXIT_USAGE)
        out.append((rx, entry.get("why", ""), entry.get("replacement", ""), exrx))
    return out


def manifest_prose(path: str, raw: str) -> str | None:
    """매니페스트 자신을 훑을 때 — `pattern`/`must_catch`/`must_pass`/`replacement` 는
    **정의**이지 주장이 아니다. 그 줄까지 잡으면 진짜 1건이 자기매칭 50건에 묻힌다
    (2026-09-11 실측). 대신 사람이 쓴 산문 필드(`reason`)만 남겨서 스캔한다 —
    매니페스트를 훑는 목적(정본이 자기 게이트 밖에 있는 것)은 그대로 지킨다."""
    if not path.endswith(".json"):
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict) or "withdrawn" not in obj:
        return None
    out = []
    for e in obj.get("withdrawn", []):
        if isinstance(e, dict) and isinstance(e.get("reason"), str):
            out.append(e["reason"])
    return "\n".join(out)


def scan(paths: list[str], bans) -> int:
    bad = 0
    for p in paths:
        f = pathlib.Path(p)
        if not f.exists():
            sys.stderr.write(f"⛔ 대상 없음: {p}\n")
            bad += 1
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        prose = manifest_prose(p, text)
        if prose is not None:
            text = prose      # 매니페스트는 산문 필드만 — 정의 필드는 주장이 아니다
        # 줄 시작 오프셋 — 매치 위치를 파일 전체 좌표로 옮기기 위해서다
        offs, acc = [], 0
        for ln in text.splitlines():
            offs.append(acc); acc += len(ln) + 1
        for lineno, line in enumerate(text.splitlines(), 1):
            for rx, why, repl, exrx in bans:
                m = rx.search(line)
                if not m:
                    continue
                if exrx is not None:
                    at = offs[lineno - 1] + m.start()
                    win = text[max(0, at - 260): at + 260]
                    if exrx.search(win):
                        continue          # 해명이 붙어 있다 — 단독 인용이 아니다
                bad += 1
                print(f"⛔ {f.name}:{lineno}  {why}")
                print(f"     {line.strip()[:110]}")
                if repl:
                    print(f"     → 대신: {repl}")
    return bad


def self_test(man: dict, bans) -> int:
    """must_catch 는 반드시 걸리고 must_pass 는 반드시 통과해야 한다."""
    fails = 0
    missing_pattern: list[str] = []
    for eid, entry in iter_withdrawn(man):
        pat = entry.get("pattern")
        if not pat:
            missing_pattern.append(eid)
            continue
        rx = re.compile(pat)
        # ⛔ 자체검사도 실제 스캔과 **같은 판정**을 써야 한다. exempt_context 를 안 보면
        #    올바른 산문을 오탐으로 보고하고, 그걸 고치려다 진짜 패턴을 무디게 만든다
        #    (2026-09-10 실측: 해명이 붙은 문장 2건이 오탐으로 잡혔다).
        _ex = entry.get("exempt_context")
        _exrx = re.compile(_ex) if _ex else None
        def _hit(txt: str) -> bool:
            m = rx.search(txt)
            if not m:
                return False
            if _exrx is not None:
                at = m.start()
                if _exrx.search(txt[max(0, at - 260): at + 260]):
                    return False
            return True
        catches = entry.get("must_catch", [])
        passes = entry.get("must_pass", [])
        if not catches:
            print(f"⚠️  [{eid}] must_catch 없음 — 이 패턴은 한 번도 발화가 확인되지 않았다")
            fails += 1
        for s in catches:
            if not _hit(s):
                print(f"⛔ [{eid}] 놓침: {s!r}")
                fails += 1
        if not passes:
            print(f"⚠️  [{eid}] must_pass 없음 — 무엇을 통과시키는지 보인 적이 없는 패턴이다")
            fails += 1
        for s in passes:
            if _hit(s):
                print(f"⛔ [{eid}] 오탐: {s!r}")
                fails += 1
    if missing_pattern:
        print(f"⚠️  pattern 없는 withdrawn 항목 {len(missing_pattern)}건 — 게이트가 못 막는다: "
              + ", ".join(missing_pattern[:6]))
        fails += len(missing_pattern)
    # manifest 전역 음성 대조 (여러 패턴이 함께 걸리는 문장)
    for s in man.get("global_must_pass", []):
        for rx, why, _, _ex2 in bans:
            if rx.search(s):
                print(f"⛔ 전역 오탐 ({why}): {s!r}")
                fails += 1
    return fails


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="철회된 주장 재등장 게이트")
    ap.add_argument("files", nargs="*", help="검사 대상 (manifest 자신도 포함하라)")
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--self-test", action="store_true", help="패턴의 must_catch/must_pass 검사")
    ap.add_argument("--list", action="store_true", help="활성 금지 패턴 출력")
    args = ap.parse_args(argv)

    man = load_manifest(args.manifest)
    bans = compile_bans(man)

    if args.list:
        for rx, why, repl, _ex in bans:
            print(f"{rx.pattern}\n    사유: {why}\n    대체: {repl or '(없음)'}")
        print(f"\n{len(bans)}개 금지 패턴.")
        return EXIT_OK

    if args.self_test:
        fails = self_test(man, bans)
        if fails:
            print(f"\n⛔ 자체검사 실패 {fails}건. 패턴을 고치거나 대조를 추가하라.")
            return EXIT_VIOLATION
        print(f"자체검사 통과 — {len(bans)}개 패턴, 음성 대조 포함.")
        return EXIT_OK

    if not args.files:
        sys.stderr.write("⛔ 검사 대상이 없다. manifest 자신도 대상에 넣어라.\n")
        return EXIT_USAGE

    bad = scan(args.files, bans)
    if bad:
        print(f"\n{bad}건. 정본은 {args.manifest} 이다.")
        return EXIT_VIOLATION
    print(f"claims gate: 철회된 주장 없음 ({len(args.files)}개 파일, {len(bans)}개 패턴)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
