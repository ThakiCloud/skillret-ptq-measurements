#!/usr/bin/env python3
"""생성된 .tex 에 파이썬 이스케이프 사고가 남았는지 코드가 판정한다.

이 세션에서 같은 함정에 세 번 걸렸다. 생성기가 파이썬이라 문자열 안의 `\\ref` 는
`\\r`(캐리지리턴) + `ef` 로 먹히고, f-string 안의 `{...}` 는 포맷 자리로 먹힌다.
둘 다 컴파일은 성공하고, PDF 에 `Table eftab:turnover` 가 그대로 찍힌 채로 나간다.
게이트가 없으면 사람이 렌더된 쪽을 읽어야만 잡힌다.
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# (이름, 정규식, 설명)
CHECKS = [
    ("carriage-return", re.compile(r"\r"),
     "캐리지리턴이 들어 있다 — 파이썬 소스에서 \\ref 를 \\\\ref 로 쓰지 않았다"),
    ("broken-ref", re.compile(r"(?<!\\)\bef\{"),
     "`ef{` — \\ref 의 백슬래시-r 이 먹혔다"),
    ("broken-label", re.compile(r"(?<!\\)\babel\{"),
     "`abel{` — \\label 의 백슬래시-l 이 먹혔다"),
    ("bare-brace-leak", re.compile(r"\{[a-z]+:[a-z0-9-]+\}(?<!\\ref\{)", re.I), None),
]


def main() -> int:
    bad = []
    for f in sorted(glob.glob(os.path.join(HERE, "tables", "*.tex"))
                    + glob.glob(os.path.join(HERE, "*.tex"))):
        txt = open(f, encoding="utf-8", newline="").read()
        for name, rx, why in CHECKS[:3]:
            for m in rx.finditer(txt):
                ctx = txt[max(0, m.start() - 40):m.start() + 30].replace("\n", " ")
                bad.append(f"⛔ {os.path.basename(f)} [{name}] {why}\n     …{ctx.strip()}…")
    for b in bad:
        print(b)
    if bad:
        print(f"\n{len(bad)}건. 생성기의 파이썬 문자열 이스케이프를 고쳐라.")
        return 1
    print("✅ 생성 .tex 이스케이프 정상")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:          # ⛔ 게이트 자신도 실패할 수 있어야 한다
        p = os.path.join(HERE, "tables", "_escape_selftest.tex")
        open(p, "w", encoding="utf-8").write("see Table~\ref{tab:x}\n")
        rc = main()
        os.remove(p)
        print("self-test:", "PASS (잡았다)" if rc == 1 else "FAIL (못 잡았다)")
        sys.exit(0 if rc == 1 else 1)
    sys.exit(main())
