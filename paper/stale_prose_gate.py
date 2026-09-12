#!/usr/bin/env python3
"""canonical 원장의 **사람이 읽는 서술**에 옛 수치가 남아 있으면 죽는다.

⛔ 왜 필요한가(2026-09-13 실측): 같은 JSON 안에서 숫자 필드(`sizes_mb`)는 실측으로
   고쳐졌는데 `finding`·`why`·`implication`·`teacher_ptq_note` 는 옛 값과 옛 프레이밍
   (byte-matched)을 그대로 들고 있었다. 숫자만 대조하는 게이트는 이걸 못 잡는다 —
   값이 바뀐 자리가 아니라 **문장이 안 바뀐 자리**이기 때문이다.

화이트리스트: 과거값을 일부러 보존하는 필드(`*_predicted`, `superseded`, `unit_note`,
`why`, `sizes_note`, `retraction*`, `corrections_log`, `withdrawn`)는 검사하지 않는다.
그 필드들이 존재하는 이유가 옛 값을 적어 두는 것이다.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ⛔ 경로를 조각으로 join 하면 배포 빌더의 `ledger`→`ledger` 재작성이 안 걸린다.
#    실측(2026-09-13): clean clone 에서 이 게이트가 **0개 파일을 검사하고 통과**했다.
#    두 레이아웃을 다 보고, 못 찾으면 조용히 통과하지 말고 죽는다.
LEDGER = next((p for p in (os.path.join(ROOT, "ledger"),
                           os.path.join(ROOT, "ledger")) if os.path.isdir(p)), None)
if LEDGER is None:
    raise SystemExit("\u26d4 ledger 디렉터리를 못 찾았다 — 검사 없이 통과시키지 않는다")

BANNED = [
    (r"\b68\.6\s?MB", "109M INT3/g16 은 실측 68.4MB (68.350080 바이트 기준)"),
    (r"\b82\.3\s?MB", "109M INT4/g16 은 실측 82.0MB"),
    (r"\b17\.1\s?MB", "22M INT4/g16 은 실측 17.0MB"),
    (r"\b372\.5\s?MB", "0.6B INT3/g16 은 실측 372.3MB"),
    (r"\b605(\.0)?\s?MB", "교사 저장 아티팩트는 1191.6MB BF16"),
    (r"\b1152(\.0)?\s?MB", "1152 는 du -sm(MiB) — 십진 MB 는 1191.6"),
    (r"\b80\.82\b", "교사 기준 점수는 78.48 (접두어 계약 수정 후)"),
    (r"byte-matched", "실제 비교는 68.4 vs 297.9 — 크기·품질 파레토 지배"),
    (r"equal-byte", "실제 비교는 68.4 vs 297.9 — 크기·품질 파레토 지배"),
]
# 옛 값을 보존하는 것이 목적인 필드 — 여기서 옛 값이 나오는 건 정상이다.
WHITELIST = re.compile(
    r"(_predicted|superseded|retract|unit_note|sizes_note|corrections_log|withdrawn"
    r"|historical|\.why$|\.why\.)", re.I)


def status_of(rel):
    p = os.path.join(LEDGER, "STATUS.yaml")
    if not os.path.exists(p):
        return "canonical"
    try:
        import yaml
    except ImportError:
        raise SystemExit("⛔ pyyaml 이 필요하다")
    d = yaml.safe_load(open(p, encoding="utf-8")) or {}
    return ((d.get("entries") or {}).get(rel) or {}).get("status", "canonical")


def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}", out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", out)
    elif isinstance(o, str):
        if WHITELIST.search(path):
            return
        for pat, fix in BANNED:
            if re.search(pat, o):
                out.append((path, pat, fix, o[:110]))


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    files = argv or [os.path.join(r, f)
                     for r, _, fs in os.walk(LEDGER) for f in fs if f.endswith(".json")]
    bad = 0
    for f in sorted(files):
        rel = os.path.relpath(f, LEDGER)
        if status_of(rel) != "canonical":
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        hits = []
        walk(d, "", hits)
        for path, pat, fix, snip in hits:
            bad += 1
            print(f"⛔ {rel}{path}\n   패턴 {pat} · {fix}\n   {snip}")
    if not files:
        print("⛔ 검사한 파일이 0개다 — 통과가 아니라 배선 오류다")
        return 1
    if bad:
        print(f"\n⛔ canonical 원장 서술에 옛 수치 {bad}건")
        return 1
    print(f"✅ canonical 원장 서술에 옛 수치 없음 ({len(files)} 파일 검사)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
