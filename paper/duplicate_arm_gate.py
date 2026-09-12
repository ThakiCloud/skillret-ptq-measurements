#!/usr/bin/env python3
"""같은 팔이 표마다 다른 값으로 실리지 않는지 코드가 판정한다.

2026-09-12 리뷰: Table 2 의 Qwen INT2 가 1.3, Table 9 의 같은 팔이 1.2 였다. 두 값 모두
정당한 측정이었지만(재측정 차 0.05pp) 독자에게는 모순으로 보인다. 사람이 표를 대조해서
잡을 일이 아니다.
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 모델 라벨 → 그 행에서 뽑을 값의 위치를 표마다 지정한다(열 의미가 표마다 다르므로).
WATCH = [
    # (표 파일, 모델 라벨 정규식, 몇 번째 숫자열인가, 팔 이름)
    ("uniform.tex", r"Qwen3-Emb-0\.6B", 7, "INT2/g16"),
    ("int2mod.tex", r"Qwen3-Emb-0\.6B", 1, "INT2/g16"),
    ("uniform.tex", r"BGE-M3", 7, "INT2/g16"),
    ("int2mod.tex", r"BGE-M3", 1, "INT2/g16"),
]


def cell(fn, model_rx, idx):
    p = os.path.join(HERE, "tables", fn)
    if not os.path.exists(p):
        return None
    for line in open(p, encoding="utf-8"):
        if re.search(model_rx, line) and "&" in line:
            # ⛔ 셀은 \textbf{1.3} 처럼 감싸여 온다. 벗기지 않으면 숫자로 안 보이고
            #    열 위치가 통째로 밀려 게이트가 조용히 통과한다(2026-09-12 실측).
            parts = [c.strip().replace("\\\\", "") for c in line.split("&")]
            vals = []
            for c in parts[1:]:
                m = re.search(r"-?\d+(?:\.\d+)?", c)
                vals.append(float(m.group(0)) if m else None)
            if idx <= len(vals):
                return vals[idx - 1]
    return None


# ⛔ 2026-09-12 리뷰: 표끼리는 맞췄는데 본문에 옛 값(1.2)이 남아 리뷰어가 잡았다.
#    표-표 일치만으로는 부족하다 — 본문이 인용하는 수치도 같은 값이어야 한다.
PROSE = [
    # (표 파일, 모델 라벨, 숫자열, 본문에서 그 값을 인용하는 정규식 — 값 자리는 (\d+\.\d))
    ("int2mod.tex", r"Qwen3-Emb-0\.6B", 1, r"reproduces the (\d+\.\d)\\% and \d+\.\d\\% of the joint arm"),
    ("int2mod.tex", r"SkillRet-0\.6B", 1, r"reproduces the \d+\.\d\\% and (\d+\.\d)\\% of the joint arm"),
]


def prose_mismatches():
    p = os.path.join(HERE, "main.tex")
    if not os.path.exists(p):
        return []
    tex = open(p, encoding="utf-8").read()
    out = []
    for fn, rx, idx, prose_rx in PROSE:
        tv = cell(fn, rx, idx)
        m = re.search(prose_rx, tex)
        if tv is None or not m:
            continue
        pv = float(m.group(1))
        if abs(tv - pv) > 1e-9:
            out.append(f"⛔ {rx}: 표({fn}) {tv} vs 본문 {pv}")
    return out


def main() -> int:
    seen, bad = {}, []
    for fn, rx, idx, arm in WATCH:
        v = cell(fn, rx, idx)
        if v is None:
            continue
        key = (rx, arm)
        if key in seen and abs(seen[key][1] - v) > 1e-9:
            bad.append(f"⛔ {arm} / {rx}: {seen[key][0]} 에서 {seen[key][1]}, {fn} 에서 {v}")
        seen.setdefault(key, (fn, v))
    bad += prose_mismatches()
    for b in bad:
        print(b)
    if bad:
        print(f"\n{len(bad)}건 — 같은 팔이 표마다 다른 값이다. 한 출처에서 읽게 고쳐라.")
        return 1
    print(f"✅ 중복 팔 {len(seen)}종 · 표-표 및 표-본문 값 불일치 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
