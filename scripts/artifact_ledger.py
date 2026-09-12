#!/usr/bin/env python3
"""크기 주장의 바이트 출처표를 만든다 — 파일을 실제로 만들고 그 파일을 잰다.

⛔ 이 스크립트는 `sizes_mb` 원장을 **베끼지 않는다**. 각 팔을 다시 패킹해서
   파일을 쓰고, 그 파일의 바이트 수와 sha256 을 적는다. 논문이 "크기 주장은
   그 크기의 파일이 존재해야 주장이다"라고 쓰는 이상, 저장소도 같은 기준을
   지켜야 machine-verifiable 하다.

정본은 `size_bytes` 이고 `size_mb_decimal` 은 파생값(bytes/1e6)이다.
"""
import argparse, csv, hashlib, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# (모델 id, 로컬 경로 env/기본, [(arm, bits, group) ...])
TARGETS = [
    ("ThakiCloud/SKILLRET-Edge-22M",        "SKILLRET_EDGE22M",
     [("int4/g16-asym", 4, 16), ("int3/g16", 3, 16)]),
    ("ThakiCloud/SKILLRET-Edge-109M",       "SKILLRET_EDGE109M",
     [("int4/g16-asym", 4, 16), ("int3/g16", 3, 16), ("int3/g32", 3, 32)]),
    ("ThakiCloud/SKILLRET-Embedding-0.6B",  "SKILLRET_06B",
     [("int3/g16", 3, 16), ("int3/g32", 3, 32)]),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_dtype(src):
    """체크포인트가 디스크에 실제로 담고 있는 dtype — 논문 표기와 맞춰야 한다."""
    try:
        from safetensors import safe_open
        st = [f for f in os.listdir(src) if f.endswith(".safetensors")]
        if not st:
            return "unknown"
        with safe_open(os.path.join(src, st[0]), framework="pt") as f:
            k = next(iter(f.keys()))
            return str(f.get_slice(k).get_dtype()).lower().replace("torch.", "")
    except Exception:
        return "unknown"


def hf_revision(model_id):
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{model_id}", timeout=20) as r:
            return json.load(r).get("sha", "")
    except Exception:
        return ""


def serialized_file(src):
    for f in sorted(os.listdir(src)):
        if f.endswith(".safetensors") or f == "pytorch_model.bin":
            return os.path.join(src, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts.csv")
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    work = a.workdir or tempfile.mkdtemp(prefix="artifact-ledger-")
    os.makedirs(work, exist_ok=True)

    rows = []
    for model_id, env, arms in TARGETS:
        src = os.environ.get(env)
        if not src or not os.path.isdir(src):
            print(f"  ⛔ {model_id}: {env} 미지정 — 건너뜀(행을 지어내지 않는다)",
                  file=sys.stderr)
            continue
        rev = hf_revision(model_id)
        dt = stored_dtype(src)
        sf = serialized_file(src)
        if sf:
            # 공개된 그대로의 가중치 파일. 학생 둘은 FP32 로 배포돼 있어 논문 표의
            # fp16 행과 다르다 — 숨기지 않고 두 행을 다 적는다.
            rows.append(dict(model=model_id, arm=f"{dt} (as published)",
                             kind=f"serialized_{dt}",
                             source_file=os.path.basename(sf),
                             size_bytes=os.path.getsize(sf),
                             sha256=sha256(sf), model_revision=rev))
        for arm, bits, group in arms:
            dst = os.path.join(work, f"{model_id.split('/')[-1]}-{bits}-{group}")
            subprocess.run([sys.executable, os.path.join(HERE, "build_release.py"),
                            "--src", src, "--dst", dst,
                            "--bits", str(bits), "--group", str(group)], check=True)
            packed = os.path.join(dst, f"model-int{bits}-g{group}.bin")
            rows.append(dict(model=model_id, arm=arm, kind="packed_quantized",
                             source_file=os.path.basename(packed),
                             size_bytes=os.path.getsize(packed),
                             sha256=sha256(packed), model_revision=rev))
            # ⛔ 논문의 전정밀 행은 fp16 이다. 공개본이 FP32 인 학생의 경우 그 값은
            #    파일이 아니라 산술이 된다 — build_release 가 쓰는 fp16 safetensors
            #    를 실제로 재서 행을 만든다. 모델당 한 번만.
            fp16 = os.path.join(dst, "model.safetensors")
            if os.path.exists(fp16) and not any(
                    r["model"] == model_id and r["arm"] == "fp16" for r in rows):
                rows.append(dict(model=model_id, arm="fp16",
                                 kind="serialized_fp16",
                                 source_file=os.path.basename(fp16),
                                 size_bytes=os.path.getsize(fp16),
                                 sha256=sha256(fp16), model_revision=rev))

    for r in rows:
        r["size_mb_decimal"] = round(r["size_bytes"] / 1e6, 3)
    cols = ["model", "arm", "kind", "source_file", "size_bytes",
            "size_mb_decimal", "sha256", "model_revision"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r[c] for c in cols} for r in rows])
    print(f"→ {a.out} ({len(rows)} 행) · 작업본 {work}")


if __name__ == "__main__":
    raise SystemExit(main())
