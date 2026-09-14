#!/usr/bin/env python3
"""산술 크기 예측(quant_ft22m.size_mb 와 같은 식) vs 실제 패킹 바이트(artifacts.csv) — 논문
§Byte ledger 의 "0.04 to 0.44% above the packed file across the seven quantized artifacts" 를
가중치 다운로드 없이 재계산한다: 텐서 shape 는 Hub safetensors 메타데이터에서 읽는다.
⛔ 감사(2026-09-14): 이 범위의 산출 근거가 구조화 원장에 없었다. 값은 코드가 낸다."""
import argparse, csv, json, math, os
from huggingface_hub import get_safetensors_metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = {"ThakiCloud/SKILLRET-Edge-22M": "ThakiCloud/SKILLRET-Edge-22M",
        "ThakiCloud/SKILLRET-Edge-109M": "ThakiCloud/SKILLRET-Edge-109M",
        "ThakiCloud/SKILLRET-Embedding-0.6B": "ThakiCloud/SKILLRET-Embedding-0.6B"}


def predicted_bytes(shapes, bits, group):
    tot_bits = 0
    for shp in shapes:
        n = math.prod(shp)
        if n < group or len(shp) < 2 or bits >= 16:
            tot_bits += n * 16
        else:
            tot_bits += n * bits + math.ceil(n / group) * 32
    return tot_bits / 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="artifacts.csv (public measurement repo)")
    ap.add_argument("--out", default=os.path.join(ROOT, "ledger/control/2026-09-14-size-prediction-gap.json"))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.artifacts)) if r["kind"] == "packed_quantized"]
    shapes = {}
    for repo in REPO.values():
        md = get_safetensors_metadata(repo)
        shapes[repo] = [t.shape for t in md.weight_map and md.files_metadata[list(md.files_metadata)[0]].tensors.values()] \
            if len(md.files_metadata) == 1 else [t.shape for f in md.files_metadata.values() for t in f.tensors.values()]
    out = []
    for r in rows:
        bits = int(r["arm"].split("/")[0].replace("int", ""))
        group = int(r["arm"].split("/")[1].split("-")[0].replace("g", ""))
        pred = predicted_bytes(shapes[REPO[r["model"]]], bits, group)
        meas = int(r["size_bytes"])
        out.append({"model": r["model"], "arm": r["arm"], "n_tensors": len(shapes[REPO[r["model"]]]),
                    "predicted_bytes": int(pred), "measured_bytes": meas,
                    "gap_pct": round(100 * (pred - meas) / meas, 3)})
    gaps = [o["gap_pct"] for o in out]
    res = {"date": "2026-09-14", "subject": "산술 예측(size_mb 식, shape 는 Hub safetensors 메타) vs 패킹 실측 바이트",
           "n_artifacts": len(out), "gap_pct_min": min(gaps), "gap_pct_max": max(gaps), "rows": out,
           "note": "예측은 모든 2-D 텐서를 그룹 양자화한다고 가정한다. 패커가 남기는 헤더/정렬 차이가 양의 갭으로 나타난다."}
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    for o in out:
        print(f"{o['model'].split('/')[-1]:28s} {o['arm']:14s} pred {o['predicted_bytes']:>12,d}  meas {o['measured_bytes']:>12,d}  gap {o['gap_pct']:+.3f}%")
    print(f"n={len(out)} gap range {min(gaps):+.2f}% .. {max(gaps):+.2f}% → {a.out}")


if __name__ == "__main__":
    main()
