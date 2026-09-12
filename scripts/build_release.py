#!/usr/bin/env python3
"""HF 공개용 릴리스 아티팩트를 만든다 — 양자화 왕복 가중치 + 진짜 패킹 바이트.

⛔ 왜 둘 다 넣나: safetensors 만 넣으면 "17MB" 주장을 검증할 수 없고, 패킹 .bin 만
   넣으면 아무도 못 돌린다. 둘을 같이 넣어야 받는 사람이 (a) 바로 점수를 재현하고
   (b) 크기 주장을 바이트로 확인한다([[evaluator-must-act]]).
"""
import argparse, json, os, shutil, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_ft22m import fake_quant_, size_mb
from pack_verify import pack
import numpy as np
from transformers import AutoModel
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group", type=int, default=16)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    m = AutoModel.from_pretrained(a.src, dtype=torch.float32)

    # 1) 양자화 왕복 — 이게 공개 점수를 내는 가중치다
    rmse, nq, packed = {}, 0, {}
    for name, p in m.named_parameters():
        if p.dim() < 2:            # bias·LayerNorm 은 건드리지 않는다(크기 기여 미미)
            continue
        r = fake_quant_(p.data, a.bits, a.group)
        rmse[name] = r; nq += p.numel()

    predicted = size_mb(m, a.bits, a.group)

    # 2) 진짜 비트패킹 — 예측 크기가 실물 바이트와 맞는지 증명한다
    blob = bytearray(); meta = []
    for name, p in m.named_parameters():
        if p.dim() < 2:
            continue
        flat = p.data.reshape(-1).numpy().astype(np.float32)
        n = flat.size; padn = (-n) % a.group
        f = np.concatenate([flat, np.repeat(flat[-1:], padn)]) if padn else flat
        g = f.reshape(-1, a.group)
        lo, hi = g.min(1, keepdims=True), g.max(1, keepdims=True)
        levels = 2 ** a.bits - 1
        sc = np.clip((hi - lo) / levels, 1e-12, None).astype(np.float16)
        lo16 = lo.astype(np.float16)
        idx = np.clip(np.round((g - lo16.astype(np.float32)) / sc.astype(np.float32)), 0, levels).astype(np.uint8)
        off = len(blob)
        blob += pack(idx.reshape(-1), a.bits)
        blob += sc.tobytes() + lo16.tobytes()
        meta.append({"name": name, "shape": list(p.shape), "offset": off, "n": n})
    with open(os.path.join(a.dst, f"model-int{a.bits}-g{a.group}.bin"), "wb") as fh:
        fh.write(bytes(blob))
    on_disk = len(blob) / 1e6

    save_file({k: v.to(torch.float16).contiguous() for k, v in m.state_dict().items()},
              os.path.join(a.dst, "model.safetensors"))
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "query_prefix.json", "special_tokens_map.json", "vocab.txt"):
        s = os.path.join(a.src, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(a.dst, f))
    json.dump({"bits": a.bits, "group": a.group, "scale_dtype": "float16",
               "scheme": "group-wise asymmetric min/max",
               "quantized_params": nq, "predicted_mb": round(predicted, 3),
               "packed_on_disk_mb": round(on_disk, 3),
               "error_pct": round(abs(on_disk - predicted) / predicted * 100, 3),
               "tensors": meta},
              open(os.path.join(a.dst, "quantization.json"), "w"), indent=1)
    print(f"예측 {predicted:.3f}MB · 실물 {on_disk:.3f}MB · 오차 "
          f"{abs(on_disk-predicted)/predicted*100:.2f}%  → {a.dst}")


if __name__ == "__main__":
    main()
