#!/usr/bin/env python3
"""RQ1 — 가중치 재구성 오차가 검색 손실을 예측하는가.

⛔ 예측한다면 RankPTQ 계열(순위를 직접 목적함수로 삼는 방법)의 전제가 사라진다.
   인코딩 없이 가중치만 보면 되므로 CPU 로 몇 분이면 끝난다.
"""
import argparse, json, os, sys, glob, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_sweep import part_of, ARMS, _sym_quant_
from quant_ft22m import fake_quant_
from transformers import AutoModel


def recon_error(path, bits, group, parts=None, sym=False):
    """파라미터 수로 가중한 상대 오차 — 텐서 크기가 제각각이라 단순 평균은 왜곡된다."""
    m = AutoModel.from_pretrained(path, dtype=torch.float32, trust_remote_code=True)
    num = den = 0.0
    for n, p in m.named_parameters():
        if p.dim() < 2:
            continue
        if parts and part_of(n) not in parts:
            continue
        o = p.data.clone()
        if sym:
            _sym_quant_(p.data, bits, group)
        else:
            fake_quant_(p.data, bits, group)
        rel = float((o - p.data).pow(2).sum().sqrt() / o.pow(2).sum().sqrt().clamp(min=1e-12))
        num += rel * p.numel(); den += p.numel()
    del m
    return num / max(den, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="tag=path 쌍")
    ap.add_argument("--sweep-dir", default="ledger/sweep")
    ap.add_argument("--out", default="ledger/control/2026-09-11-recon-vs-rank.json")
    a = ap.parse_args()

    ndcg = {}
    for f in glob.glob(os.path.join(a.sweep_dir, "*.json")):
        # ⛔ rank-*/pool-* 는 접두어가 달라 모델명 파싱이 깨진다. 그리고 철회된 팔은 넣지 않는다 —
        #    실측: 철회 전에 돌린 분석이 NaN 산물(bgem3 int4-sym)을 그림에 스파이크로 남겼다.
        if os.path.basename(f).startswith(("rank-", "pool-")):
            continue
        d = json.load(open(f))
        tag, _, corpus = os.path.basename(f)[:-5].partition("-")
        ndcg[(tag, corpus.replace("beir-", ""))] = {
            r["arm"]: r["ndcg@10"] for r in d["rows"] if not r.get("retracted")}

    rows = []
    for spec in a.models:
        tag, _, path = spec.partition("=")
        corpora = [c for (t, c) in ndcg if t == tag]
        if not corpora:
            print(f"⛔ {tag}: 스윕 결과 없음 — 건너뛴다"); continue
        for arm, kw in ARMS:
            if kw["bits"] >= 16:
                continue
            drops = []
            for c in corpora:
                r = ndcg[(tag, c)]
                if arm not in r:
                    continue
                drops.append(100 * (r["fp16"] - r[arm]) / r["fp16"])
            if not drops:
                continue
            e = recon_error(os.path.expanduser(path), kw["bits"], kw.get("group", 16),
                            kw.get("parts"), kw.get("sym", False))
            rows.append({"model": tag, "arm": arm, "recon_rel_err": round(e, 5),
                         "ndcg_drop_pct": round(float(np.mean(drops)), 3),
                         "n_corpora": len(drops)})
            print(f"  {tag:11s} {arm:20s} 재구성오차 {e:.4f}  NDCG 손실 {np.mean(drops):6.2f}%",
                  flush=True)

    # 상관: 전체 / 모델 내부 / 붕괴 팔 제외
    def corr(xs, ys):
        if len(xs) < 3: return None
        x, y = np.array(xs), np.array(ys)
        return round(float(np.corrcoef(x, y)[0, 1]), 3)

    out = {"rows": rows, "analysis": {}}
    A = out["analysis"]
    A["pooled_pearson"] = corr([r["recon_rel_err"] for r in rows],
                               [r["ndcg_drop_pct"] for r in rows])
    A["per_model"] = {t: corr([r["recon_rel_err"] for r in rows if r["model"] == t],
                              [r["ndcg_drop_pct"] for r in rows if r["model"] == t])
                      for t in {r["model"] for r in rows}}
    live = [r for r in rows if r["ndcg_drop_pct"] < 50]      # 완전 붕괴 팔 제외
    A["excluding_collapsed_pooled"] = corr([r["recon_rel_err"] for r in live],
                                           [r["ndcg_drop_pct"] for r in live])
    A["excluding_collapsed_per_model"] = {
        t: corr([r["recon_rel_err"] for r in live if r["model"] == t],
                [r["ndcg_drop_pct"] for r in live if r["model"] == t])
        for t in {r["model"] for r in live}}
    A["note"] = ("붕괴 팔(손실 50% 초과)을 넣으면 상관이 부풀려진다 — 오차도 크고 손실도 "
                 "크니까. 배분이 실제로 필요한 구간은 '살아 있는' 팔이다.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("\n상관(전체):", A["pooled_pearson"], "· 붕괴 제외:", A["excluding_collapsed_pooled"])
    print("모델별(붕괴 제외):", A["excluding_collapsed_per_model"])
    print("→", a.out)
