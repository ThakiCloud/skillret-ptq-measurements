#!/usr/bin/env python3
"""HF 의 SkillRet 리비전을 quant_sweep.py 가 기대하는 로컬 JSONL 세 개로 만든다.

⛔ 재현 절차에 이 단계가 없으면 독자는 "--corpus skillret:/path 에 무엇을 넣지?"
   에서 막힌다. Hub head 는 mutable 하므로 리비전을 반드시 고정한다.
"""
import argparse, os, urllib.request

BASE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/data/{src}.jsonl"
FILES = [("queries/test", "data_queries_test.jsonl"),
         ("skills/test",  "data_skills_test.jsonl"),
         ("qrels/test",   "data_qrels_test.jsonl")]
DEFAULT_REV = "a050ad233a504a43135bafe8cdf45574052b5729"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ThakiCloud/SKILLRET")
    ap.add_argument("--revision", default=DEFAULT_REV)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for src, out in FILES:
        url = BASE.format(repo=a.repo, rev=a.revision, src=src)
        dst = os.path.join(a.dst, out)
        with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as fh:
            fh.write(r.read())
        n = sum(1 for _ in open(dst, encoding="utf-8"))
        print(f"  {out:28s} {n:>6d} 줄")
    print(f"→ {a.dst}\n  써먹기: --corpus skillret:{a.dst}")


if __name__ == "__main__":
    raise SystemExit(main())
