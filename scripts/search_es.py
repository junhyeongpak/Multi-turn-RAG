"""
검색 모드별 동작 확인.
모든 검색 함수는 src/retriever.py 에 정의되어 있으며, 여기서는 import만 한다.

사용법:
  python scripts/search_es.py
  python scripts/search_es.py --query "your question" --mode hybrid --k 5
"""

import argparse

from src.config import TOP_K
from src.retriever import retrieve

DEFAULT_QUERY = "What's the difference between Market Cap and NAV?"


def show(mode: str, query: str, k: int):
    print("\n" + "=" * 60)
    print(f"[{mode.upper()}] {query}")
    print("=" * 60)
    try:
        docs = retrieve(query, mode=mode, top_k=k)
        for i, d in enumerate(docs, 1):
            preview = d["text"][:200].replace("\n", " ")
            print(f"\n[{i}] id={d['doc_id']}  score={d['score']:.4f}")
            print(f"    {preview}")
    except Exception as e:
        print(f"[ERROR] {mode}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--mode", default="all", choices=["all", "dense", "bm25", "elser", "hybrid"])
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    modes = ["dense", "bm25", "elser", "hybrid"] if args.mode == "all" else [args.mode]
    for m in modes:
        show(m, args.query, args.k)


if __name__ == "__main__":
    main()
