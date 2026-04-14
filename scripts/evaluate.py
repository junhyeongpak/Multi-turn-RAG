"""
검색 평가 (Recall@K, MRR@K).
data/eval/fiqa_rewrite.jsonl 형식 가정:
  {"query": "...", "relevant_ids": ["123", "456"]}
또는
  {"question": "...", "answers": [{"_id": "..."} , ...]}

mode 별 / k 별 점수를 outputs/results/ 에 저장.

사용법:
  python scripts/evaluate.py
  python scripts/evaluate.py --modes dense bm25 hybrid --ks 1 3 5 10
"""

import argparse
import csv
from datetime import datetime

from src.config import EVAL_FILE, RESULTS_DIR
from src.retriever import retrieve
from src.utils import load_jsonl


def extract_relevant_ids(record: dict) -> list[str]:
    if "relevant_ids" in record:
        return [str(x) for x in record["relevant_ids"]]
    if "answers" in record:
        out = []
        for a in record["answers"]:
            if isinstance(a, dict) and "_id" in a:
                out.append(str(a["_id"]))
            else:
                out.append(str(a))
        return out
    if "answer_ids" in record:
        return [str(x) for x in record["answer_ids"]]
    raise KeyError(f"relevant id 필드를 찾을 수 없음: {list(record.keys())}")


def extract_query(record: dict) -> str:
    for key in ("query", "question", "rewrite", "input"):
        if key in record and record[key]:
            return record[key]
    raise KeyError(f"query 필드를 찾을 수 없음: {list(record.keys())}")


def recall_at_k(retrieved_ids: list[str], gold: set[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    return len(top & gold) / len(gold) if gold else 0.0


def reciprocal_rank(retrieved_ids: list[str], gold: set[str], k: int) -> float:
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


def evaluate(mode: str, queries: list[dict], ks: list[int]) -> dict:
    max_k = max(ks)
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums |= {f"mrr@{k}": 0.0 for k in ks}
    n = 0

    for rec in queries:
        query = extract_query(rec)
        gold = set(extract_relevant_ids(rec))
        if not gold:
            continue
        docs = retrieve(query, mode=mode, top_k=max_k)
        retrieved_ids = [d["doc_id"] for d in docs]
        for k in ks:
            sums[f"recall@{k}"] += recall_at_k(retrieved_ids, gold, k)
            sums[f"mrr@{k}"] += reciprocal_rank(retrieved_ids, gold, k)
        n += 1

    return {"mode": mode, "n": n, **{k: v / n if n else 0.0 for k, v in sums.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["dense", "bm25", "elser", "hybrid"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--limit", type=int, default=0, help="평가 쿼리 수 제한 (0=전체)")
    args = parser.parse_args()

    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"평가 파일 없음: {EVAL_FILE}")

    queries = load_jsonl(EVAL_FILE)
    if args.limit:
        queries = queries[: args.limit]
    print(f"평가 쿼리 수: {len(queries):,}")

    rows = []
    for mode in args.modes:
        print(f"\n>>> [{mode}] 평가 중...")
        row = evaluate(mode, queries, args.ks)
        print(row)
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
