"""
data/raw/fiqa.jsonl 전처리 -> data/processed/preprocessed.jsonl
- _id, text 만 남김
- 간단한 검증 통계 출력 (id 중복, URL 포함, title 빈값, text 길이)
"""

import re
from collections import Counter

from src.config import PROCESSED_FILE, RAW_FILE
from src.utils import load_jsonl, save_jsonl

URL_PATTERN = re.compile(r"https?://[^\s\"\'\)\]]+", re.IGNORECASE)


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"원본 파일 없음: {RAW_FILE}")

    records = load_jsonl(RAW_FILE)
    print(f"총 레코드 수: {len(records):,}")
    print(f"샘플 키: {list(records[0].keys())}")

    # _id vs id
    mismatch = [(r["_id"], r["id"]) for r in records if r.get("_id") != r.get("id")]
    print(f"\n_id != id : {len(mismatch):,}")

    # 중복 _id
    id_count = Counter(r["_id"] for r in records)
    dup = {k: v for k, v in id_count.items() if v > 1}
    print(f"중복 _id  : {len(dup):,}")

    # URL 포함
    url_n = sum(1 for r in records if URL_PATTERN.search(r.get("text", "")[:300]))
    print(f"URL 포함  : {url_n:,} ({url_n/len(records)*100:.2f}%)")

    # title
    nonempty_title = sum(1 for r in records if (r.get("title") or "").strip())
    print(f"title 값있음: {nonempty_title:,}")

    # _id, text 만 남기기
    final = [{"_id": r["_id"], "text": r["text"]} for r in records]

    empty_text = sum(1 for r in final if not r["text"].strip())
    short_text = sum(1 for r in final if 0 < len(r["text"].strip()) < 20)
    print(f"\ntext 비어있음: {empty_text:,}")
    print(f"text <20자  : {short_text:,}")

    n = save_jsonl(final, PROCESSED_FILE)
    print(f"\n저장 완료: {PROCESSED_FILE}  ({n:,}개)")


if __name__ == "__main__":
    main()
