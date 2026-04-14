"""
Elasticsearch 인덱스 생성 -> ELSER ingest pipeline 등록 -> bulk 인덱싱.

사용법:
  python scripts/index_to_es.py             # 인덱스 없으면 생성, 있으면 그대로 사용
  python scripts/index_to_es.py --recreate  # 기존 인덱스 삭제 후 재생성
"""

import argparse
import json

from elasticsearch.helpers import bulk

from src.config import (
    DENSE_FIELD,
    ELSER_MODEL_ID,
    ELSER_PIPELINE_ID,
    EMBED_DIM,
    EMBEDDED_FILE,
    INDEX_NAME,
    TEXT_FIELD,
)
from src.es_client import get_es

INDEX_BODY = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            TEXT_FIELD: {"type": "text", "analyzer": "english"},
            DENSE_FIELD: {
                "type": "dense_vector",
                "dims": EMBED_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "ml.tokens": {"type": "sparse_vector"},
        }
    },
}

PIPELINE_BODY = {
    "processors": [
        {
            "inference": {
                "model_id": ELSER_MODEL_ID,
                "input_output": [
                    {"input_field": TEXT_FIELD, "output_field": "ml.tokens"}
                ],
            }
        }
    ]
}


def ensure_index(recreate: bool = False):
    es = get_es()
    if es.indices.exists(index=INDEX_NAME):
        if recreate:
            es.indices.delete(index=INDEX_NAME)
            print(f"인덱스 삭제: {INDEX_NAME}")
        else:
            print(f"{INDEX_NAME} 이미 존재 (재생성하려면 --recreate)")
            return
    es.indices.create(index=INDEX_NAME, body=INDEX_BODY)
    print(f"{INDEX_NAME} 생성 완료")


def ensure_pipeline():
    es = get_es()
    es.ingest.put_pipeline(id=ELSER_PIPELINE_ID, body=PIPELINE_BODY)
    print(f"ingest pipeline 등록: {ELSER_PIPELINE_ID}")


def generate_actions(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                yield {
                    "_op_type": "index",
                    "_index": INDEX_NAME,
                    "_id": str(doc["_id"]),
                    "_source": {
                        TEXT_FIELD: doc["text"],
                        DENSE_FIELD: doc["text_vector"],
                    },
                    "pipeline": ELSER_PIPELINE_ID,
                }
            except Exception as e:
                print(f"{line_num}번째 줄 오류: {e}")


def bulk_index():
    if not EMBEDDED_FILE.exists():
        raise FileNotFoundError(
            f"임베딩 파일 없음: {EMBEDDED_FILE}. 먼저 scripts/make_embeddings.py 실행."
        )
    es = get_es()
    success, errors = bulk(
        es,
        generate_actions(EMBEDDED_FILE),
        chunk_size=50,
        request_timeout=300,
        raise_on_error=False,
    )
    print(f"성공: {success}  에러: {len(errors)}")
    print(f"인덱스 문서 수: {es.count(index=INDEX_NAME)['count']:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="기존 인덱스 삭제 후 재생성")
    args = parser.parse_args()

    ensure_index(recreate=args.recreate)
    ensure_pipeline()
    bulk_index()


if __name__ == "__main__":
    main()
