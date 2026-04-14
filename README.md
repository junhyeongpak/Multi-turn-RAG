# multiturn-rag

멀티턴 대화형 RAG 파이프라인.
**Query Rewrite (OpenAI)** → **Hybrid Retrieval (Elasticsearch: Dense + BM25 + ELSER)** → **Answer Generation (Qwen3-8B 4bit)**.

## 디렉토리

```
multiturn-rag/
├─ data/                 # raw / processed / eval
├─ scripts/              # 1회성 실행 스크립트 (전처리, 임베딩, 인덱싱, 검색/평가)
├─ src/                  # import 해서 쓰는 모듈 (config, retriever, pipeline 등)
├─ notebooks/            # 실험용 노트북
└─ outputs/              # 로그 / 결과
```

## 설치

```bash
pip install -r requirements.txt
cp .env.example .env     # 키 채우기
```

## 실행 순서

```bash
# 1. 원본 jsonl(fiqa.jsonl)을 data/raw/ 에 두고 전처리
python scripts/preprocess.py

# 2. dense embedding 생성 (philschmid/bge-base-financial-matryoshka, 768d)
python scripts/make_embeddings.py

# 3. ES 인덱스 생성 + ELSER pipeline 등록 + bulk 인덱싱
python scripts/index_to_es.py

# 4. 검색 동작 확인 (dense / bm25 / elser / hybrid_rrf)
python scripts/search_es.py

# 5. retrieval 평가 (Recall@K, MRR)
python scripts/evaluate.py

# 6. 멀티턴 RAG 데모
python -m src.pipeline
```

## 모듈 책임 (중복 방지)

| 위치 | 책임 |
|---|---|
| `src/config.py` | 모든 상수 (ES 주소, 인덱스명, 필드명, 모델명, top_k, 가중치 등) |
| `src/es_client.py` | ES 연결 1군데 (`get_es()`) |
| `src/utils.py` | jsonl 로딩, embed model 캐시 |
| `src/retriever.py` | dense / bm25 / elser / hybrid_rrf 검색 함수 (단일 진실원) |
| `src/query_rewrite.py` | 멀티턴 → standalone 쿼리 재작성 체인 |
| `src/memory.py` | 세션별 chat history |
| `src/pipeline.py` | rewrite → retrieve → generate 전체 체인 |
| `scripts/*` | `src` 모듈을 import 해서 실행만 |
