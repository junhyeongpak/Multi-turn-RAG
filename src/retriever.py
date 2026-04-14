"""
검색 함수 단일 진실원.
- search_dense / search_bm25 / search_elser : ES 단일 모드 검색
- hybrid_rrf_search : dense + bm25 RRF 융합
- retrieve_docs(query, mode) : pipeline에서 쓰는 통합 dispatcher (문자열 반환)

다른 모든 모듈/스크립트는 여기서 import 만 한다.
"""

from typing import Literal

from src.config import (
    BM25_WEIGHT,
    DENSE_FIELD,
    DENSE_WEIGHT,
    ELSER_FIELD,
    ELSER_MODEL_ID,
    INDEX_NAME,
    NUM_CANDIDATES,
    RRF_K,
    TEXT_FIELD,
    TOP_K,
)
from src.es_client import get_es
from src.utils import get_embed_model

RetrievalMode = Literal["dense", "bm25", "elser", "hybrid"]


# ---------- 단일 모드 검색 ----------
def search_dense(query: str, top_k: int = TOP_K, num_candidates: int = NUM_CANDIDATES):
    es = get_es()
    qv = get_embed_model().encode(query, normalize_embeddings=True).tolist()
    res = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "_source": [TEXT_FIELD],
            "knn": {
                "field": DENSE_FIELD,
                "query_vector": qv,
                "k": top_k,
                "num_candidates": num_candidates,
            },
        },
    )
    return res["hits"]["hits"]


def search_bm25(query: str, top_k: int = TOP_K):
    es = get_es()
    res = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "_source": [TEXT_FIELD],
            "query": {"match": {TEXT_FIELD: query}},
        },
    )
    return res["hits"]["hits"]


def search_elser(query: str, top_k: int = TOP_K):
    es = get_es()
    res = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "_source": [TEXT_FIELD],
            "query": {
                "text_expansion": {
                    ELSER_FIELD: {
                        "model_id": ELSER_MODEL_ID,
                        "model_text": query,
                    }
                }
            },
        },
    )
    return res["hits"]["hits"]


# ---------- 하이브리드 (RRF) ----------
def hybrid_rrf_search(
    query: str,
    top_k: int = 10,
    final_k: int = TOP_K,
    num_candidates: int = NUM_CANDIDATES,
    rrf_k: int = RRF_K,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> list[dict]:
    """dense + bm25 결과를 RRF로 융합. final_k개의 dict 반환."""
    dense_hits = search_dense(query, top_k=top_k, num_candidates=num_candidates)
    bm25_hits = search_bm25(query, top_k=top_k)

    doc_scores: dict[str, dict] = {}

    def _ensure(doc_id: str, text: str):
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                "doc_id": doc_id,
                "text": text,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score_raw": None,
                "bm25_score_raw": None,
                "rrf_score": 0.0,
            }

    for rank, hit in enumerate(dense_hits, start=1):
        doc_id = hit["_id"]
        _ensure(doc_id, hit["_source"].get(TEXT_FIELD, ""))
        doc_scores[doc_id]["dense_rank"] = rank
        doc_scores[doc_id]["dense_score_raw"] = hit["_score"]
        doc_scores[doc_id]["rrf_score"] += dense_weight * (1 / (rrf_k + rank))

    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = hit["_id"]
        _ensure(doc_id, hit["_source"].get(TEXT_FIELD, ""))
        doc_scores[doc_id]["bm25_rank"] = rank
        doc_scores[doc_id]["bm25_score_raw"] = hit["_score"]
        doc_scores[doc_id]["rrf_score"] += bm25_weight * (1 / (rrf_k + rank))

    return sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)[:final_k]


# ---------- pipeline에서 쓰는 통합 dispatcher ----------
def retrieve(query: str, mode: RetrievalMode = "hybrid", top_k: int = TOP_K) -> list[dict]:
    """
    mode 별 검색 → 통일된 dict 리스트 반환.
    각 원소: {"doc_id", "text", "score"}
    """
    if mode == "dense":
        hits = search_dense(query, top_k=top_k)
        return [
            {"doc_id": h["_id"], "text": h["_source"].get(TEXT_FIELD, ""), "score": h["_score"]}
            for h in hits
        ]
    if mode == "bm25":
        hits = search_bm25(query, top_k=top_k)
        return [
            {"doc_id": h["_id"], "text": h["_source"].get(TEXT_FIELD, ""), "score": h["_score"]}
            for h in hits
        ]
    if mode == "elser":
        hits = search_elser(query, top_k=top_k)
        return [
            {"doc_id": h["_id"], "text": h["_source"].get(TEXT_FIELD, ""), "score": h["_score"]}
            for h in hits
        ]
    if mode == "hybrid":
        hits = hybrid_rrf_search(query, top_k=max(10, top_k * 2), final_k=top_k)
        return [
            {"doc_id": h["doc_id"], "text": h["text"], "score": h["rrf_score"]} for h in hits
        ]
    raise ValueError(f"unknown retrieval mode: {mode}")


def retrieve_as_context(query: str, mode: RetrievalMode = "hybrid", top_k: int = TOP_K) -> str:
    """LLM 프롬프트의 {context}에 그대로 끼울 수 있는 문자열로 포맷."""
    docs = retrieve(query, mode=mode, top_k=top_k)
    if not docs:
        return "(검색 결과 없음)"
    return "\n\n".join(
        f"[문서 {i} | score={d['score']:.3f}] {d['text']}" for i, d in enumerate(docs, 1)
    )
