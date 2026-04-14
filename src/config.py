"""
프로젝트 전역 설정.
다른 모듈/스크립트는 여기서만 상수를 import 한다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------- 경로 ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
RESULTS_DIR = OUTPUTS_DIR / "results"

# 데이터 파일명 (전처리/임베딩/평가에서 공통 사용)
RAW_FILE = RAW_DIR / "fiqa.jsonl"
PROCESSED_FILE = PROCESSED_DIR / "preprocessed.jsonl"
EMBEDDED_FILE = PROCESSED_DIR / "preprocessed_vectors.jsonl"
EMBED_CHECKPOINT = PROCESSED_DIR / "dense_checkpoint.txt"
EVAL_FILE = EVAL_DIR / "fiqa_rewrite.jsonl"

# ---------- Elasticsearch ----------
ES_URL = os.getenv("ES_URL", "")
ES_API_KEY = os.getenv("ES_API_KEY", "")

INDEX_NAME = "hybrid_index"
TEXT_FIELD = "text"
DENSE_FIELD = "embedding"
ELSER_FIELD = "ml.tokens"
ELSER_MODEL_ID = ".elser_model_2_linux-x86_64"
ELSER_PIPELINE_ID = "elser_pipeline"

# ---------- Embedding ----------
EMBED_MODEL_NAME = "philschmid/bge-base-financial-matryoshka"
EMBED_DIM = 768
EMBED_MAX_LEN = 512
EMBED_BATCH_SIZE = 512

# ---------- Retrieval ----------
TOP_K = 5
NUM_CANDIDATES = 100
RRF_K = 60
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5

# 검색 모드 기본값: "dense" | "bm25" | "elser" | "hybrid"
DEFAULT_RETRIEVAL_MODE = "hybrid"

# ---------- LLM ----------
ANSWER_MODEL_ID = "Qwen/Qwen3-8B"
REWRITE_MODEL = "gpt-4o-mini"
REWRITE_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.3
ANSWER_MAX_NEW_TOKENS = 512
