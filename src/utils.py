"""
공용 유틸리티.
- jsonl read/write
- SentenceTransformer 임베딩 모델 캐싱 (여러 모듈에서 모델을 두 번 로드하지 않도록)
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import torch
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MAX_LEN, EMBED_MODEL_NAME


# ---------- jsonl I/O ----------
def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """jsonl을 한 줄씩 yield (메모리 절약)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def save_jsonl(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def count_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ---------- embedding model 캐시 ----------
@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    """프로젝트 전역에서 동일한 임베딩 모델 인스턴스 사용."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True, device=device)
    model.max_seq_length = EMBED_MAX_LEN
    return model
