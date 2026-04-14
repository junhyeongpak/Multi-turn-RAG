"""
preprocessed.jsonl -> preprocessed_vectors.jsonl
체크포인트 재개 지원.
임베딩 모델은 src.utils.get_embed_model() 에서 캐싱.
"""

import json
import os

from tqdm import tqdm

from src.config import (
    EMBED_BATCH_SIZE,
    EMBED_CHECKPOINT,
    EMBED_DIM,
    EMBEDDED_FILE,
    PROCESSED_FILE,
)
from src.utils import count_lines, get_embed_model


def load_checkpoint() -> int:
    if EMBED_CHECKPOINT.exists():
        return int(EMBED_CHECKPOINT.read_text().strip() or 0)
    return 0


def save_checkpoint(idx: int) -> None:
    EMBED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    EMBED_CHECKPOINT.write_text(str(idx))


def main():
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"입력 없음: {PROCESSED_FILE}. 먼저 scripts/preprocess.py 실행.")

    EMBEDDED_FILE.parent.mkdir(parents=True, exist_ok=True)

    model = get_embed_model()
    print(f"모델 로딩 완료: {model}")

    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    print(f"총 레코드: {total:,}")

    start_idx = max(load_checkpoint(), count_lines(EMBEDDED_FILE))
    if start_idx >= total:
        print("이미 완료 상태")
        return

    print(f"시작 지점: {start_idx:,}")
    write_mode = "a" if EMBEDDED_FILE.exists() else "w"

    with open(EMBEDDED_FILE, write_mode, encoding="utf-8") as out_f:
        for i in tqdm(range(start_idx, total, EMBED_BATCH_SIZE), desc="임베딩"):
            batch = [json.loads(lines[j]) for j in range(i, min(i + EMBED_BATCH_SIZE, total))]
            texts = [d["text"] for d in batch]
            embs = model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for rec, vec in zip(batch, embs):
                rec["text_vector"] = vec.tolist()
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            save_checkpoint(i + len(batch))

    save_checkpoint(total)
    print(f"\n완료! {count_lines(EMBEDDED_FILE):,}줄 (입력 {total:,}줄)")

    # 검증
    assert count_lines(EMBEDDED_FILE) == total, "줄 수 불일치!"
    sample = json.loads(open(EMBEDDED_FILE).readline())
    assert len(sample["text_vector"]) == EMBED_DIM, "벡터 차원 불일치!"
    print(f"검증 통과: {EMBED_DIM}차원")


if __name__ == "__main__":
    main()
