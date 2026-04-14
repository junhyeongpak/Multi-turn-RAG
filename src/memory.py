"""
LangGraph 용 체크포인터 (세션별 state 저장소).

- InMemorySaver : 프로세스 재시작 시 날아감 (데모/테스트용)
- SqliteSaver   : 파일로 영속화 (프로덕션은 Postgres 등으로 교체)

pipeline.py 는 get_checkpointer() 하나만 호출하면 된다.
"""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver


@lru_cache(maxsize=1)
def get_checkpointer() -> InMemorySaver:
    """세션 state를 보관하는 체크포인터 (싱글톤)."""
    return InMemorySaver()


def reset_all() -> None:
    """전체 세션 초기화 (캐시된 saver를 버림)."""
    get_checkpointer.cache_clear()
