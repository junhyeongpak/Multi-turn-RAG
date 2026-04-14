"""
세션별 in-memory chat history.
프로덕션에서는 Redis/SQL 백엔드로 교체.
"""

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

_session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def reset_session(session_id: str) -> None:
    _session_store.pop(session_id, None)


def list_sessions() -> list[str]:
    return list(_session_store.keys())
