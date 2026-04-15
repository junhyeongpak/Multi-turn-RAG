from __future__ import annotations

import uuid

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"
RETRIEVAL_MODES = ["hybrid", "dense", "bm25"]

st.set_page_config(page_title="Multi-turn RAG", page_icon="🔎", layout="centered")

# session state 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

# sidebar
with st.sidebar:
    st.header("Mode for Retrieval")
    retrieval_mode = st.selectbox("Retrieval mode", RETRIEVAL_MODES)

    if st.button("New conversation"):
        cur_msgs = st.session_state.messages
        if cur_msgs:
            st.session_state.conversations[st.session_state.session_id] = cur_msgs.copy()
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

    st.divider()

    for sid, msgs in reversed(list(st.session_state.conversations.items())):
        first_q = next((m["content"] for m in msgs if m["role"] == "user"), None)
        if first_q:
            label = first_q[:30] + "..." if len(first_q) > 30 else first_q
            if st.button(label, key=sid):
                cur_msgs = st.session_state.messages
                if cur_msgs:
                    st.session_state.conversations[st.session_state.session_id] = cur_msgs.copy()
                st.session_state.conversations.pop(sid, None)
                st.session_state.session_id = sid
                st.session_state.messages = msgs
                st.rerun()

# 제목
st.title("Multi-turn RAG Chat")

# 대화 이력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("context"):
            with st.expander("Retrieved context"):
                st.text(msg["context"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "retrieval_mode": retrieval_mode,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data["answer"]
                context = data.get("context", "")

                st.markdown(answer)
                if context:
                    with st.expander("Retrieved context"):
                        st.text(context)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "context": context}
                )

            except httpx.ConnectError:
                st.error(
                    "FastAPI 서버에 연결할 수 없습니다. "
                    "`uvicorn api.main:app --port 8000` 으로 서버를 먼저 실행하세요."
                )
            except Exception as e:
                st.error(f"Error: {e}")
