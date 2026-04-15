from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.memory import get_checkpointer, reset_all
from src.pipeline import build_graph

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    _graph = build_graph()
    yield


app = FastAPI(title="Multi-turn RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    message: str
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    rewritten_query: str
    rewrite_class: str
    context: str


class ResetRequest(BaseModel):
    session_id: str | None = None


@app.get("/health") #클라우드 띄울때 표준이라고 함
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    config = {"configurable": {"thread_id": req.session_id}}

    try:
        result = _graph.invoke(
            {"input": req.message, "retrieval_mode": req.retrieval_mode},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        session_id=req.session_id,
        answer=result.get("generated_answer", ""),
        rewritten_query=result.get("rewritten_query", ""),
        rewrite_class=result.get("rewrite_class", ""),
        context=result.get("context", ""),
    )


@app.post("/sessions/reset")
async def reset_session(req: ResetRequest):
    if req.session_id is None:
        reset_all()
        return {"detail": "All sessions cleared"}

    checkpointer = get_checkpointer()
    storage = getattr(checkpointer, "storage", None)
    if storage and req.session_id in storage:
        del storage[req.session_id]

    return {"detail": f"Session {req.session_id} cleared"}
