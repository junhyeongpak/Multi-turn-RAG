"""
멀티턴 RAG 파이프라인 (LangGraph 버전).

흐름:
  START
    → summarize_node   (langmem: 대화가 길어지면 오래된 부분 자동 요약)
    → rewrite_node     (멀티턴 → standalone 쿼리, gpt-4o-mini)
    → retrieve_node    (ES hybrid: dense + BM25 RRF)
    → generate_node    (Qwen3-8B 4bit, 검색 문서 기반 답변)
    → END

세션 영속성은 MemorySaver (checkpointer) 가 담당. thread_id 별로 state 유지.
"""

from functools import lru_cache
from typing import Annotated

import torch
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langmem.short_term import SummarizationNode
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from typing_extensions import TypedDict

from src.config import (
    ANSWER_MAX_NEW_TOKENS,
    ANSWER_MODEL_ID,
    ANSWER_TEMPERATURE,
    DEFAULT_RETRIEVAL_MODE,
    REWRITE_MODEL,
    REWRITE_TEMPERATURE,
    TOP_K,
)
from src.memory import get_checkpointer
from src.query_rewrite import build_rewrite_chain
from src.retriever import retrieve_as_context

ANSWER_SYSTEM = """You are an assistant that answers questions based on the provided documents.
Use the retrieved documents below to answer the user's question accurately.
Do not make up information that is not in the documents. If the answer is not in the documents, say you don't know.
Always answer in English.

[Retrieved Documents]
{context}"""


# ---------- State ----------
class RAGState(TypedDict):
    """그래프 전체가 공유하는 상태."""

    messages: Annotated[list[AnyMessage], add_messages]
    # langmem 이 요약 캐시를 넣어주는 자리 (필수 키)
    context: dict
    # 노드 간 전달용
    rewritten_query: str
    retrieved_context: str


# ---------- LLM 로더 ----------
@lru_cache(maxsize=1)
def load_answer_llm() -> ChatHuggingFace:
    """Qwen3-8B 4bit 싱글톤."""
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU 환경이 필요합니다.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ANSWER_MODEL_ID,
        trust_remote_code=True,
        device_map={"": 0},
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
    )

    # Qwen3 thinking 모드 끔
    _orig_apply = tokenizer.apply_chat_template

    def _apply_no_think(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig_apply(*args, **kwargs)

    tokenizer.apply_chat_template = _apply_no_think

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id, im_end_id]

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=ANSWER_MAX_NEW_TOKENS,
        temperature=ANSWER_TEMPERATURE,
        do_sample=True,
        repetition_penalty=1.05,
        return_full_text=False,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
    base_llm = HuggingFacePipeline(pipeline=gen_pipeline)
    return ChatHuggingFace(llm=base_llm, tokenizer=tokenizer)


@lru_cache(maxsize=1)
def _get_rewrite_chain():
    return build_rewrite_chain()


@lru_cache(maxsize=1)
def _get_summarization_node() -> SummarizationNode:
    """
    langmem 공식 summarization 노드.
    - max_tokens: 이 토큰 넘으면 오래된 메시지를 요약해 SystemMessage 로 대체
    - max_summary_tokens: 요약 자체의 상한
    """
    summarizer_llm = ChatOpenAI(model=REWRITE_MODEL, temperature=REWRITE_TEMPERATURE)
    return SummarizationNode(
        model=summarizer_llm,
        max_tokens=1024,
        max_summary_tokens=256,
    )


# ---------- 노드들 ----------
def rewrite_node(state: RAGState) -> dict:
    """마지막 human 메시지를 standalone 쿼리로 변환."""
    msgs = state["messages"]
    last_human = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        return {"rewritten_query": ""}

    # 첫 턴이면 스킵
    prior = [m for m in msgs if m is not last_human]
    if not prior:
        print(f"[스킵] 첫 턴이라 재작성 생략: {last_human.content}")
        return {"rewritten_query": last_human.content}

    rewritten = _get_rewrite_chain().invoke(
        {"input": last_human.content, "chat_history": prior}
    )
    return {"rewritten_query": rewritten}


def make_retrieve_node(retrieval_mode: str, top_k: int):
    def retrieve_node(state: RAGState) -> dict:
        q = state["rewritten_query"]
        print(f"[검색 쿼리] ({retrieval_mode}) {q}")
        ctx = retrieve_as_context(q, mode=retrieval_mode, top_k=top_k)
        return {"retrieved_context": ctx}

    return retrieve_node


def generate_node(state: RAGState) -> dict:
    """Qwen3-8B로 최종 답변 생성."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM),
            MessagesPlaceholder("chat_history"),
        ]
    )
    llm = load_answer_llm()
    chain = prompt | llm

    # summarize_node 가 요약한 메시지 리스트를 우선 사용
    summarized_msgs = state.get("context", {}).get("summarized_messages")
    history: list[BaseMessage] = summarized_msgs or state["messages"]

    result = chain.invoke(
        {"context": state["retrieved_context"], "chat_history": history}
    )
    answer = result.content if hasattr(result, "content") else str(result)
    return {"messages": [AIMessage(content=answer)]}


# ---------- 그래프 빌드 ----------
@lru_cache(maxsize=4)
def build_graph(retrieval_mode: str = DEFAULT_RETRIEVAL_MODE, top_k: int = TOP_K):
    graph = StateGraph(RAGState)

    graph.add_node("summarize", _get_summarization_node())
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", make_retrieve_node(retrieval_mode, top_k))
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=get_checkpointer())


# ---------- 공개 API ----------
def ask(question: str, session_id: str = "default", retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        top_k: int = TOP_K) -> str:
    """한 턴 실행. 같은 session_id 로 연속 호출하면 멀티턴."""
    app = build_graph(retrieval_mode=retrieval_mode, top_k=top_k)
    config = {"configurable": {"thread_id": session_id}}
    out = app.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return out["messages"][-1].content


# ---------- 데모 ----------
if __name__ == "__main__":
    print("=" * 60, "\n[Turn 1]")
    r1 = ask("What's the difference between Market Cap and NAV?", session_id="user_002")
    print(f"\n[답변]\n{r1}\n")

    print("=" * 60, "\n[Turn 2]")
    r2 = ask("Which is more important?", session_id="user_002")
    print(f"\n[답변]\n{r2}\n")
