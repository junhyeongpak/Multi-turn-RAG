"""
전체 멀티턴 RAG 체인.
구성: query_rewrite (multiturn -> standalone) -> retriever (hybrid) -> Qwen3-8B 답변
"""

from functools import lru_cache

import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from src.config import (
    ANSWER_MAX_NEW_TOKENS,
    ANSWER_MODEL_ID,
    ANSWER_TEMPERATURE,
    DEFAULT_RETRIEVAL_MODE,
    TOP_K,
)
from src.memory import get_session_history
from src.query_rewrite import build_rewrite_chain
from src.retriever import retrieve_as_context

ANSWER_SYSTEM = """You are an assistant that answers questions based on the provided documents.
Use the retrieved documents below to answer the user's question accurately.
Do not make up information that is not in the documents. If the answer is not in the documents, say you don't know.
Always answer in English.

[Retrieved Documents]
{context}"""


@lru_cache(maxsize=1)
def load_answer_llm() -> ChatHuggingFace:
    """Qwen3-8B 4bit 로딩 (싱글톤)."""
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

    # Qwen3 chat template - thinking 모드 끔
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


def _build_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_rag_chain(retrieval_mode: str = DEFAULT_RETRIEVAL_MODE, top_k: int = TOP_K) -> Runnable:
    """rewrite -> retrieve -> generate 체인."""
    rewrite_chain = build_rewrite_chain()
    answer_prompt = _build_answer_prompt()
    answer_llm = load_answer_llm()

    def prepare_inputs(inputs: dict) -> dict:
        history = inputs.get("chat_history", [])
        original_input = inputs["input"]

        if not history:
            rewritten = original_input
            print(f"[스킵] 첫 턴이라 재작성 생략: {rewritten}")
        else:
            rewritten = rewrite_chain.invoke(
                {"input": original_input, "chat_history": history}
            )

        print(f"[검색 쿼리] ({retrieval_mode}) {rewritten}")
        context = retrieve_as_context(rewritten, mode=retrieval_mode, top_k=top_k)
        return {
            "input": original_input,
            "chat_history": history,
            "context": context,
        }

    return RunnableLambda(prepare_inputs) | answer_prompt | answer_llm | StrOutputParser()


def build_conversational_rag(
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE, top_k: int = TOP_K
) -> RunnableWithMessageHistory:
    """history 자동 관리되는 멀티턴 래퍼."""
    return RunnableWithMessageHistory(
        build_rag_chain(retrieval_mode=retrieval_mode, top_k=top_k),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# ---------- 데모 ----------
if __name__ == "__main__":
    rag = build_conversational_rag()
    config = {"configurable": {"session_id": "user_002"}}

    print("=" * 60, "\n[Turn 1]")
    r1 = rag.invoke(
        {"input": "What's the difference between Market Cap and NAV?"}, config=config
    )
    print(f"\n[답변]\n{r1}\n")

    print("=" * 60, "\n[Turn 2]")
    r2 = rag.invoke({"input": "Which is more important?"}, config=config)
    print(f"\n[답변]\n{r2}\n")
