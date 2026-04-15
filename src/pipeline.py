from __future__ import annotations

from functools import lru_cache
from typing import Annotated, TypedDict

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from src.config import (
    ANSWER_MAX_NEW_TOKENS,
    ANSWER_MODEL_ID,
    ANSWER_TEMPERATURE,
    DEFAULT_RETRIEVAL_MODE,
    TOP_K,
)
from src.memory import get_checkpointer
from src.query_rewrite import build_rewrite_chain
from src.retriever import retrieve_as_context


ANSWER_SYSTEM = """You are a helpful assistant answering questions using retrieved documents.

Answer the user's question using only the retrieved documents below.
Do not make up facts.
If the answer cannot be found in the retrieved documents, say: "I don't know."

Retrieved documents:
{context}
"""


class GraphState(TypedDict, total=False):
    input: str
    messages: Annotated[list[BaseMessage], add_messages]
    rewritten_query: str
    rewrite_class: str
    rewrite_raw: str
    context: str
    generated_answer: str


@lru_cache(maxsize=1)
def load_answer_llm() -> ChatHuggingFace:
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU is required.")

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

    original_apply_chat_template = tokenizer.apply_chat_template

    def _apply_no_think(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original_apply_chat_template(*args, **kwargs)

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


def build_graph(
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    top_k: int = TOP_K,
):
    rewrite_chain = build_rewrite_chain()
    answer_prompt = _build_answer_prompt()
    answer_llm = load_answer_llm()
    answer_chain = answer_prompt | answer_llm

    def answer_turn(state: GraphState) -> GraphState:
        history = list(state.get("messages", []))
        original_input = state["input"]

        if not history:
            rewritten_query = original_input
            rewrite_class = "standalone"
            rewrite_raw = original_input
            print(f"[Skip] First turn, no rewriting: {rewritten_query}")
        else:
            rewrite_result = rewrite_chain.invoke(
                {
                    "input": original_input,
                    "chat_history": history,
                }
            )
            rewritten_query = rewrite_result.get("rewritten_query", original_input)
            rewrite_class = rewrite_result.get("rewrite_class", "standalone")
            rewrite_raw = rewrite_result.get("rewrite_raw", original_input)

        print(f"[Search Query] ({retrieval_mode}) {rewritten_query}")
        context = retrieve_as_context(rewritten_query, mode=retrieval_mode, top_k=top_k)

        answer = answer_chain.invoke(
            {
                "input": original_input,
                "chat_history": history,
                "context": context,
            }
        )

        generated_answer = getattr(answer, "content", str(answer))
        ai_message = answer if isinstance(answer, AIMessage) else AIMessage(content=generated_answer)

        return {
            "messages": [
                HumanMessage(content=original_input),
                ai_message,
            ],
            "generated_answer": generated_answer,
            "rewritten_query": rewritten_query,
            "rewrite_class": rewrite_class,
            "rewrite_raw": rewrite_raw,
            "context": context,
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("answer_turn", answer_turn)
    workflow.add_edge(START, "answer_turn")
    workflow.add_edge("answer_turn", END)

    return workflow.compile(checkpointer=get_checkpointer())