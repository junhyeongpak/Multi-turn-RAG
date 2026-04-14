"""
멀티턴 대화 → standalone 쿼리 재작성 체인.
OpenAI(gpt-4o-mini) 사용. 출력은 JSON 파싱하여 'reworded version' 만 반환.
"""

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from src.config import REWRITE_MODEL, REWRITE_TEMPERATURE

REWRITE_SYSTEM = """Given the following conversation, please reword the final utterance from the user into a single utterance that does not need the history to understand the user's intent. Output in proper JSON format indicating the "class" (standalone or non-standalone) and the "reworded version" of the last utterance. Use this format: {{"class": "type of last utterance", "reworded version": "the last utterance rewritten into a standalone question, IF NEEDED"}}.

In your rewording of the last utterance, do not do any unnecessary rephrasing or introduction of new terms or concepts that were not mentioned in the prior part of the conversation. Be minimal, by staying as close as possible to the shape and meaning of the last user utterance. If the last user utterance is already clear and standalone, the reworded version should be THE SAME as the last user utterance, and the class should be 'standalone'."""


def parse_rewrite_json(raw: str) -> str:
    try:
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        parsed = json.loads(cleaned)
        reworded = parsed.get("reworded version", "").strip()
        cls = parsed.get("class", "").strip()
        print(f"[재작성 결과] class={cls}, query={reworded}")
        return reworded if reworded else raw
    except Exception as e:
        print(f"[경고] JSON 파싱 실패 ({e}), 원문 사용")
        return raw


def build_rewrite_chain() -> Runnable:
    llm = ChatOpenAI(model=REWRITE_MODEL, temperature=REWRITE_TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm | StrOutputParser() | RunnableLambda(parse_rewrite_json)
