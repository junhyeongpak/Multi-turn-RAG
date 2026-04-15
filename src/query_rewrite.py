from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from src.config import REWRITE_MODEL, REWRITE_TEMPERATURE


REWRITE_SYSTEM = """You are a query rewriting assistant.

Given the following conversation, reword the final user utterance into a single utterance that does not need the conversation history to understand the user's intent.

Return valid JSON only with this schema:
{{
  "class": "standalone" | "non-standalone",
  "reworded version": "<the last utterance rewritten into a standalone question, IF NEEDED>"
}}

Rewriting rules:
- Be MINIMAL. Stay as close as possible to the shape and meaning of the last user utterance.
- Do NOT introduce new terms or concepts that were not mentioned earlier in the conversation.
- Do NOT do unnecessary rephrasing.
- Only resolve coreferences, ellipses, or ambiguity using the prior conversation.
- The output should be a natural-language question/utterance, NOT a keyword search query.
- If the last user utterance is already clear and standalone, set "class" to "standalone" and return the utterance UNCHANGED in "reworded version".
- Otherwise, set "class" to "non-standalone".

Output rules:
- Output JSON only. No markdown code fences. No explanations. Do not answer the question.

Examples:
Conversation:
User: Who is the CEO of Apple Inc.?
Agent: The CEO of Apple Inc. is Tim Cook.
User: its address?
Output: {{"class": "non-standalone", "reworded version": "What is the address of Apple Inc.?"}}

Conversation:
User: What is the capital of France?
Output: {{"class": "standalone", "reworded version": "What is the capital of France?"}}
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _last_question_like_span(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""

    parts = re.split(r"(?<=[?.!])\s+", cleaned)
    question_parts = [p.strip() for p in parts if "?" in p]
    if question_parts:
        return question_parts[-1]

    short_parts = [p.strip() for p in parts if 0 < len(p.strip()) <= 160]
    if short_parts:
        return short_parts[-1]

    return cleaned[:160].strip()


def _sanitize_query(candidate: str, original_input: str) -> str:
    candidate = " ".join((candidate or "").split()).strip()
    original_input = " ".join((original_input or "").split()).strip()

    if not candidate:
        return original_input

    if len(candidate) > 220:
        shortened = _last_question_like_span(candidate)
        return shortened or original_input

    return candidate


def parse_rewrite_output(payload: dict[str, Any]) -> dict[str, str]:
    raw = str(payload.get("raw", "") or "").strip()
    last_input = str(payload.get("last_input", "") or "").strip()

    cleaned = _strip_code_fences(raw)
    json_candidate = _extract_json_object(cleaned) or cleaned

    rewrite_class = "standalone"
    rewritten_query = last_input

    try:
        parsed = json.loads(json_candidate)

        if isinstance(parsed, dict):
            rewrite_class = str(parsed.get("class", "standalone")).strip() or "standalone"
            reworded = str(parsed.get("reworded version", "")).strip()
            rewritten_query = _sanitize_query(reworded, last_input)
        else:
            rewritten_query = _sanitize_query(cleaned, last_input)

    except Exception as e:
        rewritten_query = _sanitize_query(cleaned, last_input)
        print(f"[WARN] Failed to parse rewrite JSON ({e}); using safe fallback.")

    print(f"[Rewrite Result] class={rewrite_class}, query={rewritten_query}")

    return {
        "rewritten_query": rewritten_query,
        "rewrite_class": rewrite_class,
        "rewrite_raw": raw,
    }


def build_rewrite_chain() -> Runnable:
    # response_format=json_object 로 OpenAI JSON mode 강제 → 항상 valid JSON 반환
    llm = ChatOpenAI(
        model=REWRITE_MODEL,
        temperature=REWRITE_TEMPERATURE,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return (
        {
            "raw": prompt | llm | StrOutputParser(),
            "last_input": lambda x: x["input"],
        }
        | RunnableLambda(parse_rewrite_output)
    )