import json
import os
from pathlib import Path
from typing import Any
import re

import requests


ENDPOINT = "https://coding.dashscope.aliyuncs.com/v1/chat/completions"
DATA_FILE = Path(__file__).with_name("gog_faq_data.json")
FALLBACK_ANSWER = (
    "I don't have specific information about that in my knowledge base. "
    "You may want to check the official Guns of Glory help center or ask in your alliance chat."
)


def _load_docs() -> tuple[str, int]:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[Document {i}: {doc['title']}]\n{doc['content']}")
    return "\n\n---\n\n".join(parts), len(docs)


ALL_CONTEXT, DOC_COUNT = _load_docs()

SYSTEM_PROMPT = """You are a Guns of Glory customer service assistant. Your job is to help players with their questions about the game.

You have been given the COMPLETE knowledge base of Guns of Glory FAQs and guides. Use this information to answer player questions.

RULES:
1. Answer based on the provided documents. Use the information available to give the best possible answer.
2. If the documents directly answer the question, provide a clear and concise response.
3. If the documents contain related or partial information, use it to give a helpful answer.
4. Always cite which document you used, e.g. "[Document 3]" or "[Document 15]".
5. ONLY say "I don't have information about that" if NONE of the documents are relevant to the question.
6. Do NOT invent specific numbers, stats, or game mechanics that aren't mentioned in the documents.
7. Keep your tone friendly and helpful, like a knowledgeable alliance member.
8. If the question is in Chinese, answer in Chinese. If in English, answer in English."""


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def keyword_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if len(t) > 2}


def is_future_update_query(query: str) -> bool:
    tokens = keyword_tokens(query)
    if "update" not in tokens:
        return False
    if "next" in tokens or "upcoming" in tokens:
        return True
    return "when" in tokens and ("coming" in tokens or "release" in tokens)


def ask_long_context(query: str) -> dict[str, Any]:
    if is_future_update_query(query):
        return {
            "answer": FALLBACK_ANSWER,
            "model_used": None,
            "sources": [],
            "status": "no_retrieval",
            "total_tokens": None,
            "doc_count": DOC_COUNT,
        }

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return {
            "answer": "DASHSCOPE_API_KEY is missing. Cannot generate answer.",
            "model_used": None,
            "sources": [],
            "status": "error",
            "total_tokens": None,
        }

    user_message = (
        "Here is the complete Guns of Glory knowledge base:\n\n"
        f"{ALL_CONTEXT}\n\n---\n\n"
        f"Player question: {query}\n\n"
        "Please answer the player's question based on the documents above."
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for model_name in ["qwen3.5-plus", "qwen3-max-2026-01-23"]:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                return {
                    "answer": data["choices"][0]["message"]["content"],
                    "model_used": model_name,
                    "sources": [],
                    "status": "success",
                    "total_tokens": usage.get("total_tokens"),
                    "doc_count": DOC_COUNT,
                }
        except Exception:
            continue

    return {
        "answer": "Sorry, could not generate an answer.",
        "model_used": None,
        "sources": [],
        "status": "error",
        "total_tokens": None,
        "doc_count": DOC_COUNT,
    }
