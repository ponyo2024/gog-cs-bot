import json
import os
import time

import requests


DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable.")

ENDPOINT = "https://coding.dashscope.aliyuncs.com/v1/chat/completions"
SOURCE_FILES = ["gog_faq_data.json"]


def load_all_documents() -> tuple[str, int]:
    """Load all raw documents and format them as a single context string."""
    docs = []
    for filename in SOURCE_FILES:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                docs.extend(json.load(f))
        except FileNotFoundError:
            print(f"[WARN] {filename} not found")

    if not docs:
        raise RuntimeError("No documents found. Ensure gog_faq_data.json exists.")

    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[Document {i}: {doc['title']}]\n{doc['content']}")

    context = "\n\n---\n\n".join(parts)
    print(f"Loaded {len(docs)} documents, ~{len(context) // 4} tokens")
    return context, len(docs)


ALL_CONTEXT, DOC_COUNT = load_all_documents()

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


def ask(query: str) -> dict:
    """Full long-context pipeline: send all docs and the query in one prompt."""
    user_message = f"""Here is the complete Guns of Glory knowledge base:

{ALL_CONTEXT}

---

Player question: {query}

Please answer the player's question based on the documents above."""

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    models = ["qwen3.5-plus", "qwen3-max-2026-01-23"]

    for model_name in models:
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
            start_time = time.time()
            resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
            latency = time.time() - start_time

            if resp.status_code == 200:
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "answer": answer,
                    "model_used": model_name,
                    "status": "success",
                    "latency_seconds": round(latency, 2),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "doc_count": DOC_COUNT,
                }

            print(f"  Model {model_name} failed: {resp.status_code} - {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"  Model {model_name} timed out (>120s)")
        except Exception as exc:
            print(f"  Model {model_name} error: {exc}")

    return {
        "answer": "Sorry, could not generate an answer.",
        "model_used": None,
        "status": "error",
        "latency_seconds": None,
        "doc_count": DOC_COUNT,
    }
