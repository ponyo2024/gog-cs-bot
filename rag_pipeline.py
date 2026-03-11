import hashlib
import os
import re
from collections import defaultdict
from typing import Any

import requests
from sentence_transformers import SentenceTransformer
from supabase import create_client


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variable.")

FALLBACK_ANSWER = (
    "I don't have specific information about that in my knowledge base. "
    "You may want to check the official Guns of Glory help center or ask in your alliance chat."
)
INITIAL_RETRIEVAL_K = 6
FINAL_CONTEXT_K = 3
RETRIEVAL_THRESHOLD = 0.35
FALLBACK_RETRIEVAL_THRESHOLD = 0.20
PROMPT_CHARS_PER_SOURCE = 1200

model = SentenceTransformer("all-MiniLM-L6-v2")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


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


def doc_signature(doc: dict[str, Any]) -> str:
    raw = f"{doc.get('title','')}::{normalize_text((doc.get('content') or '')[:300])}"
    return hashlib.md5(raw.encode()).hexdigest()


def rerank_score(query: str, doc: dict[str, Any]) -> float:
    score = float(doc.get("similarity", 0.0))
    query_terms = keyword_tokens(query)
    title_terms = keyword_tokens(doc.get("title", ""))
    content_terms = keyword_tokens((doc.get("content") or "")[:400])
    title_overlap = len(query_terms & title_terms)
    content_overlap = len(query_terms & content_terms)
    if doc.get("type") == "faq":
        score += 0.08
    score += min(0.18, title_overlap * 0.06)
    score += min(0.10, content_overlap * 0.02)
    return score


def retrieve_candidates(
    query: str,
    top_k: int = INITIAL_RETRIEVAL_K,
    threshold: float = RETRIEVAL_THRESHOLD,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(query).tolist()
    result = supabase.rpc(
        "match_gog_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": top_k,
        },
    ).execute()
    return result.data or []


def keyword_overlap_count(query: str, doc: dict[str, Any]) -> int:
    query_terms = keyword_tokens(query)
    doc_terms = keyword_tokens(f"{doc.get('title', '')} {(doc.get('content') or '')[:500]}")
    return len(query_terms & doc_terms)


def merge_unique_docs(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(primary)
    seen = {doc_signature(d) for d in primary}
    for doc in secondary:
        sig = doc_signature(doc)
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(doc)
    return merged


def rerank_and_filter(query: str, docs: list[dict[str, Any]], final_k: int = FINAL_CONTEXT_K) -> list[dict[str, Any]]:
    deduped = []
    seen_signatures = set()
    per_title_counts: dict[str, int] = defaultdict(int)

    ranked = sorted(docs, key=lambda d: rerank_score(query, d), reverse=True)
    for doc in ranked:
        signature = doc_signature(doc)
        title = doc.get("title", "")
        if signature in seen_signatures:
            continue
        if per_title_counts[title] >= 2 and len(deduped) >= final_k:
            continue
        doc["rerank_score"] = rerank_score(query, doc)
        seen_signatures.add(signature)
        per_title_counts[title] += 1
        deduped.append(doc)
        if len(deduped) >= final_k:
            break
    return deduped


def should_return_no_retrieval(query: str, docs: list[dict[str, Any]]) -> bool:
    if not docs:
        return True
    best_score = float(docs[0].get("similarity", 0.0))
    best_keyword_overlap = max(keyword_overlap_count(query, d) for d in docs)
    unique_titles = {d.get("title") for d in docs}
    all_guides = all(d.get("type") == "guide" for d in docs)
    if best_score < FALLBACK_RETRIEVAL_THRESHOLD and best_keyword_overlap == 0:
        return True
    if len(unique_titles) == 1 and all_guides and best_score < RETRIEVAL_THRESHOLD + 0.10 and best_keyword_overlap == 0:
        return True
    return False


def retrieve(query: str, top_k: int = FINAL_CONTEXT_K, threshold: float = RETRIEVAL_THRESHOLD) -> list[dict[str, Any]]:
    if is_future_update_query(query):
        return []
    candidate_k = max(INITIAL_RETRIEVAL_K, top_k + 2)
    raw_docs = retrieve_candidates(query, top_k=candidate_k, threshold=threshold)
    best_overlap = max((keyword_overlap_count(query, d) for d in raw_docs), default=0)
    if not raw_docs or best_overlap == 0:
        fallback_docs = retrieve_candidates(
            query,
            top_k=max(candidate_k, INITIAL_RETRIEVAL_K + 2),
            threshold=FALLBACK_RETRIEVAL_THRESHOLD,
        )
        raw_docs = merge_unique_docs(raw_docs, fallback_docs)
    reranked = rerank_and_filter(query, raw_docs, final_k=top_k)
    if should_return_no_retrieval(query, reranked):
        return []
    return reranked


def _build_context(context_docs: list[dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(context_docs, start=1):
        snippet = (doc.get("content") or "")[:PROMPT_CHARS_PER_SOURCE].strip()
        parts.append(
            f"[Source {i}: {doc.get('title', 'Untitled')} | similarity={doc.get('similarity', 0.0):.4f}]\n"
            f"{snippet}"
        )
    return "\n\n---\n\n".join(parts)


def _model_candidates() -> list[str]:
    return [
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "qwen3-coder-next",
        "qwen3-coder-plus",
        "glm-5",
        "kimi-k2.5",
    ]


def postprocess_answer(answer: str, source_count: int) -> str:
    if FALLBACK_ANSWER in answer:
        return FALLBACK_ANSWER
    if "[Source" not in answer and source_count > 0:
        answer = answer.rstrip() + "\n\n[Source 1]"
    return answer


def generate_answer(query: str, context_docs: list[dict[str, Any]]) -> dict[str, Any]:
    if not DASHSCOPE_API_KEY:
        return {"answer": "DASHSCOPE_API_KEY is missing. Cannot generate answer.", "model_used": None, "sources": [], "status": "error"}

    context_str = _build_context(context_docs)
    system_prompt = """You are a Guns of Glory customer service assistant. Your job is to help players with their questions about the game.

RULES:
1. Answer based on the provided context. Use the information available to give the best possible answer.
2. If the context directly answers the question, provide a clear and concise response.
3. If the context contains related or partial information, use it to give a helpful answer. Explain what you can based on available information.
4. Always cite which source you used, e.g. "[Source 1]" or "[Source 2]".
5. ONLY say "I don't have information about that" if the context is completely unrelated to the question. This should be rare.
6. Do NOT invent specific numbers, stats, or game mechanics that aren't mentioned in the context.
7. Keep your tone friendly and helpful, like a knowledgeable alliance member.
8. If the question is in Chinese, answer in Chinese. If in English, answer in English."""
    user_message = (
        "Context from the Guns of Glory knowledge base:\n\n"
        f"{context_str}\n\n---\n\n"
        f"Player question: {query}\n\n"
        "Answer using only the context above."
    )

    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
    endpoint = "https://coding.dashscope.aliyuncs.com/v1/chat/completions"
    last_error = None

    for model_name in _model_candidates():
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                answer = postprocess_answer(answer, len(context_docs))
                return {
                    "answer": answer,
                    "model_used": model_name,
                    "sources": [
                        {"title": d.get("title"), "url": d.get("url"), "similarity": d.get("similarity")}
                        for d in context_docs
                    ],
                    "status": "success",
                }
            last_error = f"{model_name}: {resp.status_code} - {resp.text[:300]}"
            print(f"  Model {model_name} failed: {resp.status_code}")
        except Exception as e:
            last_error = f"{model_name}: {e}"
            print(f"  Model {model_name} error: {e}")

    return {
        "answer": f"Sorry, I couldn't generate an answer. Error: {last_error}",
        "model_used": None,
        "sources": [],
        "status": "error",
    }


def ask(query: str, top_k: int = FINAL_CONTEXT_K) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(f"Question: {query}")
    docs = retrieve(query, top_k=top_k, threshold=RETRIEVAL_THRESHOLD)
    print(f"Retrieved {len(docs)} documents")
    for i, d in enumerate(docs, start=1):
        sim = d.get("similarity", 0.0)
        rerank = d.get("rerank_score", sim)
        print(f"  #{i} [sim={sim:.4f} rerank={rerank:.4f}] {d.get('title')}")

    if not docs:
        return {"answer": FALLBACK_ANSWER, "model_used": None, "sources": [], "status": "no_retrieval"}

    result = generate_answer(query, docs)
    print(f"\nAnswer ({result['status']}, model: {result.get('model_used')}):")
    print(result["answer"])
    return result
