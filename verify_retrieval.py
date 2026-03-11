import os
from typing import Any

from sentence_transformers import SentenceTransformer
from supabase import create_client
from rag_pipeline import retrieve, FALLBACK_ANSWER


def test_retrieval(
    model: SentenceTransformer, supabase: Any, query: str, top_k: int = 3
) -> list[dict[str, Any]]:
    _ = model
    _ = supabase
    rows = retrieve(query, top_k=top_k)

    print(f"\nQuery: {query}")
    print(f"Results: {len(rows)}")
    for i, doc in enumerate(rows, start=1):
        sim = doc.get("similarity", 0.0)
        content = (doc.get("content") or "").replace("\n", " ")
        print(f"  #{i} [similarity: {sim:.4f}]")
        print(f"  Title: {doc.get('title')}")
        print(f"  Source: {doc.get('source')}")
        print(f"  Content: {content[:200]}...")
    return rows


def main() -> None:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing env vars. Need SUPABASE_URL and SUPABASE_SERVICE_KEY.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    supabase = create_client(supabase_url, supabase_key)

    test_queries = [
        "How do I unblock someone in chat?",
        "What is the difference between wounded conversion and battlefield treatment?",
        "How should I protect my resources from enemy attacks?",
        "What is the best troop composition for PvP?",
        "How does the resonance boost for lord equipment work?",
        "How do I get more glory banners?",
    ]

    all_scores = []
    zero_result = 0
    print("=" * 60)
    for q in test_queries:
        rows = test_retrieval(model, supabase, q, top_k=3)
        all_scores.extend([r.get("similarity", 0.0) for r in rows])
        if not rows:
            zero_result += 1
        print("=" * 60)

    count_res = supabase.table("gog_documents").select("id", count="exact").limit(1).execute()
    count = count_res.count if hasattr(count_res, "count") else None
    print("\n=== VERIFICATION SUMMARY ===")
    if count is not None:
        print(f"Total rows in gog_documents: {count}")
    if all_scores:
        print(f"Average similarity: {sum(all_scores)/len(all_scores):.4f}")
    else:
        print(FALLBACK_ANSWER)
    print(f"Queries with no retrieval: {zero_result}")


if __name__ == "__main__":
    main()
