import json
from statistics import mean

from rag_pipeline import FALLBACK_ANSWER, ask, retrieve


TEST_CASES = [
    {"query": "How do I unblock someone in chat?", "expected": ["unblock"]},
    {"query": "How many Shooting Gallery coins do I need in total?", "expected": ["shooting gallery"]},
    {"query": "How do I get Glory Banners?", "expected": ["glory banners"]},
    {"query": "What is the difference between wounded conversion and battlefield treatment?", "expected": ["battlefield treatment", "wounded conversion"]},
    {"query": "Should I upgrade my castle first or level up all buildings evenly?", "expected": ["all buildings", "castle"]},
    {"query": "How does the resonance boost for lord equipment work?", "expected": ["resonance", "lord equipment"]},
    {"query": "How should I protect myself from enemy rally attacks?", "expected": ["protecting yourself", "rally"]},
    {"query": "What is the best troop composition for PvP?", "expected": ["troop composition", "pvp"]},
    {"query": "How do I grow faster as a mid-level player?", "expected": ["growth strategy", "march slots"]},
    {"query": "When is the next Guns of Glory update coming?", "expected": []},
    {"query": "How do I unblock a player in chat?", "expected": ["unblock"]},
    {"query": "How does resonance work for lord equipment?", "expected": ["resonance", "lord equipment"]},
    {"query": "What is the best PvP troop composition?", "expected": ["troop composition", "pvp"]},
    {"query": "How do I get basic development ingot?", "expected": ["basic development ingot"]},
    {"query": "How many total shooting gallery coins are needed?", "expected": ["shooting gallery"]},
]


def relevance_flags(retrieved: list[dict], expected_keywords: list[str]) -> tuple[bool, bool]:
    if not retrieved or not expected_keywords:
        return (False, False) if retrieved else (False, False)
    titles = [r.get("title", "").lower() for r in retrieved]
    top1 = any(keyword in titles[0] for keyword in expected_keywords)
    top3 = any(any(keyword in title for keyword in expected_keywords) for title in titles[:3])
    return top1, top3


def main() -> None:
    results = []
    top1_scores = []
    top1_hits = 0
    top3_hits = 0
    citation_count = 0
    fallback_count = 0

    for case in TEST_CASES:
        query = case["query"]
        retrieved = retrieve(query, top_k=3)
        result = ask(query)
        top1_relevant, top3_relevant = relevance_flags(retrieved, case["expected"])
        top1_hits += int(top1_relevant)
        top3_hits += int(top3_relevant)
        citation_present = "[Source" in result["answer"]
        citation_count += int(citation_present)
        fallback_used = result["answer"] == FALLBACK_ANSWER
        fallback_count += int(fallback_used)
        if retrieved:
            top1_scores.append(float(retrieved[0].get("similarity", 0.0)))

        record = {
            "query": query,
            "expected_keywords": case["expected"],
            "status": result["status"],
            "model_used": result.get("model_used"),
            "answer": result["answer"],
            "citation_present": citation_present,
            "fallback_used": fallback_used,
            "top_1_relevant": top1_relevant,
            "correct_source_in_top3": top3_relevant,
            "retrieval_titles": [r.get("title") for r in retrieved],
            "retrieval_scores": [r.get("similarity") for r in retrieved],
            "sources": result.get("sources", []),
        }
        results.append(record)
        print()

    with open("rag_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success = sum(1 for r in results if r["status"] == "success")
    no_retrieval = sum(1 for r in results if r["status"] == "no_retrieval")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"Total queries: {len(results)}")
    print(f"  Success: {success}")
    print(f"  No retrieval: {no_retrieval}")
    print(f"  Errors: {errors}")
    print(f"  Top-1 similarity average: {mean(top1_scores):.4f}" if top1_scores else "  Top-1 similarity average: n/a")
    print(f"  Top-1 relevant hits: {top1_hits}/{len(results)}")
    print(f"  Correct source in top-3: {top3_hits}/{len(results)}")
    print(f"  Fallback count: {fallback_count}")
    print(f"  Cited answer count: {citation_count}")

    used_models = sorted({r["model_used"] for r in results if r.get("model_used")})
    if used_models:
        print(f"  Models used: {', '.join(used_models)}")

    print("\nResults saved to: rag_test_results.json")


if __name__ == "__main__":
    main()
