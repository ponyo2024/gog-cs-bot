import ast
import json
import sys
import time

from long_context_pipeline import ask


def load_test_queries() -> list[str]:
    with open("test_rag.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="test_rag.py")

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TEST_CASES":
                    cases = ast.literal_eval(node.value)
                    return [case["query"] for case in cases]
    raise RuntimeError("Could not load TEST_CASES from test_rag.py")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results = []
    test_queries = load_test_queries()

    for i, query in enumerate(test_queries, start=1):
        print(f"\n{'=' * 60}")
        print(f"Q{i}: {query}")

        result = ask(query)
        result["query"] = query
        results.append(result)

        print(f"Status: {result['status']}")
        print(f"Model: {result.get('model_used', 'N/A')}")
        print(f"Latency: {result.get('latency_seconds', 'N/A')}s")
        if result.get("total_tokens"):
            print(f"Tokens used: {result['total_tokens']}")
        print(f"Answer: {result['answer'][:300]}...")

        time.sleep(2)

    with open("long_context_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("LONG CONTEXT SUMMARY")
    print("=" * 60)
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    latencies = [r["latency_seconds"] for r in results if r.get("latency_seconds")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    idk_count = sum(
        1
        for r in results
        if r["status"] == "success"
        and ("don't have" in r["answer"].lower() or "i don't know" in r["answer"].lower())
    )

    print(f"Total queries: {len(results)}")
    print(f"Success: {success}")
    print(f"Errors: {errors}")
    print(f"'I don't know' answers: {idk_count}")
    print(f"Average latency: {avg_latency:.2f}s")

    token_counts = [r["total_tokens"] for r in results if r.get("total_tokens")]
    if token_counts:
        print(f"Average tokens per query: {sum(token_counts) // len(token_counts)}")
        print(f"Total tokens consumed: {sum(token_counts)}")

    print("\nResults saved to: long_context_test_results.json")


if __name__ == "__main__":
    main()
