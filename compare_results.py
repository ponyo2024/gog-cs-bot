import json


with open("rag_test_results.json", "r", encoding="utf-8") as f:
    rag_results = json.load(f)

with open("long_context_test_results.json", "r", encoding="utf-8") as f:
    lc_results = json.load(f)

print("=" * 80)
print("RAG vs LONG CONTEXT COMPARISON")
print("=" * 80)

for i, (rag, lc) in enumerate(zip(rag_results, lc_results), start=1):
    query = rag.get("query", lc.get("query", "N/A"))
    print(f"\nQ{i}: {query}")
    print(f"  RAG status: {rag['status']}")
    print(f"  LC  status: {lc['status']}")

    rag_answer = rag.get("answer", "")[:200]
    lc_answer = lc.get("answer", "")[:200]

    print(f"  RAG answer: {rag_answer}...")
    print(f"  LC  answer: {lc_answer}...")

    if lc.get("latency_seconds"):
        print(f"  LC latency: {lc['latency_seconds']}s")

print("\n" + "=" * 80)
print("AGGREGATE COMPARISON")
print("=" * 80)

rag_success = sum(1 for r in rag_results if r["status"] == "success")
lc_success = sum(1 for r in lc_results if r["status"] == "success")

rag_idk = sum(
    1
    for r in rag_results
    if "don't have" in r.get("answer", "").lower() or "i don't know" in r.get("answer", "").lower()
)
lc_idk = sum(
    1
    for r in lc_results
    if "don't have" in r.get("answer", "").lower() or "i don't know" in r.get("answer", "").lower()
)

print(f"{'Metric':<30} {'RAG':>10} {'Long Context':>15}")
print(f"{'-' * 55}")
print(f"{'Successful answers':<30} {rag_success:>10} {lc_success:>15}")
print(f"{'I dont know responses':<30} {rag_idk:>10} {lc_idk:>15}")

lc_latencies = [r["latency_seconds"] for r in lc_results if r.get("latency_seconds")]
if lc_latencies:
    print(f"{'Avg latency (s)':<30} {'N/A':>10} {sum(lc_latencies) / len(lc_latencies):>15.2f}")

lc_tokens = [r["total_tokens"] for r in lc_results if r.get("total_tokens")]
if lc_tokens:
    print(f"{'Avg tokens per query':<30} {'~1500':>10} {sum(lc_tokens) // len(lc_tokens):>15}")
    print(f"{'Total tokens (15 queries)':<30} {'~22500':>10} {sum(lc_tokens):>15}")

print("\nResults saved. Review both JSON files for detailed comparison.")
