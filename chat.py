from rag_pipeline import ask


def main() -> None:
    print("=" * 60)
    print("  Guns of Glory Customer Service Bot")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\nPlayer: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        result = ask(query)
        print(f"\nBot: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                sim = s.get("similarity", 0.0) or 0.0
                print(f"  - {s.get('title')} (relevance: {sim:.2f})")


if __name__ == "__main__":
    main()

