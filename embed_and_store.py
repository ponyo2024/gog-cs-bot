import json
import os
from typing import Any

from sentence_transformers import SentenceTransformer
from supabase import create_client


def load_chunks(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_existing_ids(supabase: Any) -> set[str]:
    existing_ids: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        res = (
            supabase.table("gog_documents")
            .select("id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            break
        existing_ids.update(r["id"] for r in rows if "id" in r)
        if len(rows) < page_size:
            break
        offset += page_size
    return existing_ids


def delete_stale_ids(supabase: Any, stale_ids: list[str]) -> None:
    batch_size = 100
    for i in range(0, len(stale_ids), batch_size):
        batch = stale_ids[i : i + batch_size]
        supabase.table("gog_documents").delete().in_("id", batch).execute()


def build_rows(chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> list[dict[str, Any]]:
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append(
            {
                "id": chunk["chunk_id"],
                "source": chunk["source"],
                "type": chunk["type"],
                "title": chunk["title"],
                "url": chunk.get("url"),
                "content": chunk["content"],
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "embedding": emb,
            }
        )
    return rows


def main() -> None:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing env vars. Need SUPABASE_URL and SUPABASE_SERVICE_KEY.")

    print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")

    supabase = create_client(supabase_url, supabase_key)
    chunks = load_chunks("gog_chunks.json")
    print(f"Loaded {len(chunks)} chunks")

    existing_ids = fetch_existing_ids(supabase)
    print(f"Existing rows in table: {len(existing_ids)}")
    current_ids = {c["chunk_id"] for c in chunks}
    stale_ids = sorted(existing_ids - current_ids)
    if stale_ids:
        print(f"Deleting stale rows: {len(stale_ids)}")
        delete_stale_ids(supabase, stale_ids)
        existing_ids -= set(stale_ids)

    pending = chunks
    print(f"Chunks to upsert this run: {len(pending)}")
    if not pending:
        print("No chunks to process.")
        return

    batch_size = 50
    stored = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        texts = [c["content"] for c in batch]
        embeddings = model.encode(texts).tolist()

        if i == 0 and embeddings:
            dim = len(embeddings[0])
            print(f"Embedding dimension: {dim}")
            if dim != 384:
                raise RuntimeError(f"Expected embedding dim 384, got {dim}")

        rows = build_rows(batch, embeddings)
        try:
            supabase.table("gog_documents").upsert(rows).execute()
            stored += len(rows)
        except Exception as e:
            print(f"ERROR on batch {i}-{i + batch_size}: {e}")
            for row in rows:
                try:
                    supabase.table("gog_documents").upsert(row).execute()
                    stored += 1
                except Exception as inner:
                    print(f"PERMANENT FAIL: {row['id']} - {inner}")

        print(f"Progress: {stored}/{len(chunks)}")

    final = supabase.table("gog_documents").select("id", count="exact").limit(1).execute()
    final_count = final.count if hasattr(final, "count") else None
    print("\n=== EMBEDDING COMPLETE ===")
    print(f"Stored in this run: {stored}")
    if final_count is not None:
        print(f"Final row count: {final_count}")


if __name__ == "__main__":
    main()
