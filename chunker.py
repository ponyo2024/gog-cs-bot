import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHARS_PER_TOKEN = 4
MAX_CHUNK_CHARS = CHUNK_SIZE * CHARS_PER_TOKEN
OVERLAP_CHARS = CHUNK_OVERLAP * CHARS_PER_TOKEN
MIN_CHUNK_CHARS = 100
MIN_FAQ_CHARS = 30
MAX_CHUNKS_PER_TITLE = 12

INPUT_FILES = ["gog_faq_data.json", "gog_aihelp_data.json"]
OUTPUT_FILE = "gog_chunks.json"

BOILERPLATE_PATTERNS = [
    r"Play Guns of Glory on Bluestacks.*",
    r"Download Goodnight Bots.*",
    r"Use GoGBot.*",
    r"Register.*Login.*",
    r"Toggle navigation.*",
    r"gamesguideinfo\.com.*?guns-of-glory",
    r"Tier 13 troops added.*?More News",
]


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_title_only_text(content: str, title: str) -> bool:
    norm_content = normalize_text(content)
    norm_title = normalize_text(title.replace("Guns of Glory", "").replace("-", ""))
    if not norm_content or not norm_title:
        return False

    remaining = content
    title_variants = [
        title,
        title.replace("Guns of Glory - ", ""),
        "Guns of Glory - " + title,
    ]
    for prefix in title_variants:
        if prefix and remaining.lower().startswith(prefix.lower()):
            remaining = remaining[len(prefix) :].strip()
            break

    remaining_clean = remaining.strip(" \n\t-:?")
    return len(remaining_clean) < 50


def split_oversized_paragraph(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_CHUNK_CHARS:
        return [paragraph]
    parts = []
    step = max(1, MAX_CHUNK_CHARS - OVERLAP_CHARS)
    start = 0
    while start < len(paragraph):
        end = min(start + MAX_CHUNK_CHARS, len(paragraph))
        parts.append(paragraph[start:end].strip())
        if end == len(paragraph):
            break
        start += step
    return [p for p in parts if p]


def chunk_by_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        for part in split_oversized_paragraph(para):
            if len(current) + len(part) + 2 > MAX_CHUNK_CHARS and current:
                chunks.append(current.strip())
                overlap = current[-OVERLAP_CHARS:] if OVERLAP_CHARS > 0 else ""
                current = (overlap + "\n\n" + part).strip() if overlap else part
            else:
                current = (current + "\n\n" + part).strip()
    if current:
        chunks.append(current.strip())
    return chunks


def make_chunk_id(source: str, title: str, chunk_index: int) -> str:
    raw = f"{source}:{title}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def content_signature(title: str, content: str) -> str:
    window = normalize_text(content[:500])
    raw = f"{normalize_text(title)}::{window}"
    return hashlib.md5(raw.encode()).hexdigest()


def token_set(content: str) -> set[str]:
    return {t for t in normalize_text(content).split() if len(t) > 2}


def overlap_ratio(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def process_document(doc: dict) -> list[dict]:
    content = clean_text(doc["content"])
    min_chars = MIN_FAQ_CHARS if doc["type"] in ("faq", "official_faq") else MIN_CHUNK_CHARS
    if len(content) < min_chars or is_title_only_text(content, doc["title"]):
        return []

    if doc["type"] in ("faq", "official_faq") and len(content) <= MAX_CHUNK_CHARS:
        return [
            {
                "chunk_id": make_chunk_id(doc["source"], doc["title"], 0),
                "source": doc["source"],
                "type": doc["type"],
                "title": doc["title"],
                "url": doc["url"],
                "content": content,
                "chunk_index": 0,
                "total_chunks": 1,
                "char_count": len(content),
                "est_tokens": len(content) // CHARS_PER_TOKEN,
            }
        ]

    raw_chunks = chunk_by_paragraphs(content)
    kept: list[str] = []
    previous = ""
    seen_signatures: set[str] = set()
    for chunk in raw_chunks:
        if len(chunk) < MIN_CHUNK_CHARS or is_title_only_text(chunk, doc["title"]):
            continue
        signature = content_signature(doc["title"], chunk)
        if signature in seen_signatures:
            continue
        if previous and overlap_ratio(previous, chunk) > 0.92:
            continue
        seen_signatures.add(signature)
        kept.append(chunk)
        previous = chunk
        if len(kept) >= MAX_CHUNKS_PER_TITLE:
            break

    result = []
    for i, chunk_text in enumerate(kept):
        result.append(
            {
                "chunk_id": make_chunk_id(doc["source"], doc["title"], i),
                "source": doc["source"],
                "type": doc["type"],
                "title": doc["title"],
                "url": doc["url"],
                "content": chunk_text,
                "chunk_index": i,
                "total_chunks": len(kept),
                "char_count": len(chunk_text),
                "est_tokens": len(chunk_text) // CHARS_PER_TOKEN,
            }
        )
    return result


def dedupe_across_documents(chunks: list[dict]) -> list[dict]:
    per_title: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk in chunks:
        per_title[(chunk["source"], chunk["title"])].append(chunk)

    deduped: list[dict] = []
    for _, title_chunks in per_title.items():
        accepted: list[dict] = []
        seen_signatures: set[str] = set()
        for chunk in title_chunks:
            signature = content_signature(chunk["title"], chunk["content"])
            if signature in seen_signatures:
                continue
            if accepted and overlap_ratio(accepted[-1]["content"], chunk["content"]) > 0.92:
                continue
            seen_signatures.add(signature)
            accepted.append(chunk)
            if len(accepted) >= MAX_CHUNKS_PER_TITLE:
                break

        for index, chunk in enumerate(accepted):
            chunk["chunk_index"] = index
            chunk["total_chunks"] = len(accepted)
            chunk["chunk_id"] = make_chunk_id(chunk["source"], chunk["title"], index)
            deduped.append(chunk)

    return sorted(deduped, key=lambda c: (c["source"], c["title"], c["chunk_index"]))


def main() -> None:
    documents = []
    for filename in INPUT_FILES:
        path = Path(filename)
        if not path.exists():
            print(f"[WARN] File not found: {filename} (skipping)")
            continue
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        documents.extend(docs)
        print(f"Loaded {len(docs)} documents from {filename}")

    if not documents:
        raise RuntimeError("No input documents found. Run the scrapers first.")

    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(process_document(doc))

    all_chunks = dedupe_across_documents(all_chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    by_type = defaultdict(int)
    by_title = defaultdict(int)
    for chunk in all_chunks:
        by_type[chunk["type"]] += 1
        by_title[chunk["title"]] += 1

    total_tokens = sum(c["est_tokens"] for c in all_chunks)
    avg_tokens = (total_tokens // len(all_chunks)) if all_chunks else 0
    print("\n=== CHUNKING COMPLETE ===")
    print(f"Input documents: {len(documents)}")
    print(f"Output chunks: {len(all_chunks)}")
    for chunk_type, count in sorted(by_type.items()):
        print(f"  {chunk_type}: {count} chunks")
    print(f"Total estimated tokens: ~{total_tokens:,}")
    print(f"Average tokens per chunk: ~{avg_tokens}")
    print("Most repeated titles:")
    for title, count in sorted(by_title.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {count:>2}  {title}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
