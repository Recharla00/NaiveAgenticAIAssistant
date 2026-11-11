"""Step 3: local documents -> chunks -> embeddings -> persistent Chroma."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import warnings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

LESSON_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = LESSON_DIR / "knowledge_base"
DEFAULT_DATA = LESSON_DIR / "data"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"


def prepare_documents(input_path, chunk_size=600, chunk_overlap=100):
    """Read each source once and preserve PDF page numbers (one-based)."""
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Use chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
    source = Path(input_path).resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    root = source if source.is_dir() else source.parent
    files = sorted(path for path in (source.rglob("*") if source.is_dir() else [source])
                   if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".markdown"})
    if not files:
        raise ValueError("No PDF or Markdown documents found.")
    documents, sources = [], []
    for path in files:
        data = path.read_bytes()
        name = path.relative_to(root).as_posix()
        sources.append({"source": name, "sha256": sha256(data).hexdigest()})
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError(f"Encrypted PDFs are not supported: {name}")
            nonempty = 0
            for number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": name, "page": number}))
                    nonempty += 1
                else:
                    warnings.warn(f"No text on {name}, page {number}; skipped. Scans need OCR.")
            if not nonempty:
                raise ValueError(f"No extractable text in {name}. Scanned PDFs need OCR.")
        else:
            text = data.decode("utf-8-sig").strip()
            if not text:
                raise ValueError(f"Empty Markdown document: {name}")
            documents.append(Document(page_content=text, metadata={"source": name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              add_start_index=True)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, 1):
        identity = json.dumps({"text": chunk.page_content, "metadata": chunk.metadata}, sort_keys=True)
        chunk.metadata["chunk_id"] = sha256(identity.encode()).hexdigest()
        chunk.metadata["chunk_number"] = index
    if not chunks:
        raise ValueError("The input produced no chunks.")
    corpus = {"input_root": str(root), "sources": sources,
              "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    return chunks, corpus


def persist_index(chunks, corpus, embedding_function, model_name, model_digest, data_dir=DEFAULT_DATA, batch_size=8):
    """Activate a complete snapshot. Unchanged snapshots reuse their vectors.

Each corpus/model snapshot gets its own collection, so removed or edited
documents cannot leak old chunks into the active index. Older collections are
retained on disk. One ingestion process should write this data folder at a time.
"""
    from langchain_chroma import Chroma

    if not chunks or batch_size < 1:
        raise ValueError("Provide chunks and a positive batch size.")
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs detected.")
    identity = {"format_version": 1, "corpus": corpus, "chunk_ids": ids,
                "embedding_model": model_name, "embedding_digest": model_digest}
    fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    collection = "knowledge_" + fingerprint[:24]
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(collection_name=collection, persist_directory=str(data_dir / "chroma"),
                   embedding_function=embedding_function,
                   collection_metadata={"embedding_model": model_name, "embedding_digest": model_digest})
    existing = set(store.get(include=[])["ids"])
    missing = [chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing]
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        print(f"Embedding/storing chunks {start + 1}-{start + len(batch)} of {len(missing)} new chunks...", flush=True)
        store.add_documents(batch, ids=[chunk.metadata["chunk_id"] for chunk in batch])
    stored = store.get(include=[])["ids"]
    if set(stored) != set(ids):
        raise ValueError("Stored IDs do not match this snapshot; active index was not changed.")
    manifest = {**identity, "collection": collection, "count": len(ids),
                "persist_directory": str(data_dir / "chroma")}
    # Publish only after all vectors were stored. A failed run keeps the old index active.
    pending = data_dir / "active_index.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending.replace(data_dir / "active_index.json")
    return manifest, len(missing)
