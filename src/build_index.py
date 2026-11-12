import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from build_chunks import load_corpus

# --- Configuration ---
MODEL_NAME = "intfloat/multilingual-e5-small"
STRATEGY = "v1"

CHUNKS_PATH = Path(f"data/chunks_{STRATEGY}.jsonl")
INDEX_DIR = Path("data/index")
EMBEDDINGS_PATH = INDEX_DIR / f"embeddings_{STRATEGY}.npy"
CHUNK_IDS_PATH = INDEX_DIR / f"chunk_ids_{STRATEGY}.json"


def main():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_corpus(CHUNKS_PATH)
    print(f"{len(chunks)} chunks loaded")

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # E5 models expect a "passage: " prefix on documents
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Row i of the matrix corresponds to chunk_ids[i]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CHUNK_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved chunk IDs to {CHUNK_IDS_PATH}")


if __name__ == "__main__":
    main()