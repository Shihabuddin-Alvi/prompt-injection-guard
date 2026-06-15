"""Deduplicate synthetic examples against existing training data via embeddings."""

import json
from pathlib import Path

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

SYNTHETIC_PATH = Path("data/synthetic_raw.jsonl")
OUTPUT_PATH = Path("data/synthetic_deduped.jsonl")
DB_PATH = "data/unified.duckdb"
SIMILARITY_THRESHOLD = 0.92
BATCH_SIZE = 64


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def run():
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("Loading synthetic examples...")
    with open(SYNTHETIC_PATH) as f:
        synthetic = [json.loads(line) for line in f]
    print(f"  {len(synthetic)} synthetic examples")

    print("Loading training data...")
    con = duckdb.connect(DB_PATH)
    train_texts = [
        r[0] for r in con.execute("SELECT input_text FROM split_train").fetchall()
    ]
    con.close()
    print(f"  {len(train_texts)} training examples")

    print("Embedding training data...")
    train_embeddings = model.encode(
        train_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Embedding synthetic data...")
    synthetic_texts = [ex["input_text"] for ex in synthetic]
    synthetic_embeddings = model.encode(
        synthetic_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Computing similarities...")
    kept = []
    dropped = 0
    chunk_size = 100
    for i in range(0, len(synthetic), chunk_size):
        chunk_emb = synthetic_embeddings[i : i + chunk_size]
        sims = cosine_similarity_matrix(chunk_emb, train_embeddings)
        max_sims = sims.max(axis=1)
        for j, max_sim in enumerate(max_sims):
            ex = synthetic[i + j]
            if max_sim > SIMILARITY_THRESHOLD:
                dropped += 1
            else:
                ex["max_train_similarity"] = float(max_sim)
                kept.append(ex)

    print(f"\nKept: {len(kept)}")
    print(f"Dropped (similarity > {SIMILARITY_THRESHOLD}): {dropped}")
    print(f"Drop rate: {dropped / len(synthetic):.3f}")

    with open(OUTPUT_PATH, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
