"""Near-duplicate removal using MinHash LSH."""

import re
import duckdb
import pandas as pd
from datasketch import MinHash, MinHashLSH

DB_PATH = "data/unified.duckdb"
SIMILARITY_THRESHOLD = 0.95
NUM_PERM = 128


def text_to_shingles(text: str, k: int = 3) -> set:
    text = re.sub(r"\s+", " ", text.lower().strip())
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def make_minhash(shingles: set) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles:
        m.update(s.encode("utf8"))
    return m


def find_duplicates(df: pd.DataFrame) -> set:
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}

    for _, row in df.iterrows():
        shingles = text_to_shingles(row["input_text"])
        if len(shingles) < 3:
            continue
        m = make_minhash(shingles)
        minhashes[row["id"]] = m

    duplicate_ids = set()
    seen_ids = set()

    for id_, m in minhashes.items():
        if id_ in duplicate_ids:
            continue
        lsh.insert(id_, m)
        seen_ids.add(id_)
        result = lsh.query(m)
        for match_id in result:
            if match_id != id_ and match_id not in duplicate_ids:
                duplicate_ids.add(match_id)

    return duplicate_ids


def deduplicate(db_path: str = DB_PATH) -> None:
    con = duckdb.connect(db_path)
    df = con.execute("SELECT id, input_text FROM unified").fetchdf()
    print(f"Total rows before dedup: {len(df)}")

    duplicate_ids = find_duplicates(df)
    print(f"Duplicates found: {len(duplicate_ids)}")
    print(f"Rows after dedup: {len(df) - len(duplicate_ids)}")

    if duplicate_ids:
        ids_str = ", ".join(f"'{i}'" for i in duplicate_ids)
        con.execute(f"DELETE FROM unified WHERE id IN ({ids_str})")
        print("Duplicates removed from unified table.")

    con.close()


if __name__ == "__main__":
    deduplicate()
