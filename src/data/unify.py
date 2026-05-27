"""Unify all dataset sources into a single DuckDB table."""

import os
import re
import duckdb
import pandas as pd

from src.data.loaders import (
    load_deepset,
    load_jasperai,
    load_lakera,
    load_safeguard,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS unified (
    id               VARCHAR PRIMARY KEY,
    input_text       VARCHAR NOT NULL,
    label            INTEGER NOT NULL,
    attack_category  VARCHAR NOT NULL,
    source           VARCHAR NOT NULL,
    split            VARCHAR NOT NULL
)
"""

LAKERA_THRESHOLD = 0.5

CATEGORY_PATTERNS = {
    "system_prompt_leak": [
        r"repeat.*system",
        r"print.*prompt",
        r"reveal.*instruction",
        r"what.*your.*prompt",
        r"show.*system",
        r"output.*above",
        r"ignore.*above.*say",
    ],
    "jailbreak": [
        r"dan\b",
        r"do anything now",
        r"jailbreak",
        r"pretend.*no restriction",
        r"act as if.*no limit",
        r"developer mode",
        r"evil.*mode",
        r"unrestricted",
    ],
    "role_play": [
        r"pretend you are",
        r"act as",
        r"roleplay",
        r"you are now",
        r"imagine you are",
        r"play the role",
        r"simulate",
    ],
    "indirect": [
        r"translate.*ignore",
        r"summarize.*instead",
        r"the document says.*ignore",
        r"hidden instruction",
        r"embedded.*command",
    ],
    "direct": [
        r"ignore.*previous.*instruction",
        r"ignore.*above.*instruction",
        r"disregard.*instruction",
        r"forget.*instruction",
        r"new instruction",
        r"override",
    ],
}


def classify_attack_category(text: str) -> str:
    t = text.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, t):
                return category
    return "unknown"


def _make_id(source: str, split: str, idx: int) -> str:
    return f"{source}__{split}__{idx}"


def _add_category(df: pd.DataFrame) -> pd.Series:
    return df["input_text"].apply(classify_attack_category)


def _map_deepset(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["input_text"] = df["text"]
    out["label"] = df["label"].astype(int)
    out["source"] = "deepset"
    out["split"] = split
    out["id"] = [_make_id("deepset", split, i) for i in range(len(df))]
    out["attack_category"] = _add_category(out)
    return out[["id", "input_text", "label", "attack_category", "source", "split"]]


def _map_jasperai(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["input_text"] = df["text"]
    out["label"] = df["label"].astype(int)
    out["source"] = "jasperai"
    out["split"] = split
    out["id"] = [_make_id("jasperai", split, i) for i in range(len(df))]
    out["attack_category"] = _add_category(out)
    return out[["id", "input_text", "label", "attack_category", "source", "split"]]


def _map_lakera(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["input_text"] = df["text"]
    out["label"] = (df["similarity"] >= LAKERA_THRESHOLD).astype(int)
    out["source"] = "lakera"
    out["split"] = split
    out["id"] = [_make_id("lakera", split, i) for i in range(len(df))]
    out["attack_category"] = _add_category(out)
    return out[["id", "input_text", "label", "attack_category", "source", "split"]]


def _map_safeguard(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["input_text"] = df["text"]
    out["label"] = df["label"].astype(int)
    out["source"] = "safeguard"
    out["split"] = split
    out["id"] = [_make_id("safeguard", split, i) for i in range(len(df))]
    out["attack_category"] = _add_category(out)
    return out[["id", "input_text", "label", "attack_category", "source", "split"]]


def build_unified_db(db_path: str = "data/unified.duckdb") -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS unified")
    con.execute(SCHEMA)

    sources = [
        ("deepset", load_deepset, _map_deepset),
        ("jasperai", load_jasperai, _map_jasperai),
        ("lakera", load_lakera, _map_lakera),
        ("safeguard", load_safeguard, _map_safeguard),
    ]

    for name, loader, mapper in sources:
        dataset = loader()
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            mapped = mapper(df, split)
            con.execute("INSERT INTO unified SELECT * FROM mapped")
            print(f"  {name}/{split}: {len(mapped)} rows inserted")

    total = con.execute("SELECT COUNT(*) FROM unified").fetchone()[0]
    print(f"\nTotal rows in unified: {total}")
    return con


if __name__ == "__main__":
    con = build_unified_db()
    print("\nCategory distribution:")
    print(
        con.execute(
            "SELECT attack_category, label, COUNT(*) as n "
            "FROM unified GROUP BY attack_category, label "
            "ORDER BY attack_category, label"
        ).fetchdf()
    )
