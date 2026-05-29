"""Stratified train/val/test split from unified DuckDB table."""

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

DB_PATH = "data/unified.duckdb"
RANDOM_STATE = 42


def build_splits(db_path: str = DB_PATH) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(db_path)
    df = con.execute("SELECT * FROM unified").fetchdf()

    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["attack_category"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp["attack_category"],
    )

    for name, split_df in [
        ("split_train", train),
        ("split_val", val),
        ("split_test", test),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM split_df")

    con.close()

    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        dist = split_df["attack_category"].value_counts()
        print(f"\n{name} category counts:\n{dist}")

    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    build_splits()
