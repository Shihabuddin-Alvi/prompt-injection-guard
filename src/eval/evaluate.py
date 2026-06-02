import argparse
import json

import duckdb
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

DB_PATH = "data/unified.duckdb"
MAX_LENGTH = 256


def load_test_set(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path)
    df = con.execute(
        "SELECT input_text, label, attack_category FROM split_test"
    ).fetchdf()
    con.close()
    df["label"] = df["label"].astype(int)
    return df


def predict(df: pd.DataFrame, model_path: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    texts = df["input_text"].tolist()
    all_preds = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            preds = torch.argmax(outputs.logits, dim=1).numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


def evaluate(model_path: str, output_path: str = "docs/eval_results.json"):
    print(f"Loading test set from {DB_PATH}")
    df = load_test_set(DB_PATH)
    print(f"Test set size: {len(df)}")

    print(f"Running inference with model from {model_path}")
    preds = predict(df, model_path)
    labels = df["label"].values

    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"\nMacro F1: {macro_f1:.4f}")

    print("\nPer-category recall:")
    for cat in df["attack_category"].unique():
        mask = df["attack_category"] == cat
        cat_labels = labels[mask]
        cat_preds = preds[mask]
        cat_f1 = f1_score(cat_labels, cat_preds, average="macro", zero_division=0)
        print(f"  {cat}: F1={cat_f1:.4f}  n={mask.sum()}")

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=["benign", "injection"]))

    print("\nConfusion matrix:")
    print(confusion_matrix(labels, preds))

    results = {
        "macro_f1": macro_f1,
        "n_test": len(df),
        "model_path": model_path,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n--- GATE RESULT ---")
    if macro_f1 >= 0.85:
        print(f"PASSED: macro F1 {macro_f1:.4f} >= 0.85")
    else:
        print(f"FAILED: macro F1 {macro_f1:.4f} < 0.85")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="docs/eval_results.json")
    args = parser.parse_args()
    evaluate(model_path=args.model_path, output_path=args.output)
