"""
Phase 0 baseline evaluation.
Runs v1 and v2 on split_test and writes results to docs/baseline.md.
"""

from __future__ import annotations

import os
from datetime import datetime

import duckdb
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()
TOKEN = os.environ["HF_TOKEN"]
DB_PATH = "data/unified.duckdb"
BATCH_SIZE = 32
DEVICE = "cpu"

MODELS = {
    "v1": "alvi42/prompt-injection-guard-v1",
    "v2": "alvi42/prompt-injection-guard-v2",
}


def load_test_set(db_path: str) -> tuple[list[str], list[int]]:
    con = duckdb.connect(db_path)
    rows = con.execute(
        "SELECT input_text, label FROM split_test ORDER BY id"
    ).fetchall()
    texts = [r[0] for r in rows]
    labels = [int(r[1]) for r in rows]
    return texts, labels


def predict(
    texts: list[str],
    model_name: str,
    batch_size: int = BATCH_SIZE,
) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=TOKEN)
    model.eval()
    model.to(DEVICE)

    preds: list[int] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).tolist())
        if i % 256 == 0:
            print(f"  {i}/{len(texts)}")
    return preds


def format_report(name: str, labels: list[int], preds: list[int]) -> str:
    report = classification_report(
        labels, preds, target_names=["benign", "injection"], digits=4
    )
    cm = confusion_matrix(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    lines = [
        f"## {name}",
        "",
        f"Macro F1: {macro_f1:.4f}",
        "",
        "```",
        report,
        "```",
        "",
        "Confusion matrix (rows=actual, cols=predicted):",
        "",
        "```",
        "              pred_benign  pred_injection",
        f"actual_benign      {cm[0][0]:5d}           {cm[0][1]:5d}",
        f"actual_inject      {cm[1][0]:5d}           {cm[1][1]:5d}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("Loading test set...")
    texts, labels = load_test_set(DB_PATH)
    print(f"Test set: {len(texts)} examples")

    results: list[str] = [
        "# Phase 0 Baseline",
        "",
        f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        "Test set: split_test from unified.duckdb",
        f"Examples: {len(texts)}",
        "",
        "Note: attack_category is 'unknown' for the majority of examples.",
        "Per-category breakdown by taxonomy family is not available at this phase.",
        "That breakdown is the Phase 4 deliverable.",
        "",
    ]

    for name, model_id in MODELS.items():
        print(f"\nRunning {name} ({model_id})...")
        preds = predict(texts, model_id)
        block = format_report(name, labels, preds)
        results.append(block)
        macro_f1 = f1_score(labels, preds, average="macro")
        print(f"  Macro F1: {macro_f1:.4f}")

    os.makedirs("docs", exist_ok=True)
    out_path = "docs/baseline.md"
    with open(out_path, "w") as f:
        f.write("\n".join(results))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
