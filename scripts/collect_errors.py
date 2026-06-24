"""Run classifier v1 on the val set and collect all misclassified examples."""

from pathlib import Path

import duckdb
import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from scipy.special import softmax
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "alvi42/prompt-injection-guard-v1"
ONNX_DIR = Path("checkpoints/v1-onnx")
DB_PATH = "data/unified.duckdb"
MAX_LENGTH = 256
BATCH_SIZE = 32


def run():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = ORTModelForSequenceClassification.from_pretrained(ONNX_DIR)

    con = duckdb.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, input_text, label, attack_category FROM split_val"
    ).fetchall()
    print(f"Val set: {len(rows)} examples")

    errors = []
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [r[1] for r in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        logits = model(**encoded).logits
        probs = softmax(logits, axis=1)
        preds = np.argmax(probs, axis=1)

        for j, row in enumerate(batch):
            row_id, text, true_label, category = row
            pred = int(preds[j])
            if pred != true_label:
                error_type = "false_positive" if pred == 1 else "false_negative"
                errors.append(
                    {
                        "id": row_id,
                        "input_text": text,
                        "true_label": true_label,
                        "predicted_label": pred,
                        "error_type": error_type,
                        "attack_category": category,
                        "confidence": float(probs[j][pred]),
                    }
                )

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

    print(f"\nTotal errors: {len(errors)}")
    print(
        f"  False positives: {sum(1 for e in errors if e['error_type'] == 'false_positive')}"
    )
    print(
        f"  False negatives: {sum(1 for e in errors if e['error_type'] == 'false_negative')}"
    )

    con.execute("DROP TABLE IF EXISTS v1_errors")
    con.execute(
        """
        CREATE TABLE v1_errors (
            id TEXT,
            input_text TEXT,
            true_label INTEGER,
            predicted_label INTEGER,
            error_type TEXT,
            attack_category TEXT,
            confidence FLOAT
        )
    """
    )
    for e in errors:
        con.execute(
            "INSERT INTO v1_errors VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                e["id"],
                e["input_text"],
                e["true_label"],
                e["predicted_label"],
                e["error_type"],
                e["attack_category"],
                e["confidence"],
            ],
        )
    con.close()
    print(f"\nSaved to v1_errors table in {DB_PATH}")


if __name__ == "__main__":
    run()
