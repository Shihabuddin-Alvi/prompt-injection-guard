"""Compare v1 and v2 on the adversarial slice (n=651, injection-heavy)."""

import os

import duckdb
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]


def build_adversarial_slice() -> tuple[list[str], np.ndarray]:
    con = duckdb.connect("data/unified.duckdb")
    df = con.execute("""
        SELECT input_text, label
        FROM split_val
        WHERE attack_category IN ('role_play', 'indirect') OR label = 1
        """).fetchdf()
    con.close()
    return df["input_text"].tolist(), df["label"].astype(int).to_numpy()


def run_inference(repo_id: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=HF_TOKEN)
    model.eval()

    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=1).tolist())
    return np.array(preds)


def main() -> None:
    texts, labels = build_adversarial_slice()
    print(f"Adversarial slice: {len(texts)} examples")

    print("\nRunning v1 inference...")
    v1_preds = run_inference("alvi42/prompt-injection-guard-v1", texts)
    print("=== v1 on adversarial slice ===")
    print(classification_report(labels, v1_preds, target_names=["benign", "injection"]))

    print("\nRunning v2 inference...")
    v2_preds = run_inference("alvi42/prompt-injection-guard-v2", texts)
    print("=== v2 on adversarial slice ===")
    print(classification_report(labels, v2_preds, target_names=["benign", "injection"]))


if __name__ == "__main__":
    main()
