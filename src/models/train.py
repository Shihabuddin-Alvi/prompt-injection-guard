"""Fine-tune DeBERTa-v3-base for prompt injection classification."""

import argparse

import duckdb
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/deberta-v3-base"
DB_PATH = "data/unified.duckdb"
MAX_LENGTH = 256


def load_splits(
    db_path: str, smoke_test: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(db_path)
    train_df = con.execute("SELECT input_text, label FROM split_train").fetchdf()
    val_df = con.execute("SELECT input_text, label FROM split_val").fetchdf()
    con.close()

    if smoke_test:
        train_df = train_df.sample(n=500, random_state=42)
        val_df = val_df.sample(n=100, random_state=42)

    return train_df, val_df


def tokenize(df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    dataset = Dataset.from_pandas(
        df.rename(columns={"input_text": "text", "label": "labels"})
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def train(
    output_dir: str = "checkpoints/v1",
    smoke_test: bool = False,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
):
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("Loading data...")
    train_df, val_df = load_splits(DB_PATH, smoke_test=smoke_test)
    print(f"  train: {len(train_df)}  val: {len(val_df)}")

    train_dataset = tokenize(train_df, tokenizer)
    val_dataset = tokenize(val_df, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints/v1")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        num_epochs=args.epochs,
    )
