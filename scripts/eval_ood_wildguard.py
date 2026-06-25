"""OOD evaluation on WildGuardMix adversarial subset.

Scope: adversarial=True examples from WildGuardMix wildguardtest.
This is NOT a prompt injection benchmark — WildGuardMix covers general
harmful inputs (hate speech, fraud, cyberattack, etc). It serves as an
out-of-distribution generalization check: does the classifier transfer
to adversarial harmful inputs outside its training distribution?

Limitation: without a prompt-injection-specific OOD set, we cannot
directly validate whether synthetic data improved detection of the
failure clusters (non-English injections, short inputs, indirect
jailbreaks). That benchmark requires a purpose-built adversarial set.
"""

import os
import json
import argparse
import numpy as np
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, f1_score

load_dotenv()

MAX_LENGTH = 256


def load_wildguard_adversarial() -> tuple[list[str], list[int]]:
    """Load adversarial subset and map harm labels to binary injection labels."""
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=token)
    df = ds["test"].to_pandas()
    adv = df[df["adversarial"]].copy()
    # harmful -> 1 (injection proxy), unharmful -> 0 (benign proxy)
    adv["label"] = (adv["prompt_harm_label"] == "harmful").astype(int)
    texts = adv["prompt"].tolist()
    labels = adv["label"].tolist()
    return texts, labels


def predict(texts: list[str], model_path: str, batch_size: int = 32) -> np.ndarray:
    """Run inference with the classifier."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            preds = torch.argmax(outputs.logits, dim=1).numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def run_ood_eval(model_path: str, output_path: str) -> None:
    """Evaluate model on WildGuardMix adversarial subset."""
    print("Loading WildGuardMix adversarial subset...")
    texts, labels = load_wildguard_adversarial()
    labels_arr = np.array(labels)
    print(f"  Total adversarial examples: {len(texts)}")
    print(f"  Harmful (injection proxy): {labels_arr.sum()}")
    print(f"  Unharmful (benign proxy): {(1 - labels_arr).sum()}")

    print(f"\nRunning inference with {model_path}...")
    preds = predict(texts, model_path)

    macro_f1 = f1_score(labels_arr, preds, average="macro")
    print(f"\nMacro F1 (OOD): {macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            labels_arr, preds, target_names=["benign_proxy", "harmful_proxy"]
        )
    )

    results = {
        "benchmark": "wildguardmix_adversarial",
        "scope": "adversarial=True subset, harm label as injection proxy",
        "limitation": "not prompt-injection-specific; measures generalization to adversarial harmful inputs",
        "n_total": len(texts),
        "n_harmful": int(labels_arr.sum()),
        "n_unharmful": int((1 - labels_arr).sum()),
        "macro_f1": macro_f1,
        "model_path": model_path,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="docs/ood_results.json")
    args = parser.parse_args()
    run_ood_eval(model_path=args.model_path, output_path=args.output)
