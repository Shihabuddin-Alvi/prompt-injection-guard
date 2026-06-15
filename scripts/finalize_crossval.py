"""Produce final filtered synthetic dataset from crossval progress file."""

import json
from pathlib import Path

PROGRESS_PATH = Path("data/synthetic_crossval_progress.jsonl")
OUTPUT_PATH = Path("data/synthetic_filtered.jsonl")


def run():
    with open(PROGRESS_PATH) as f:
        examples = [json.loads(line) for line in f]

    kept = [ex for ex in examples if ex["agree"]]
    agree = len(kept)
    disagree = len(examples) - agree

    print(f"Total labeled: {len(examples)}")
    print(f"Agree (kept): {agree}")
    print(f"Disagree (dropped): {disagree}")
    print(f"Agreement rate: {agree / len(examples):.3f}")

    with open(OUTPUT_PATH, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
