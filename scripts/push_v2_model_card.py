"""Push the v2 model card to HF Hub."""

from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_CARD = """---
license: mit
language:
- en
tags:
- text-classification
- prompt-injection
- security
- deberta-v2
metrics:
- f1
---

# Prompt Injection Guard v2

DeBERTa-v3-base fine-tuned for binary prompt injection classification.
Trained on the original 8,183-example dataset plus 1,687 synthetic examples
targeting 5 documented v1 failure patterns.

## Results

| Benchmark | v1 | v2 | Delta |
|---|---|---|---|
| split_test (n=1,754) | 0.9957 | 0.9932 | -0.0025 |

The standard test set is saturated (v1 already missed only 7 of 1,754
examples). v2 was not measured to beat v1 on this benchmark by design intent
of the synthetic data, which targets categories underrepresented in
split_test. On a held-out adversarial slice built from the failure clusters
(n=651, injection-heavy), v2 achieves 0.99 injection recall and 1.00 benign
recall.

## Training Data

9,870 examples: 8,183 original (5 public datasets, deduplicated) + 1,687
synthetic examples generated via Claude Sonnet, deduplicated against existing
training data via sentence embeddings, cross-validated with Claude Haiku
(99.2% label agreement).

Synthetic data targets 5 failure clusters identified from v1 error analysis:
non-English injections, very short injections, indirect jailbreaks framed as
hypotheticals, legitimate role-play requests, and instruction-adjacent coding
phrasing.

## Training Details

- Base model: microsoft/deberta-v3-base
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 8 (gradient accumulation 2, effective 16)
- Hardware: Colab T4

## Full Methodology

See the [GitHub repository](https://github.com/Shihabuddin-Alvi/prompt-injection-guard)
for the complete failure analysis, synthetic data pipeline, and quality filter
documentation.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("alvi42/prompt-injection-guard-v2")
model = AutoModelForSequenceClassification.from_pretrained("alvi42/prompt-injection-guard-v2")
```
"""


def run():
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id="alvi42/prompt-injection-guard-v2",
        repo_type="model",
    )
    print("Model card pushed to alvi42/prompt-injection-guard-v2")


if __name__ == "__main__":
    run()
