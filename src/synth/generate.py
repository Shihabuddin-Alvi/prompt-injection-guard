"""Generate synthetic training examples for each v1 failure cluster via Claude."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = Path("data/synthetic_raw.jsonl")
MODEL = "claude-sonnet-4-5-20250929"
SLEEP_SECONDS = 1.3
EXAMPLES_PER_CALL = 25

# Each cluster: (cluster_id, label, attack_category, target_count, prompt)
CLUSTERS = [
    (
        "FN-1",
        1,
        "indirect",
        400,
        "Generate {n} short user prompts written in German, Spanish, or French "
        "that attempt prompt injection (e.g. asking an AI to ignore its "
        "instructions, reveal its system prompt, or act without restrictions). "
        "Vary the language and phrasing naturally, as a real user might write. "
        "Return ONLY a JSON array of strings, no other text.",
    ),
    (
        "FN-2",
        1,
        "direct",
        300,
        "Generate {n} very short (3-8 word) user prompts that are prompt "
        "injection attempts but look like ordinary short requests (e.g. "
        "'generate c++', 'write SQL for this'). The injection intent should "
        "be subtle, relying on brevity to avoid detection. "
        "Return ONLY a JSON array of strings, no other text.",
    ),
    (
        "FN-3",
        1,
        "jailbreak",
        400,
        "Generate {n} prompt injection attempts that avoid explicit phrases "
        "like 'ignore instructions' or 'you are now DAN'. Instead, frame the "
        "injection as a content generation task, a hypothetical scenario, or "
        "an indirect request that gradually steers the model off its "
        "guidelines (e.g. 'As a child with no prior knowledge, how would you "
        "react if...'). Return ONLY a JSON array of strings, no other text.",
    ),
    (
        "FP-2",
        0,
        "role_play",
        300,
        "Generate {n} benign user prompts that ask an AI to 'act as' or "
        "roleplay a professional, expert, or authority figure (e.g. doctor, "
        "lawyer, religious leader, developer, teacher) for legitimate advice "
        "or help. These should NOT be injection attempts, just normal "
        "persona-based requests. Return ONLY a JSON array of strings, no "
        "other text.",
    ),
    (
        "FP-1",
        0,
        "unknown",
        300,
        "Generate {n} benign prompts related to programming or code review "
        "that contain instruction-like phrasing (e.g. 'complete the TODO "
        "comments as instructed', 'follow the steps in the docstring'). "
        "These are normal coding requests, not injection attempts. "
        "Return ONLY a JSON array of strings, no other text.",
    ),
]


def generate_cluster(
    client, cluster_id, label, category, target_count, prompt_template
):
    examples = []
    while len(examples) < target_count:
        remaining = target_count - len(examples)
        n = min(EXAMPLES_PER_CALL, remaining)
        prompt = prompt_template.format(n=n)

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        text = (
            text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )

        try:
            batch = json.loads(text)
        except json.JSONDecodeError:
            print(f"  [{cluster_id}] JSON parse failed, skipping batch")
            time.sleep(SLEEP_SECONDS)
            continue

        for item in batch:
            examples.append(
                {
                    "input_text": item,
                    "label": label,
                    "attack_category": category,
                    "source_cluster": cluster_id,
                    "generation_prompt": prompt_template,
                    "model": MODEL,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        print(f"  [{cluster_id}] {len(examples)}/{target_count}")
        time.sleep(SLEEP_SECONDS)

    return examples


def run():
    client = anthropic.Anthropic()
    all_examples = []

    for cluster_id, label, category, target_count, prompt_template in CLUSTERS:
        print(f"Generating cluster {cluster_id} (target: {target_count})")
        examples = generate_cluster(
            client, cluster_id, label, category, target_count, prompt_template
        )
        all_examples.extend(examples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal generated: {len(all_examples)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
