"""Generate synthetic training examples for F1-F6 positive families and
NR1-NR6 hard negatives, per constitution.md and attack_taxonomy.md.

Phase 2 of AUDIT_AND_ROADMAP.md. Scaffolding only. Structure copied from
src/synth/generate.py (cluster-loop, batch JSON parse, provenance write).
Do not run until the prompt templates are reviewed against the committed
constitution and taxonomy text.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = Path("data/family_data_raw.jsonl")
MODEL = "claude-sonnet-4-5-20250929"
SLEEP_SECONDS = 1.3
EXAMPLES_PER_CALL = 25

CONSTITUTION_VERSION = "1.1"
TAXONOMY_VERSION = "1.0"

FAMILIES = [
    (
        "F1",
        1,
        "train",
        200,
        "Generate {n} short user prompts that attempt direct instruction "
        "override: an explicit attempt to cancel, replace, or supersede the "
        "system or developer instructions. Vary imperative verbs, reference "
        "'previous,' 'above,' or 'earlier' instructions, and use explicit "
        "replacement or reset language. Vary wording, length, tone, and "
        "framing widely so no single phrasing repeats. "
        "Return ONLY a JSON array of strings, no other text.",
    ),
    (
        "F2",
        1,
        "train",
        200,
        "Generate {n} short user prompts that attempt role and persona "
        "manipulation: assigning the model a new identity or persona whose "
        "purpose is to remove its constraints. Include identity "
        "reassignment, claims that the new persona has no restrictions, and "
        "framing that separates the persona from the model's normal "
        "behavior. Apply the F2/H4 boundary test from constitution.md: each "
        "example must resolve to a direct persona reassignment on its own, "
        "not depend on a fictional or simulated frame to make sense. If a "
        "generated example only works as an instruction inside a story or "
        "hypothetical, it belongs to H4, not F2, and must not appear here. "
        "Vary wording, length, tone, and framing widely. "
        "Return ONLY a JSON array of strings, no other text.",
    ),
    (
        "F3",
        1,
        "train",
        200,
        "Generate {n} short user prompts that attempt encoding and "
        "obfuscation: hiding an instruction inside an encoding or "
        "transformation and asking the model to decode and act on it. The "
        "decoded instruction itself must always be harmless and generic, such "
        "as asking the model to reveal a placeholder phrase, respond with a "
        "fixed word, or state its current instructions. Never encode a "
        "real-world harmful request (violence, hacking, illegal acts) even as "
        "an example. Vary the encoding across base64, leetspeak, reversed "
        "text, and homoglyphs. Vary wording, length, tone, and framing "
        "widely. Return ONLY a JSON array of strings, no other text.",
    ),
    (
        "F4",
        1,
        "train",
        200,
        "Generate {n} short user prompts that attempt payload splitting: "
        "distributing a malicious instruction across fragments and asking "
        "the model to assemble and act on the result. Include instructions "
        "to remember and later combine labeled fragments. Vary wording, "
        "length, tone, and framing widely. Return ONLY a JSON array of "
        "strings, no other text.",
    ),
    (
        "F5",
        1,
        "train",
        200,
        "Generate {n} short user prompts that attempt refusal suppression "
        "and output constraint: forcing a compliant opening, banning "
        "caveats or refusals, or demanding formatting that preempts "
        "judgment. Vary wording, length, tone, and framing widely. Return "
        "ONLY a JSON array of strings, no other text.",
    ),
    (
        "F6",
        1,
        "train",
        200,
        "Generate {n} short user prompts that attempt delimiter and "
        "instruction-hierarchy spoofing: fake structural markers or role "
        "tags that impersonate a higher-authority message, such as "
        "counterfeit system or developer tags inside user content. Vary "
        "wording, length, tone, and framing widely. Return ONLY a JSON "
        "array of strings, no other text.",
    ),
    (
        "NR1",
        0,
        "train",
        133,
        "Generate {n} benign user prompts where a user revises their own "
        "work and asks the model to disregard an earlier draft or earlier "
        "instructions that the user themselves gave. These share vocabulary "
        "with injection attempts (disregard, ignore, earlier) but are "
        "legitimate self-revision. Return ONLY a JSON array of strings, no "
        "other text.",
    ),
    (
        "NR2",
        0,
        "train",
        133,
        "Generate {n} benign prompts from a security researcher, student, "
        "developer, or writer who discusses, quotes, or explains prompt "
        "injection for legitimate study or documentation. The intent is "
        "educational or analytical, not an attempt to execute an "
        "instruction. Return ONLY a JSON array of strings, no other text.",
    ),
    (
        "NR3",
        0,
        "train",
        133,
        "Generate {n} benign prompts that contain a quoted prompt injection "
        "example as illustration inside otherwise educational or analytical "
        "text, where the surrounding intent is clearly study, not a live "
        "instruction to follow. Return ONLY a JSON array of strings, no "
        "other text.",
    ),
    (
        "NR4",
        0,
        "train",
        133,
        "Generate {n} ordinary benign task instructions to an AI assistant "
        "that fall within its intended purpose: writing, summarizing, "
        "coding, analysis, planning. No adversarial content. Return ONLY a "
        "JSON array of strings, no other text.",
    ),
    (
        "NR5",
        0,
        "train",
        133,
        "Generate {n} benign prompts that use words like 'ignore,' "
        "'system,' 'override,' or 'instructions' in their normal, "
        "non-adversarial sense (e.g. 'ignore the typo in my last message,' "
        "'what operating system do you run on'). Return ONLY a JSON array "
        "of strings, no other text.",
    ),
    (
        "NR6",
        0,
        "train",
        135,
        "Generate {n} benign prompts asking the model to summarize, "
        "translate, or analyze untrusted content (an email, a document, a "
        "web page excerpt) without asking the model to obey any "
        "instructions found inside that content. Return ONLY a JSON array "
        "of strings, no other text.",
    ),
]


def generate_family(client, family_id, label, split, target_count, prompt_template):
    examples = []
    consecutive_failures = 0
    max_consecutive_failures = 5
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
            consecutive_failures += 1
            print(
                f"  [{family_id}] JSON parse failed ({consecutive_failures}/{max_consecutive_failures})"
            )
            print(f"  [{family_id}] raw response: {text[:300]!r}")
            if consecutive_failures >= max_consecutive_failures:
                print(
                    f"  [{family_id}] giving up after {max_consecutive_failures} consecutive failures"
                )
                break
            time.sleep(SLEEP_SECONDS)
            continue
        consecutive_failures = 0
        for item in batch:
            examples.append(
                {
                    "text": item,
                    "label": label,
                    "family": family_id,
                    "split": split,
                    "constitution_version": CONSTITUTION_VERSION,
                    "taxonomy_version": TAXONOMY_VERSION,
                    "generator_model": MODEL,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        print(f"  [{family_id}] {len(examples)}/{target_count}")
        time.sleep(SLEEP_SECONDS)
    return examples


def run():
    client = anthropic.Anthropic()
    all_examples = []
    for family_id, label, split, target_count, prompt_template in FAMILIES:
        print(f"Generating family {family_id} (target: {target_count})")
        examples = generate_family(
            client, family_id, label, split, target_count, prompt_template
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
