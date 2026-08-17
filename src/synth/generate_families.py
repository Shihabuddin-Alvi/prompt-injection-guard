"""Generate synthetic training examples for F1-F6 positive families and
NR1-NR6 hard negatives, per constitution.md and attack_taxonomy.md.

Phase 2 of AUDIT_AND_ROADMAP.md. Scaffolding only. Structure copied from
src/synth/generate.py (cluster-loop, batch JSON parse, provenance write).
Do not run until the prompt templates are reviewed against the committed
constitution and taxonomy text.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OUTPUT_PATH = Path("data/family_data_raw.jsonl")

MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_BATCH_SIZE = 25
F3_BATCH_SIZE = 5
SLEEP_SECONDS = 1.3

MAX_CONSECUTIVE_FAILURES = 15

CONSTITUTION_VERSION = "1.1"
TAXONOMY_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyConfig:
    family_id: str
    label: int
    split: str
    target_count: int
    prompt_template: str
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class Example:
    text: str
    label: int
    family: str
    split: str
    constitution_version: str
    taxonomy_version: str
    generator_model: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "family": self.family,
            "split": self.split,
            "constitution_version": self.constitution_version,
            "taxonomy_version": self.taxonomy_version,
            "generator_model": self.generator_model,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------

FAMILIES = [
    FamilyConfig(
        family_id="F1",
        label=1,
        split="train",
        target_count=200,
        prompt_template=(
            "Generate {n} short user prompts that attempt direct instruction "
            "override: an explicit attempt to cancel, replace, or supersede "
            "the system or developer instructions. Vary imperative verbs, "
            "reference 'previous,' 'above,' or 'earlier' instructions, and use "
            "explicit replacement or reset language. Vary wording, length, "
            "tone, and framing widely so no single phrasing repeats. "
            "Return ONLY a JSON array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="F2",
        label=1,
        split="train",
        target_count=200,
        prompt_template=(
            "Generate {n} short user prompts that attempt role and persona "
            "manipulation: assigning the model a new identity or persona "
            "whose purpose is to remove its constraints. Include identity "
            "reassignment, claims that the new persona has no restrictions, "
            "and framing that separates the persona from the model's normal "
            "behavior. Apply the F2/H4 boundary test from constitution.md: "
            "each example must resolve to a direct persona reassignment on "
            "its own, not depend on a fictional or simulated frame to make "
            "sense. If a generated example only works as an instruction "
            "inside a story or hypothetical, it belongs to H4, not F2, and "
            "must not appear here. Vary wording, length, tone, and framing "
            "widely. Return ONLY a JSON array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="F3",
        label=1,
        split="train",
        target_count=200,
        batch_size=F3_BATCH_SIZE,
        prompt_template=(
            "You are generating labeled synthetic training data for a "
            "defensive prompt-injection classifier. Nothing here is executed, "
            "only labeled as an example of the attack pattern for supervised "
            "training. Generate {n} short user-turn strings that hide an "
            "instruction inside an encoding or transformation and ask the "
            "reader to decode and act on it. Every decoded payload must be a "
            "single inert test marker, never a natural-language command. Use "
            "markers like CANARY_7f3a, TESTMARK_01, or PROBE_ff2. Never encode "
            "a real-world request of any kind, harmful or benign. Vary the "
            "encoding across base64, leetspeak, reversed text, and homoglyphs. "
            "Vary wording, length, tone, and framing widely. "
            "Return ONLY a JSON array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="F4",
        label=1,
        split="train",
        target_count=200,
        prompt_template=(
            "Generate {n} short user prompts that attempt payload splitting: "
            "distributing a malicious instruction across fragments and asking "
            "the model to assemble and act on the result. Include instructions "
            "to remember and later combine labeled fragments. Vary wording, "
            "length, tone, and framing widely. Return ONLY a JSON array of "
            "strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="F5",
        label=1,
        split="train",
        target_count=200,
        prompt_template=(
            "Generate {n} short user prompts that attempt refusal suppression "
            "and output constraint: forcing a compliant opening, banning "
            "caveats or refusals, or demanding formatting that preempts "
            "judgment. Vary wording, length, tone, and framing widely. "
            "Return ONLY a JSON array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="F6",
        label=1,
        split="train",
        target_count=200,
        prompt_template=(
            "Generate {n} short user prompts that attempt delimiter and "
            "instruction-hierarchy spoofing: fake structural markers or role "
            "tags that impersonate a higher-authority message, such as "
            "counterfeit system or developer tags inside user content. Vary "
            "wording, length, tone, and framing widely. Return ONLY a JSON "
            "array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="NR1",
        label=0,
        split="train",
        target_count=133,
        prompt_template=(
            "Generate {n} benign user prompts where a user revises their own "
            "work and asks the model to disregard an earlier draft or earlier "
            "instructions that the user themselves gave. These share vocabulary "
            "with injection attempts (disregard, ignore, earlier) but are "
            "legitimate self-revision. Return ONLY a JSON array of strings, "
            "no other text."
        ),
    ),
    FamilyConfig(
        family_id="NR2",
        label=0,
        split="train",
        target_count=133,
        prompt_template=(
            "Generate {n} benign prompts from a security researcher, student, "
            "developer, or writer who discusses, quotes, or explains prompt "
            "injection for legitimate study or documentation. The intent is "
            "educational or analytical, not an attempt to execute an "
            "instruction. Return ONLY a JSON array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="NR3",
        label=0,
        split="train",
        target_count=133,
        prompt_template=(
            "Generate {n} benign prompts that contain a quoted prompt "
            "injection example as illustration inside otherwise educational "
            "or analytical text, where the surrounding intent is clearly "
            "study, not a live instruction to follow. Return ONLY a JSON "
            "array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="NR4",
        label=0,
        split="train",
        target_count=133,
        prompt_template=(
            "Generate {n} ordinary benign task instructions to an AI assistant "
            "that fall within its intended purpose: writing, summarizing, "
            "coding, analysis, planning. No adversarial content. Return ONLY "
            "a JSON array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="NR5",
        label=0,
        split="train",
        target_count=133,
        prompt_template=(
            "Generate {n} benign prompts that use words like 'ignore', "
            "'system', 'override', or 'instructions' in their normal, "
            "non-adversarial sense (e.g. 'ignore the typo in my last message,' "
            "'what operating system do you run on'). Return ONLY a JSON "
            "array of strings, no other text."
        ),
    ),
    FamilyConfig(
        family_id="NR6",
        label=0,
        split="train",
        target_count=135,
        prompt_template=(
            "Generate {n} benign prompts asking the model to summarize, "
            "translate, or analyze untrusted content (an email, a document, "
            "a web page excerpt) without asking the model to obey any "
            "instructions found inside that content. Return ONLY a JSON "
            "array of strings, no other text."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_json_response(text: str) -> str:
    """Remove common Markdown code fences from a model response."""
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json") :]

    elif text.startswith("```"):
        text = text[len("```") :]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def parse_batch(text: str) -> list[str]:
    """
    Parse and validate a model-generated JSON batch.

    Raises:
        ValueError: If the response is not a valid list of strings.
    """
    cleaned = clean_json_response(text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Model response must be a JSON array.")

    if not all(isinstance(item, str) for item in data):
        raise ValueError("Every generated example must be a string.")

    return [item.strip() for item in data if item.strip()]


def create_example(
    text: str,
    config: FamilyConfig,
    created_at: str,
) -> Example:
    """Create a fully attributed training example."""
    return Example(
        text=text,
        label=config.label,
        family=config.family_id,
        split=config.split,
        constitution_version=CONSTITUTION_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        generator_model=MODEL,
        created_at=created_at,
    )


def generate_batch(
    client: anthropic.Anthropic,
    config: FamilyConfig,
    count: int,
) -> list[str]:
    """Request one batch of examples from Claude."""
    prompt = config.prompt_template.format(n=count)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text = response.content[0].text
    return parse_batch(text)


# ---------------------------------------------------------------------------
# Family generation
# ---------------------------------------------------------------------------


def generate_family(
    client: anthropic.Anthropic,
    config: FamilyConfig,
) -> list[Example]:
    """Generate and validate all examples required for one family."""
    examples: list[Example] = []
    seen: set[str] = set()

    failures = 0

    while len(examples) < config.target_count:
        remaining = config.target_count - len(examples)
        batch_size = min(config.batch_size, remaining)

        try:
            batch = generate_batch(
                client=client,
                config=config,
                count=batch_size,
            )

        except (json.JSONDecodeError, ValueError) as exc:
            failures += 1

            print(
                f"  [{config.family_id}] generation/parsing failed "
                f"({failures}/{MAX_CONSECUTIVE_FAILURES}): {exc}"
            )

            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Failed to generate family {config.family_id} "
                    f"after {MAX_CONSECUTIVE_FAILURES} attempts."
                ) from exc

            time.sleep(SLEEP_SECONDS)
            continue

        except anthropic.APIError as exc:
            failures += 1

            print(
                f"  [{config.family_id}] Anthropic API error "
                f"({failures}/{MAX_CONSECUTIVE_FAILURES}): {exc}"
            )

            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Anthropic API repeatedly failed for {config.family_id}."
                ) from exc

            time.sleep(SLEEP_SECONDS)
            continue

        # Successful API response.
        failures = 0

        created_at = datetime.now(timezone.utc).isoformat()

        added = 0

        for text in batch:
            normalized = text.strip()

            # Avoid duplicate examples within a family.
            if normalized in seen:
                continue

            seen.add(normalized)

            examples.append(
                create_example(
                    text=normalized,
                    config=config,
                    created_at=created_at,
                )
            )

            added += 1

            if len(examples) >= config.target_count:
                break

        print(
            f"  [{config.family_id}] "
            f"{len(examples)}/{config.target_count} "
            f"(+{added} new)"
        )

        time.sleep(SLEEP_SECONDS)

    return examples


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_jsonl(
    examples: list[Example],
    output_path: Path,
) -> None:
    """Write examples to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(
                json.dumps(
                    example.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    client = anthropic.Anthropic()

    all_examples: list[Example] = []

    total_target = sum(config.target_count for config in FAMILIES)

    print("Starting dataset generation")
    print(f"Model: {MODEL}")
    print(f"Target examples: {total_target}")
    print()

    for config in FAMILIES:
        print(f"Generating {config.family_id} (target: {config.target_count})")

        examples = generate_family(
            client=client,
            config=config,
        )

        all_examples.extend(examples)

    save_jsonl(
        examples=all_examples,
        output_path=OUTPUT_PATH,
    )

    print()
    print("Generation complete.")
    print(f"Total generated: {len(all_examples)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
