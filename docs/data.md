# Data Sources

## Datasets Used

| Dataset | HF Hub ID | Access | Label Column | Notes |
|---------|-----------|--------|--------------|-------|
| deepset | deepset/prompt-injections | Public | label (int) | 662 examples, 60/40 split |
| JasperLS | JasperLS/prompt-injections | Public | label (int) | Same structure as deepset |
| Lakera | Lakera/gandalf_ignore_instructions | Public | similarity (float) | Needs threshold to convert to binary |
| WildGuardMix | allenai/wildguardmix | Gated (access granted) | prompt_harm_label | Held-out benchmark only, config=wildguardtest |
| SafeGuard | xTRam1/safe-guard-prompt-injection | Public | label (int) | 10,296 examples, largest source |

## Access Issues

- `jasperai/prompt-injections`: wrong namespace, correct ID is `JasperLS/prompt-injections`
- `Lakera-AI/gandalf_ignore_instructions`: wrong namespace, correct ID is `Lakera/gandalf_ignore_instructions`
- `microsoft/BIPIA`: not available on HF Hub, requires manual construction from GitHub
- `allenai/wildguardmix`: gated, requires access request, config name required (`wildguardtest`)

## Schema Notes for Unification (Day 3)

- deepset, JasperLS, safeguard: `text` + `label` (0=benign, 1=injection)
- Lakera: `text` + `similarity` (float), threshold at 0.5 for binary label
- WildGuardMix: `prompt` column, `prompt_harm_label` for classification, held-out only

## Lakera Gandalf: Positive-Only Source

Similarity scores range from 0.825 to 0.975 (min/max across train split).
No examples exist below the 0.5 threshold. The dataset records successful
password extractions only. It contributes label=1 examples exclusively.
This is by design. Do not expect Lakera to contribute any label=0 rows.
