# Prompt Injection Attack Taxonomy

Reference document for Prompt Injection Guard. It defines the attack families the classifier must detect, split into training families and held-out families.

Held-out families never appear in training data. They exist to answer one question: does the classifier learn the concept of injection, or does it memorize surface patterns? A classifier that scores well on training families but fails on held-out families has memorized. A classifier that generalizes to held-out families has learned.

## How to use this document

- Generate synthetic training data only from TRAIN families.
- Generate the held-out evaluation set only from HELD-OUT families.
- Report classifier performance separately for each held-out family, not as a single average.
- The target result is high recall on held-out families with low over-refusal on benign inputs.

## Core definitions

- Trusted content: the developer system prompt and the application's own instructions.
- Untrusted content: anything from the user, retrieved documents, web pages, files, emails, or tool outputs.
- Instruction-data confusion: the root vulnerability. The model treats untrusted content as instructions instead of as data.
- Direct injection: the malicious instruction comes straight from the user input.
- Indirect injection: the malicious instruction hides inside content the model processes as data, such as a document, a retrieved passage, or a tool result.

## Split rationale

The held-out set is chosen to be a real generalization test, not an interpolation test. The four held-out families all share the concept of injection but differ from the training families in delivery vector: indirect delivery, tool-result delivery, multi-turn delivery, and fiction-wrapped delivery. A classifier trained only on single-turn direct attacks has to extend the concept to catch these. That extension is the thing worth measuring.

---

## Training families

### F1. Direct instruction override
- Type: direct
- Definition: an explicit attempt to cancel, replace, or supersede the system or developer instructions.
- Threat model: the user tries to seize control of the model's behavior outright.
- Illustrative pattern: a request to disregard prior instructions and follow new ones supplied by the user instead.
- Detection signals: imperative verbs aimed at prior instructions, references to "previous," "above," or "earlier" instructions, explicit replacement or reset language.

### F2. Role and persona manipulation
- Type: direct
- Definition: assigning the model a new identity or persona whose purpose is to remove its constraints.
- Threat model: the user reframes the model as an entity that has no rules.
- Illustrative pattern: instructing the model to act as an unrestricted alternate assistant with a different rule set.
- Detection signals: identity reassignment, claims that a new persona has no restrictions, framing that separates the persona from the model's normal behavior.

### F3. Encoding and obfuscation
- Type: direct
- Definition: hiding the malicious instruction inside an encoding or transformation to slip past surface filters.
- Threat model: the attacker assumes the defense keys on plain-text keywords.
- Illustrative pattern: a payload delivered as base64, leetspeak, reversed text, homoglyphs, zero-width characters, or a request to decode and then act.
- Detection signals: encoded blocks paired with an instruction to decode and execute, unusual character sets, decode-then-do structure.

### F4. Payload splitting
- Type: direct
- Definition: distributing the malicious instruction across fragments so that no single fragment looks harmful.
- Threat model: the attacker defeats per-chunk detection by assembling intent from parts.
- Illustrative pattern: storing fragments under labels across the input, then asking the model to concatenate and act on the result.
- Detection signals: instructions to remember and later combine fragments, assembly language, references that only resolve when parts are joined.

### F5. Refusal suppression and output constraint
- Type: direct
- Definition: constraining the model's output to block refusal or hedging.
- Threat model: the attacker forces a compliant opening so the model cannot decline.
- Illustrative pattern: instructions to omit all warnings and begin the reply with a fixed compliant phrase.
- Detection signals: bans on caveats or refusals, forced response prefixes, formatting demands that preempt judgment.

### F6. Delimiter and instruction-hierarchy spoofing
- Type: direct
- Definition: faking structural markers to impersonate a higher-authority message.
- Threat model: the attacker exploits how the model weighs system versus user content.
- Illustrative pattern: fake closing and opening tags that pretend to end the system prompt and start a new one.
- Detection signals: counterfeit role markers, fake system or developer tags inside user content, structural tokens that try to redraw the message boundary.

---

## Held-out families

### H1. Indirect injection via retrieved content
- Type: indirect
- Definition: the malicious instruction is embedded in a document, web page, or retrieved passage that the model reads as data.
- Threat model: the highest-severity class for RAG and agentic systems. The user never sees the payload. The content does.
- Illustrative pattern: a passage that, while presented as reference material, instructs the model to take an action on the user's behalf.
- Detection signals: imperative instructions directed at the model appearing inside content framed as data, action requests embedded in otherwise descriptive text.

### H2. Tool-output injection
- Type: indirect
- Definition: the malicious instruction is embedded in the output of a tool the agent called, such as an API response, file content, or search result.
- Threat model: an agent trusts tool results and acts on injected instructions returned by a compromised or hostile source.
- Illustrative pattern: a function result that contains, alongside legitimate data, an instruction for the agent to perform a further action.
- Detection signals: instructions to the agent appearing inside structured tool output, action directives mixed into returned data.

### H3. Multi-turn crescendo
- Type: multi-turn delivery
- Definition: the injection is built across a conversation, where each turn is benign but the cumulative effect is malicious.
- Threat model: the attacker evades single-message detection by escalating gradually.
- Illustrative pattern: a sequence of small, individually acceptable steps that together lead the model outside its intended behavior.
- Detection signals: gradual escalation across turns, later turns that reference and build on earlier setup, intent that is visible only in aggregate.

### H4. Virtualization and nested framing
- Type: direct
- Definition: wrapping the payload in fiction, a hypothetical, a simulation, or a quoted example to launder it past the model's judgment.
- Threat model: the attacker bets that a fictional or hypothetical frame disables the model's guard.
- Illustrative pattern: a story or simulation in which a character supplies the instruction the attacker actually wants executed.
- Detection signals: nested framing that delivers a real instruction through a fictional vehicle, simulation or story wrappers around an actionable request.

---

## Benign contrast set

The held-out evaluation must include benign inputs that share vocabulary and structure with attacks. These measure over-refusal, which is the metric most portfolio classifiers ignore. Examples of benign-but-adjacent content:

- A user revising their own work who asks the model to disregard an earlier draft they themselves wrote.
- A security student, researcher, or developer discussing or quoting prompt injection for legitimate study.
- A document that contains an injection example as a quoted illustration inside analytical text.
- Ordinary task instructions that use words like "ignore," "system," or "instructions" in a non-adversarial way.

A classifier that flags these has high over-refusal. Driving over-refusal down without losing held-out recall is the core difficulty of the project.
