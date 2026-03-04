# Sign Language Synthesis

## Goal

Build a system to synthesize **continuous, natural-looking sign language video** from text input.

**Core problem:** Full sentence-level sign language video datasets are extremely scarce. Instead, only isolated word-level videos exist. The goal is to combine these words into complete, smooth sentences as if signed by a real person.

---

## Pipeline Overview

```
Input: "Hello my name is Ram"   ← text
              |
    [Layer 1: NLP / Grammar]
    - Convert to sign language gloss order
    - Sign language has different grammar from spoken language
    - e.g. "I don't like it" → ASL gloss: "I LIKE NOT"
              |
    [Layer 2: Synthesis]
    - Lookup skeleton (.npy) for each gloss token
    - Trim rest frames from start/end of each word
    - Generate transition frames between consecutive words
    - Render skeleton sequence -> video
              |
Output: Complete, smooth signed sentence video
```

---

## Why Transitions Matter

In real signing, hands **never stop** between words — they move directly from the end pose of one sign into the start pose of the next (coarticulation). Since isolated word videos have "rest poses" at their boundaries, two things are required:

1. **Trim rest poses** — remove the idle hand frames at word boundaries
2. **Generate transitions** — synthesize smooth motion from end of word A to start of word B

---

## Completeness Levels

| Level | Requirements |
|---|---|
| **Intelligible** — viewer understands the meaning | Word concat + trim + transitions |
| **Fluent** — looks natural | + Coarticulation modeling |
| **Native-like** — indistinguishable from a real signer | + Facial expressions (NMMs) + Prosody + Spatial grammar |

Current target: **Fluent** — high-quality transition generation.

---

## What's Still Missing

- [ ] Layer 1: NLP to convert text → sign language gloss
- [ ] Non-manual markers: facial expressions (grammar for questions, negation, etc.)
- [ ] Prosody: rhythm and speed variation based on semantic emphasis
- [ ] End-to-end evaluation with real users (deaf community)
