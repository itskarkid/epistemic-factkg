# Synthetic Data — EpistemicFactKG

Consolidated reference for shortcut-breaking strategy, templates, IS jitter, and NEI distribution.
Original decisions: [ADR-012](adr/012-shortcut-leakage-and-synthetic-data-strategy.md), [ADR-013](adr/013-synthetic-pipeline.md), [ADR-017](adr/017-is-jitter-synthetic.md), [ADR-018](adr/018-nei-heavy-synthetic-distribution.md).

---

## Purpose

Synthetic records serve two goals:

1. **Shortcut-breaking** ([ADR-012](adr/012-shortcut-leakage-and-synthetic-data-strategy.md)):
   Force the model to use the EC formula rather than a stance→verdict lookup.
   A shortcut-breaking record has the same stance value but a different verdict depending
   on source trust and inference strength. Minimum 35% of synthetic records must be
   shortcut-breaking.

2. **NEI coverage** ([ADR-018](adr/018-nei-heavy-synthetic-distribution.md)):
   Crowdsourced datasets (AVeriTeC) under-represent NEI. Synthetic data compensates
   by targeting ~55% NEI-producing templates.

---

## Template Distribution (NEI-heavy)

| Category | Templates | Shortcut-breaking | Share |
|---|---|---|---|
| High-trust clear-verdict | `high_trust_supported`, `high_trust_refuted` | No | ~21% + ~24% |
| Low-trust → NEE | `low_trust_nee`, `low_trust_refuted_nee` | Yes | ~11% + ~11% |
| Asymmetric compound | `strong_support_weak_refute`, `weak_support_strong_refute` | Yes | — |
| Weak vs weak | `weak_vs_weak_nee` | Yes | ~12% |
| Inference → NEE | `inference_nee` | Yes | ~11% |
| Non-apprehension | `non_apprehension_absent`, `non_apprehension_refuted`, `non_apprehension_weak_nee` | Partial | ~10% |
| Conflicting | `conflicting` | Yes | — |
| Perception (AI2THOR) | `perception_direct` | No | — |

NEI-heavy design raises NEI share in full training set from ~11% toward ~24%.

---

## IS Jitter ([ADR-017](adr/017-is-jitter-synthetic.md))

Template-assigned IS values are Gaussian-jittered to prevent ISHead from memorizing
discrete template values:

```
IS_final = clip(IS_template + N(0, σ=0.05), min=0.10, max=1.0)
```

Without jitter, ISHead achieves near-zero loss on synthetic splits but generalises poorly
to AVeriTeC (continuous IS values from the heuristic rubric).

---

## Generation Pipeline ([ADR-013](adr/013-synthetic-pipeline.md))

**Command:** `just generate-synthetic` (or `just build rebuild=true`)

**Inputs:**
- `data/registry/seed_pool.jsonl` — 25 hand-curated (claim, evidence) seed pairs
- `data/raw/ai2thor/claims_all.jsonl` — real AI2THOR triples for perception/non_apprehension templates
- `configs/config.yaml` → `synthetic.n_records`, `synthetic.distribution`
- Optional `ANTHROPIC_API_KEY` for LLM-backed linguistic variation

**Output:** `data/raw/synthetic/synthetic_current.jsonl`

The generator emits records in schema v3.0 directly. EC formula values are computed
deterministically from template parameters — `derivation_method: "aggregated_from_evidence"`.

---

## Dataset Composition Target ([ADR-006](adr/006-dataset-composition-and-generation-strategy.md))

| Source | Target share | N (approx) |
|--------|-------------|------------|
| AI2THOR | 28% | ~1,800 |
| AVeriTeC | 56% | 3,568 |
| Synthetic | 16% | ~1,000 |
| **Total** | 100% | **~6,400** |

AI2THOR generation: 10 `one_hop` + 10 `conjunction` + 10 `negation` + 150 `absence` per scene × 10 scenes.
Negation/conjunction claims are doubled via corruption (supported↔refuted flip).
