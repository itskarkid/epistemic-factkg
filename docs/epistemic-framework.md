# Epistemic Framework — EpistemicFactKG

Consolidated reference for the EC formula, source trust registry, and IS cap.
Original decisions: [ADR-001](adr/001-epistemic-framework.md), [ADR-009](adr/009-source-trust-registry.md), [ADR-021](adr/021-is-cap-by-source-trust.md).

---

## Core Formula

Per-evidence epistemic confidence:

$$EC_i = 1 - (1 - ST_i)^{EW_i \times IS_i}$$

| Symbol | Meaning | Range |
|--------|---------|-------|
| ST | Source Trust — registry lookup by `source_id` | [0, 1] |
| EW | Evidence Weight — Pramana-type weight or H1 stance probability | [0, 1] |
| IS | Inference Strength — logical binding strength | [0, 1] |
| EC | Epistemic Confidence — combined reliability signal | [0, 1] |

Verdict aggregation (product-of-complements):

$$\text{SupportScore} = 1 - \prod_{i \in \text{Supports}}(1 - EC_i)$$
$$\text{RefuteScore}  = 1 - \prod_{i \in \text{Refutes}}(1 - EC_i)$$

Evidence with `not_enough_evidence` or `conflicting_evidence` stance is excluded from both aggregations.

---

## Source Trust (ST)

ST is assigned statically from `data/registry/source_trust_registry.jsonl` keyed by `source_id`.
It does not adapt per-claim. Unknown sources fall back to `DEFAULT_SOURCE_TRUST = 0.40`.

| Source Type | ST Range | Notes |
|---|---|---|
| simulation / sensor | 1.00 | AI2THOR simulator ground truth |
| academic / scientific | 0.88–0.92 | Peer-reviewed publications |
| government | 0.85–0.90 | Official government sources |
| knowledge_graph | 0.80–0.85 | Wikidata, DBpedia |
| fact_checker | 0.80–0.85 | Snopes, PolitiFact, etc. |
| news_media | 0.65–0.85 | MBFC rating mapped to range |
| ngo_org | 0.55–0.70 | NGOs, think tanks |
| web_text | 0.45–0.60 | General web sources |
| social_media | 0.30–0.40 | Twitter, Reddit, etc. — capped |
| unknown | 0.40 | Default fallback |

MBFC → ST mapping: Very High (0.90–1.0), High (0.80–0.89), Mostly Factual (0.70–0.79),
Mixed (0.55–0.69), Low/Very Low (0.30–0.54).

Web archives (web.archive.org) resolve to the original domain's ST ([ADR-020](adr/020-webarchive-source-trust-resolution.md)).

---

## Evidence Weights (EW) — Pramana Types

Used at graph-build time for `ev.ew` scalar. Multi-type items use product-of-complements:
`EW = 1 - Π(1 - wᵢ)`.

| Pramana Type | EW | Description |
|---|---|---|
| perception | 0.95 | Sensor/simulator ground truth |
| testimony | 0.80 | Web text, PDFs, citations |
| non_apprehension | 0.75 | Sensor-confirmed absence |
| comparison_analogy | 0.65 | Numeric/statistical reasoning |
| inference | 0.55 | Multi-step synthesis |
| postulation_derivation | 0.40 | Hypothetical — excluded from GNN training |

---

## Inference Strength (IS)

IS captures logical binding strength, not stance direction. A high-trust source with
an abstractive answer still gets IS = 0.6; ST captures source quality separately.

| IS | Description |
|----|-------------|
| 1.0 | Direct ground truth (simulator state, primary measurement) |
| 0.8 | One-step extractive (direct quote, official record) |
| 0.6 | Abstractive / multi-source synthesis |
| 0.4 | Partial or circumstantial |
| 0.2 | Speculative or highly hedged |
| 0.0 | Unanswerable |

**IS cap by source trust ([ADR-021](adr/021-is-cap-by-source-trust.md)):**
For sources below `ST_THRESHOLD = 0.45`, IS is capped:
`IS_final = max(0.10, min(IS_raw, ST))` if `ST < 0.45`.
This prevents low-trust sources from providing high EC regardless of evidence type.
Effect: IS mean shifts from ~0.73 → ~0.63 after applying cap.

**IS gradient detach ([ADR-022](adr/022-is-gradient-detach.md)):**
IS tensor is detached before entering the EC formula during training. Without detach,
IS drifts to verdict-optimal values (RMSE ~0.23); with detach, IS regression remains
interpretable (RMSE ~0.10).
