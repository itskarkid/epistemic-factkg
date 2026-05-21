# ADR-029: v3-nli — Claim-Aware H1 StanceHead on GNN Output (supersedes ADR-024 Part 2)

**Status:** Accepted  
**Date:** 2026-05-21  
**Supersedes (Part 2 only):** ADR-024 (NLI probs bypass H1 in EC formula)  
**Builds on:** ADR-024 Part 1 (NLI probs as input features), ADR-023 (HybridVerdictHead)

---

## Context

ADR-024 adopted a two-part design for v3-nli:

- **Part 1** (retained): append offline DeBERTa NLI probs as 3d input features to
  evidence nodes (408d total: 405d base + 3d NLI).
- **Part 2** (superseded by this ADR): bypass H1 StanceHead in the EC formula —
  read `ev.x[:, -3:]` directly and reorder columns for `SymbolicAggregator`.

Part 2 was problematic for two reasons:

1. **H1 bypass breaks generalisation across models.** All other models (baseline,
   v1-hgnn, v2-hgnn) route evidence stance through H1. Part 2 created a
   special-cased EC path that made v3-nli architecturally inconsistent and harder
   to ablate.

2. **Claim-aware ev_ctx was introduced after ADR-024.** When H1 and ISHead were
   upgraded to receive `cat([ev_emb, claim_emb[batch_ptr]])` as input — a
   claim-aware 512d context vector — H1 gained direct access to the claim
   embedding and the GNN-enriched evidence representation simultaneously.
   This makes H1 a strong enough stance classifier to replace the raw NLI bypass:
   the GNN encoder already propagates the NLI-augmented features through graph
   message-passing, and the claim context lets H1 compare evidence to the
   specific claim being verified.

**Experimental outcome:** removing the NLI bypass and running H1 on claim-aware
GNN output maintained v3-nli's advantage over v2-hgnn on held-out data, while
making the architecture consistent across all four models.

---

## Decision

**Part 2 of ADR-024 is retired.** The H1 bypass is removed.

The v3-nli EC formula path is now identical to v2-hgnn, except evidence node
features are 408d instead of 405d:

```
NLI probs (offline DeBERTa, frozen)
    → ev.x[:, -3:]  appended at graph-build time
    → EpistemicEncoder (GNN: 408d input, 256d output)
    → cat([ev_emb, claim_emb[batch_ptr]])  [N_ev, 512d]
    → H1 StanceHead  → stance logits [N_ev, 3]   (claim-aware, GNN-enriched)
    → IS Head        → IS scalar [N_ev, 1]
    → SymbolicAggregator (same EC formula as v1-hgnn/v2-hgnn)
    → HybridVerdictHead
```

Evidence node feature layout (v2 = 408d):
```
[text_emb(384) | modality(8) | evidence_type(5) | source_type(6) | reasoning(6)]
= 405d   (GraphConfig.v1)
+ [p_contradiction | p_entailment | p_neutral]   ← DeBERTa-v3-small MNLI
= 408d   (GraphConfig.v2)
```

Note: The base increased from 400d to 405d after the feature audit (modality 5→8,
source_type 3→6, reasoning 0→6 added). ADR-024 references to "400d" and "403d"
are corrected here to 405d and 408d respectively.

---

## Consequences

- `NLIHybridHGNN._soft_verdict_logits` no longer reads `ev.x[:, -3:]` directly.
  It uses `stance_logits` from H1 (same as `EpistemicHGNN` and `HybridHGNN`).
- The `nli_probs` field in `build_prediction_payload` still reads `ev.x[:, -3:]`
  for **display only** in the app's evidence breakdown table — not for EC computation.
- `GraphConfig.v2()` encodes the 408d evidence dimension; all pipeline components
  select it automatically when `model == "v3-nli"`.
- v3-nli and v2-hgnn now share the same EC formula path, making ablation clean:
  the only structural difference is evidence input dimensionality (408d vs 405d).
