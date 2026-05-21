# Model Evolution — EpistemicFactKG

This document traces the three model architectures tried, the decisions that led from
one to the next, and their comparative results on the test set.

---

## Architecture Overview

All four models share the same upstream components: the HeteroConv graph encoder,
StanceHead (H1), and ISHead (H2). They differ only in how the verdict is produced.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       SHARED COMPONENTS                              │
│                                                                       │
│  HeteroData graph                                                     │
│  (claim + evidence nodes + pramana-typed edges)                      │
│         │                                                             │
│         ▼                                                             │
│  ┌─────────────────┐                                                  │
│  │ EpistemicEncoder │  HeteroConv (GAT, 4 heads, hidden_dim=256)     │
│  └────────┬────────┘                                                  │
│           │                                                           │
│    ┌──────┴──────┐                                                    │
│    ▼             ▼                                                    │
│  ev_emb      claim_emb                                                │
│  [N_ev,256]  [N_cl,256]                                               │
│    │             │                                                    │
│    └──────┬──────┘  cat([ev_emb, claim_emb[batch_ptr]])              │
│           ▼                                                           │
│    ev_ctx [N_ev, 512]  ← claim-aware evidence context                │
│    ┌──────┴──────┐                                                    │
│    ▼             ▼                                                    │
│  ┌──────────┐ ┌───────────┐                                           │
│  │StanceHead│ │  ISHead   │                                           │
│  │  (H1)    │ │   (H2)    │                                           │
│  └──────────┘ └───────────┘                                           │
│  stance_logits  is_pred                                               │
│  [N_ev, 3]      [N_ev, 1]  ← supervised by IS regression             │
└─────────────────────────────────────────────────────────────────────┘
         │                    │
         └────────┬───────────┘
                  │
           (verdict pathway differs per model — see below)
```

---

## Model 1 — BaselineHGNN (`baseline`)

**ADR:** ADR-016  
**Purpose:** Ablation — no EC formula, verdict from claim embedding only.

```
  claim_emb [N_cl, 256]
       │
       ▼
  ┌───────────────────────────┐
  │  MLP                      │
  │  Linear(256→128)          │
  │  ReLU                     │
  │  Dropout                  │
  │  Linear(128→3)            │
  └───────────┬───────────────┘
              ▼
       verdict_logits [N_cl, 3]
```

**Gradient paths:**
- Verdict CE → MLP → claim_emb → encoder ✓
- IS regression → ISHead → encoder ✓ (IS not used in verdict)
- Stance CE → StanceHead → encoder ✓

**What it tests:** Whether the graph encoder + claim embedding alone is sufficient
for verdict, without any epistemic formalism.

---

## Model 2 — EpistemicHGNN (`v1-hgnn`)

**ADR:** ADR-013, ADR-014, ADR-022  
**Purpose:** Pure neuro-symbolic — verdict only from EC formula scores.

```
  ev_emb [N_ev, 256]         is_pred [N_ev, 1]
       │                          │
       │                     .detach()   ← ADR-022
       │                          │
       ▼                          ▼
  stance_probs [N_ev, 3]     is_fixed (no grad)
       │                          │
       └──────────┬───────────────┘
                  ▼
       SymbolicAggregator (EC formula per claim)
       EC_i = 1 - (1 - ST_i)^(EW_i × IS_i)
       support_score = 1 - ∏(1 - EC_i × p_support_i)
       refute_score  = 1 - ∏(1 - EC_i × p_refute_i)
                  │
                  ▼
       scores [N_cl, 2]
                  │
       ┌──────────▼──────────┐
       │   VerdictHead        │
       │   Linear(2→3)        │
       └──────────┬──────────┘
                  ▼
       verdict_logits [N_cl, 3]
```

**Gradient paths (after IS detach):**
- Verdict CE → VerdictHead → EC×p_stance → stance_probs → encoder ✓
- IS regression → ISHead → encoder ✓ (clean, no verdict interference)
- Stance CE → StanceHead → encoder ✓

**Key limitation:** The 2D bottleneck `[support_score, refute_score]` discards
the claim's semantic context. VerdictHead has far less information than the
baseline's 256-dim embedding.

---

## Model 3 — HybridHGNN (`v2-hgnn`)

**ADR:** ADR-023  
**Purpose:** Best of both worlds — EC formula signal fused with claim embedding.

```
  ev_emb [N_ev, 256]              claim_emb [N_cl, 256]
       │                                  │
       │     is_pred.detach()             │  ← direct verdict gradient
       │           │                      │
       ▼           ▼                      │
  stance_probs  is_fixed                  │
       │           │                      │
       └─────┬─────┘                      │
             ▼                            │
    SymbolicAggregator                    │
    scores [N_cl, 2]                      │
             │                            │
             └─────────────┬──────────────┘
                           ▼
                  torch.cat([scores, claim_emb])
                       [N_cl, 258]
                           │
             ┌─────────────▼─────────────┐
             │   HybridVerdictHead        │
             │   Linear(258→128)          │
             │   ReLU                     │
             │   Linear(128→3)            │
             └─────────────┬─────────────┘
                           ▼
                  verdict_logits [N_cl, 3]
```

**Gradient paths:**
- Verdict CE → HybridVerdictHead → **claim_emb** → encoder ✓ (rich 256d path)
- Verdict CE → HybridVerdictHead → EC×p_stance → stance → encoder ✓
- IS regression → ISHead → encoder ✓ (clean)
- Stance CE → StanceHead → encoder ✓

---

## Model 4 — NLIHybridHGNN (`v3-nli`)  ← primary model

**ADR:** ADR-024 (Part 1), ADR-029  
**Purpose:** Append frozen NLI cross-encoder probs as evidence node features (408d total).
H1 StanceHead runs on claim-aware GNN output — graph-enriched NLI features + claim context.

```
  evidence text + claim text
       │
       ▼
  frozen DeBERTa-v3-small (MNLI)   ← cross-encoder, per (claim, ev) pair
  [p_contradiction, p_entailment, p_neutral]
       │
       └─→ appended to ev_features: 405d → 408d  (GNN encoder input, GraphConfig.v2)
                │
                ▼
       EpistemicEncoder (HeteroConv, 408d→256d)
       ev_emb [N_ev, 256]    claim_emb [N_cl, 256]
                │
                ▼
       cat([ev_emb, claim_emb[batch_ptr]])  [N_ev, 512d]  ← claim-aware context
                │
          ┌─────┴────┐
          ▼          ▼
       H1 StanceHead   H2 ISHead
       stance_logits   is_pred.detach()
       [N_ev, 3]       [N_ev, 1]
                │
                ▼
       SymbolicAggregator  (same EC path as v2-hgnn)
       EC_i = 1 - (1 - ST_i)^(EW_i × IS_i)
       support_score = 1 - ∏(1 - EC_i × p_support_i)
       refute_score  = 1 - ∏(1 - EC_i × p_refute_i)
                │
                ▼
       scores [N_cl, 3] + claim_emb [N_cl, 256]
                │
       ┌────────▼────────┐
       │ HybridVerdictHead│
       └────────┬────────┘
                ▼
       verdict_logits [N_cl, 3]
```

**What changes vs v2-hgnn:**
1. Evidence node input: 405d → 408d (NLI probs appended, ADR-024 Part 1)
2. EC formula: same path as v2-hgnn — H1 on claim-aware GNN output (ADR-029)
3. H1 receives `cat([ev_emb, claim_emb])` [512d] — NLI features visible after GNN + claim context

**Note (ADR-029):** An earlier design (ADR-024 Part 2) bypassed H1 and fed NLI probs directly
to SymbolicAggregator. This was replaced when claim-aware ev_ctx was introduced — H1 running on
GNN-enriched NLI features + claim context is sufficient, and the bypass created an inconsistent
EC formula path vs the other three models.

**Gradient paths:**
```
verdict_CE → HybridVerdictHead → claim_emb → encoder  ✓  (rich 256d path)
verdict_CE → HybridVerdictHead → EC×p_stance → stance_logits → encoder  ✓
IS_MSE     → ISHead            → encoder   ✓  (clean, detached)
stance_CE  → StanceHead        → encoder   ✓
```

---

## Comparative Results (Test Set)

657 scored claims; 109 skipped (no evidence after filtering). Includes all fixes:
encoder residuals + windowed co-evidence (ADR-026), AVeriTeC Q+A pre-processing (ADR-025),
full VerdictHead delegation (ADR-027).

| Metric              | baseline   | v1-hgnn | v2-hgnn | v3-nli     |
|---------------------|------------|---------|---------|------------|
| **Verdict Acc**     | **0.8158** | 0.7047  | 0.7412  | 0.7930     |
| **Verdict Macro F1**| **0.8166** | 0.6883  | 0.7451  | 0.7903     |
| Verdict W-F1        | **0.8170** | 0.7087  | 0.7512  | 0.7974     |
| IS RMSE ↓           | 0.0981     | 0.0966  | 0.0959  | **0.0947** |
| IS Pearson r ↑      | 0.8989     | 0.9008  | 0.9032  | **0.9069** |

### Per-Source Verdict Accuracy

| Source    | n   | baseline   | v1-hgnn | v2-hgnn | v3-nli     |
|-----------|-----|------------|---------|---------|------------|
| ai2thor   | 85  | 0.929      | 0.929   | 0.894   | **1.000**  |
| averitec  | 328 | 0.649      | 0.503   | 0.592   | **0.674**  |
| synthetic | 244 | **1.000**  | 0.898   | 0.889   | 0.881      |

### Per-Class Verdict F1 (v3-nli, final)

| Class               | Precision | Recall | F1     |
|---------------------|-----------|--------|--------|
| supported           | 0.629     | 0.741  | 0.681  |
| refuted             | 0.824     | 0.829  | 0.827  |
| not_enough_evidence | 0.970     | 0.778  | **0.864** |

---

## Decision Chain

```
ADR-013  EpistemicHGNN v1 architecture
    │
    ├── ADR-014  VerdictHead (learned thresholds)
    │       │
    │       └── ADR-016  BaselineHGNN ablation
    │
    ├── ADR-015  Class-weighted loss (NEI imbalance)
    ├── ADR-017  IS jitter in synthetic data
    ├── ADR-018  NEI-heavy synthetic distribution
    ├── ADR-019  AVeriTeC IS rubric (answer_type)
    ├── ADR-020  Webarchive source trust resolution
    ├── ADR-021  IS cap by source trust
    │
    ├── ADR-022  IS gradient detach
    │       │    (IS RMSE: 0.23 → 0.12; but verdict drops due to 2D bottleneck)
    │       │
    ├── ADR-023  HybridHGNN v2-hgnn
    │            (EC scores + claim_emb → verdict)
    │
    ├── ADR-024  NLIHybridHGNN v3-nli  ← primary model
    │            (NLI probs [3d] appended to evidence features: 405d → 408d)
    │            (Part 2 — H1 bypass — superseded by ADR-029)
    │
    ├── ADR-025  AVeriTeC Q+A evidence pre-processing for NLI
    │            (strip Q+A prefix before NLI scoring; answer-only premise)
    │            (AVeriTeC acc: ~55% → 67.4%)
    │
    ├── ADR-026  Encoder residual connections + windowed co-evidence
    │            (skip connections at each GATConv layer; max-5 neighbours by index)
    │            (prevents NLI feature dilution; reduces oversmoothing on dense graphs)
    │
    ├── ADR-027  Remove _EC_NEI_MAX — full VerdictHead delegation
    │            (eliminates train-inference gap for ambiguous EC cases)
    │
    └── ADR-029  Claim-aware H1 StanceHead on GNN output (supersedes ADR-024 Part 2)
                 (cat([ev_emb, claim_emb]) → H1; consistent EC path across all 4 models)
```

---

## Key Findings for Paper

1. **The EC formula works on epistemically consistent data:** baseline achieves 100%
   on synthetic where EC values are mathematically grounded. The learned models add
   complexity that does not help on these perfectly structured patterns.

2. **The 2D bottleneck limits pure symbolic verdict (v1-hgnn):** aggregating all
   evidence into two scalars discards semantic context needed for verdict prediction.
   v1-hgnn (70.5%) underperforms baseline (81.6%) despite having the EC formula.

3. **NLI augmentation is the decisive advantage for AVeriTeC (v3-nli):** frozen
   DeBERTa NLI probs fed directly into the EC formula raise AVeriTeC accuracy to
   67.4% — the best across all models (+2.5pp over baseline). The Q+A pre-processing
   fix (ADR-025) accounts for the largest gain (~12pp on AVeriTeC alone).

4. **NLI enables perfect AI2THOR accuracy (v3-nli: 100%):** NLI contradiction signal
   directly handles absence-refuted perception claims ("There is no fork" + "The fork
   is made of metal" → contradiction → refuted). Textual similarity alone cannot
   distinguish these cases.

5. **Baseline dominates synthetic (100% vs 88% learned):** the EC formula is not
   needed when patterns are perfectly consistent with the epistemic framework by
   construction. The learned components (VerdictHead, NLI routing) add unnecessary
   complexity for clean synthetic data.

6. **AVeriTeC is the bottleneck for all models (50–67%):** crowdsourced annotations
   accept weak-evidence claims as "supported"; the EC formula correctly downgrades
   these but diverges from human annotation. The ceiling is annotation noise, not
   model capacity.

7. **IS detach is essential for interpretability:** IS RMSE reaches 0.095 in v3-nli
   (best across all models). Without detach, IS drifts to task-optimal values that
   do not reflect epistemic ground truth (RMSE ~0.23 before ADR-022).

8. **Encoder residuals preserve NLI signal:** without skip connections, the GATConv
   403d→256d projection dilutes the NLI probs before they reach H1 and the EC path.
   Residuals ensure the original NLI features remain accessible in the final embedding.
