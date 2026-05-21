# Project Plan: Epistemic FactKG — Neuro-Symbolic GNN

**Status:** Active — V1 implementation in progress  
**Supersedes:** `docs/superseded/v0/project-plan.md` (7-phase plan, Phase 4 shortcut discovery)  
**Motivation:** `docs/experiments/01_shortcut_discovery.md`

---

## Research Question

Can a GNN trained on Pramana-categorized evidence improve fact verification accuracy beyond text-only baselines — when the model is *forced* to learn epistemic reliability rather than routing on stance labels?

The original plan answered this question with a failed experiment: stance-typed edges encoded the verdict directly (100% accuracy from epoch 1, no learning). The new architecture removes that structural shortcut.

---

## Core Architecture: Neuro-Symbolic Hybrid

The V1 architecture separates learning from reasoning:

```
v3.0 Training Data (AI2THOR + AVeriTeC + Synthetic)
        ↓
  Shared HeteroConv Encoder   ← learns: text semantics, evidence types,
        ↓                               reasoning strategy, source category
   ┌────┴────┐
   H1        H2
 Stance    IS Regression
 (3-class)   (scalar 0–1)
   ↓          ↓
ground-truth  ground-truth IS
stance labels (per evidence, v3.0)
   ↓
Symbolic Aggregation (no trainable parameters)
EC_i = 1-(1-ST_i)^(EW_i × IS_i)
SupportScore, RefuteScore
   ↓
Rule-based Verdict
```

**What the encoder is forbidden from knowing:** stance labels (removed from edge types), verdict (no verdict gradient). It must learn stance from text and epistemic features.

**Key design decisions:**
- No stance edges — replaced by neutral `connected_to` reverse edge
- Verdict emerges from symbolic EC formula, not from a learned classifier
- Training loss: `stance_loss + λ × is_loss` only (λ = 0.5 default)
- Config-driven (`GraphConfig`) — V1 → V2 is a config change, not a model rewrite

---

## V3.0 Dataset (Three Sources)

| Source | Count | Role |
|---|---|---|
| AI2THOR | ~1,800 | Perception + non_apprehension ground truth via simulator |
| AVeriTeC | 3,568 | Real-world web fact-checking target domain |
| Synthetic | ~1,000 | Shortcut-breaking: same stance → different verdict based on EC reliability |

Shortcut-breaking fraction ≥ 35% enforced by `SyntheticDataValidator` (ADR-012).  
`postulation_derivation` excluded from GNN training (ADR-005).  
3-class verdict: supported / refuted / not_enough_evidence (ADR-007).

---

## Implementation Phases

### Phase 0 — Regenerate V3.0 Training Data

Run `just synthetic && just build && just filter && just check-train`.

Verify every evidence item has `inference_strength` (float), `evidence_types` (list), `source_id` (string). The unified JSONL must be schema v3.0 before any graph building.

`data/raw/ai2thor/claims_all.jsonl` does not need rebuilding — it is frozen raw simulator output; the adapter adds v3.0 fields at conversion time.

### Phase 1 — Graph Builder (no shortcuts)

**Files:** `src/core/gnn/graph_builder.py`, `src/core/gnn/types.py`, `src/core/gnn/featurizer.py`, `src/core/claims/labels.py`

Remove stance-typed edges. Build:
- Claim nodes: 390-d (384 text + 6 reasoning-strategy one-hot)
- Evidence nodes: 400-d (384 text + 5 modality + 5 evidence_types multi-hot + 6 source_type one-hot)
- Edge types: `has_evidence`, `connected_to` (neutral reverse), `co_evidence` (evidence↔evidence), `has_triple`, `from_triple`
- Labels on evidence nodes: `stance_y` (int), `is_y` (float), `ew` (float), `st` (float) — last two as separate tensors, NOT encoder input

**Add to `labels.py`:** `ReasoningStrategy` enum with 6 values unified across AI2THOR, AVeriTeC, and synthetic.

**Add to `featurizer.py`:** `encode_evidence_types()` (5-d), `encode_source_type()` (6-d), `encode_reasoning_strategy()` (6-d).

**Checkpoint:** Print one `HeteroData`. Check node types (claim, evidence, triple — no epistemic), edge types (5 total), `.stance_y` and `.is_y` tensors present, claim dim=390, evidence dim=400.

### Phase 2 — Multi-Head Model

**New files:** `src/core/gnn/config.py`, `src/core/gnn/encoder.py`, `src/core/gnn/heads.py`, `src/core/gnn/aggregator.py`  
**Rewrite:** `src/core/gnn/model.py`

- `GraphConfig` — single source of truth for node dims and edge types; `EpistemicEncoder` builds HeteroConv from it dynamically (never hardcodes node/edge names)
- H1 (`StanceHead`): `Linear(256→128)→ReLU→Linear(128→3)`, supervised on `stance_y`
- H2 (`ISHead`): `Linear(256→128)→ReLU→Linear(128→1)→Sigmoid`, supervised on `is_y`
- `SymbolicAggregator`: stateless EC formula + SupportScore/RefuteScore; reuses `compute_evidence_confidence()` from `labels.py`
- `EpistemicHGNN`: thin assembler of the above; `forward()` returns `stance_logits`, `is_pred`, `support_score`, `refute_score`; `get_verdict()` applies rule-based thresholds

**Checkpoint:** Single forward pass. Verify output shapes and verdict string output.

### Phase 3 — Multi-Task Training

**Files:** `src/core/gnn/train.py`, `src/cli/train_gnn.py`

Loss: `total = CrossEntropyLoss(stance_logits, stance_y) + λ × MSELoss(is_pred, is_y)`.  
No verdict loss. `--is-loss-weight` (default 0.5) controls λ.  
`train_gnn.py` is already scaffolded (clean argparse, `NotImplementedError` placeholder).

**Checkpoint:** 5 epochs on small subset. Both `stance_loss` and `is_loss` tracked; total loss decreasing.

### Phase 4 — Evaluation

**Files:** `src/cli/evaluate_gnn.py`, `src/core/gnn/metrics.py`

Add `compute_rmse()` and `compute_pearson_r()` to `metrics.py`.

Output per run to `out/results/<run>/`:
- `stance_metrics.json` — per-evidence stance accuracy + per-class breakdown
- `is_metrics.json` — IS RMSE, Pearson R
- `verdict_metrics.json` — macro F1, accuracy, confusion matrix (same format as v0 `metrics.json`)

**Success criteria:** H1 stance accuracy > 80%, H2 IS RMSE < 0.20, verdict accuracy ≥ v0 baseline.

---

## V1 → V2 Extension Path

V2 adds source nodes as a separate node type (so the encoder learns cross-claim source patterns). Because `EpistemicEncoder` reads `GraphConfig` dynamically, the upgrade requires:
- 2 config lines (add source dim, add edge types)
- Graph builder additions (source node construction)
- No model code changes

See the detailed I/O specification and extensibility design in the implementation plan file (`.claude/plans/`).

---

## Key ADRs

| ADR | Topic |
|---|---|
| ADR-001 | Pramana epistemic framework — 6 categories, weights, EC formula |
| ADR-002 | Ports and adapters architecture |
| ADR-003 | Floorplan-based train/val/test split |
| ADR-005 | Exclude postulation_derivation from GNN training |
| ADR-006 | Dataset composition (28% AI2THOR / 56% AVeriTeC / 16% synthetic) |
| ADR-007 | 3-class verdict (drop conflicting_evidence) |
| ADR-008 | Evaluation protocol — macro F1 primary, per-Pramana breakdown |
| ADR-009 | Source trust registry |
| ADR-010 | Per-evidence epistemic modeling — EC formula, IS rubric |
| ADR-011 | Evidence labeling rules (AI2THOR strategy→types, AVeriTeC 4-pass) |
| ADR-012 | Shortcut leakage rationale and synthetic data strategy |
| ADR-013 | Synthetic pipeline — 15 templates, pluggable text clients, seed pool |

---

## Justfile Targets

| Target | Status |
|---|---|
| `just build` | Valid — runs all adapters, outputs v3.0 unified JSONL |
| `just synthetic` | Valid — generates shortcut-breaking synthetic batch |
| `just validate` | Valid |
| `just filter` | Valid — excludes postulation_derivation and conflicting_evidence |
| `just check-train` | Valid |
| `just split` | Valid |
| `just graph` | **Stale** — rewrite in Phase 1 |
| `just train` | **Stale (Phase 3 placeholder)** — currently raises NotImplementedError |
| `just eval` | **Stale** — rewrite in Phase 4 |
