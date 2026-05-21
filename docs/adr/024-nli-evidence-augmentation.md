# ADR-024: NLIHybridHGNN (v3-nli) — NLI Probs as Features + Direct EC Signal

**Status:** Partially superseded — Part 1 (NLI as input features, 408d) retained; Part 2 (H1 bypass) superseded by ADR-029  
**Date:** 2026-05-17  
**Builds on:** ADR-023 (HybridHGNN v2-hgnn)

> **Note:** Part 2 of this ADR (NLI probs bypass H1 in the EC formula) is no longer
> the implementation. H1 StanceHead now runs on claim-aware GNN output
> `cat([ev_emb, claim_emb])`, making the NLI bypass unnecessary. See ADR-029.

---

## Context

H1 (StanceHead) predicts evidence stance from `ev_emb` — the GNN-encoded evidence
node embedding. In v2-hgnn, `ev_emb` only receives claim information indirectly
through graph message-passing over the `has_evidence` / `connected_to` edges.
This indirect path is insufficient for arbitrary natural-language stance detection:
H1 was trained on AVeriTeC/AI2THOR/synthetic text and does not generalise to
unseen domain pairs.

**Failure mode observed:** claim "Apple is red" + evidence "Apple is yellow" →
v2-hgnn predicts SUPPORTED, because the high lexical similarity (both about
apples and colours) dominates the GNN encoding and H1 cannot detect the
semantic contradiction.

A pre-trained NLI cross-encoder directly compares a (claim, evidence) text pair
and produces calibrated stance probabilities (contradiction / entailment / neutral),
generalising to any domain. The question is how to incorporate this signal.

**Options evaluated:**

| Option | Description | Verdict |
|--------|-------------|---------|
| A | NLI probs directly in EC formula (bypass H1) | Loses H1 stance supervision signal |
| B | Append NLI probs as input features only (403d) | H1 still mediates EC — doesn't generalise |
| C | Train H1 jointly with NLI cross-encoder (fine-tune) | Adds 184M parameters; out of scope |
| **A+B** | **NLI probs as features AND directly in EC formula** | **Adopted — see below** |

Option B was implemented first. Evaluation revealed it was insufficient: H1 never
learned to map NLI contradiction signal to a refutes prediction because the training
data does not cover the contradiction patterns the NLI model detects (e.g. simple
property-value contradictions like "apple is red / apple is yellow"). The NLI signal
was being diluted through the GNN's 403d→256d projection before reaching H1.

**Root cause confirmed experimentally:** the NLI cross-encoder outputs
`p_contradiction ≈ 0.998` for ("An apple is red", "An apple is yellow"), yet
v3-nli (Option B only) still predicted SUPPORTED. The NLI signal was present in
the features but H1 had not learned to use it for the EC formula.

Final decision: combine A+B — NLI probs are appended to ev_features (for the GNN
encoder) AND fed directly to the EC formula in `_soft_verdict_logits`, bypassing H1.

---

## Decision

Two changes together constitute v3-nli:

**Part 1 — NLI probs as evidence node features (403d):**

```
evidence features: [text_emb(384) | modality(8) | evidence_type(5) | source_type(3)]
                   = 400d   (GraphConfig.v1)

               + [p_contradiction | p_entailment | p_neutral]   ← NLI cross-encoder
                   = 403d   (GraphConfig.v2)
```

The NLI probs are stored in `ev.x[:, -3:]` at graph-build time. The GNN encoder
can learn to propagate this signal, but this alone is insufficient (see context above).

**Part 2 — NLI probs bypass H1 in the EC formula:**

`NLIHybridHGNN` overrides `_soft_verdict_logits` to read `ev.x[:, -3:]` directly
and feed it to `SymbolicAggregator`, skipping H1's stance prediction entirely for
the verdict pathway. The MNLI label order `[contradiction, entailment, neutral]` is
reordered to match the EC formula's convention `[supports, refutes, neutral]`:

```python
# NLIHybridHGNN._soft_verdict_logits
nli_probs    = ev.x[:, -3:]             # [p_contradiction, p_entailment, p_neutral]
stance_probs = nli_probs[:, [1, 0, 2]] # [p_supports, p_refutes, p_neutral]
# → SymbolicAggregator.compute_soft_scores(stance_probs, is_pred, ew, st)
```

H1 (`StanceHead`) still trains on its own stance CE loss and still influences the
encoder through its gradient path. It is no longer consulted for the EC formula.

**Gradient paths in v3-nli:**
```
verdict_CE → HybridVerdictHead → claim_emb   → encoder ✓  (rich 256d path)
verdict_CE → HybridVerdictHead → EC scores   → NLI probs  (frozen, no grad) —
IS_MSE     → ISHead            → encoder     ✓  (clean, detached)
stance_CE  → StanceHead        → encoder     ✓  (H1 still supervised)
```

**Academic framing:**
> "We augment evidence node representations with stance probabilities from a frozen
> NLI cross-encoder and route these probabilities directly through the EC formula,
> providing a generalised claim-evidence comparison signal that is independent of
> H1's training distribution."

---

## Alternatives Considered

**Features only (Option B alone):**
Implemented first. Insufficient — H1 does not learn to use the NLI contradiction
signal for the EC formula because the training distribution lacks the contradiction
patterns the NLI model detects. The 3 NLI dimensions are diluted in the 403d→256d
GNN projection. Superseded by the A+B combination.

**Fine-tune the NLI cross-encoder jointly:**
Adds 184M trainable parameters and significant compute. The frozen cross-encoder
already produces well-calibrated probs on out-of-domain text. Deferred as future work.

**C. Stance-typed edges instead of NLI features:**
Replace the single `has_evidence` edge type with three typed edges (supports /
refutes / neutral) derived from NLI probs. This changes the graph topology and
requires re-designing the HeteroConv message-passing. The feature-augmentation
approach achieves the same signal more simply. Deferred as future work.

---

## Results (test set, n=657 scored claims; 109 skipped — no evidence after filtering)

Final architecture: A+B combined (NLI probs as features + NLI probs directly in EC formula),
with AVeriTeC Q+A pre-processing (ADR-025), encoder residuals + windowed co-evidence (ADR-026),
and full VerdictHead delegation (ADR-027).

| Model    | Verdict Acc | Macro F1 | IS RMSE | averitec | synthetic | ai2thor |
|----------|-------------|----------|---------|----------|-----------|---------|
| baseline | 0.8158      | 0.8166   | 0.0981  | 0.649    | 1.000     | 0.929   |
| v1-hgnn  | 0.7047      | 0.6883   | 0.0966  | 0.503    | 0.898     | 0.929   |
| v2-hgnn  | 0.7412      | 0.7451   | 0.0959  | 0.592    | 0.889     | 0.894   |
| **v3-nli** | **0.7930** | **0.7903** | **0.0947** | **0.674** | 0.881 | **1.000** |

Key findings:
- v3-nli achieves 100% on AI2THOR — NLI contradiction signal directly identifies absence-refuted
  claims that textual similarity alone cannot distinguish.
- v3-nli leads on AVeriTeC (67.4%) — Q+A pre-processing (ADR-025) provides the largest
  single improvement (+12pp on AVeriTeC).
- baseline dominates synthetic (100%) — synthetic data has perfect EC signal by construction;
  the learned components add noise on patterns the EC formula already handles exactly.
- v1-hgnn underperforms baseline (70.5% vs 81.6%) — the 2D EC bottleneck without claim
  embedding context is insufficient; see ADR-023 for the diagnosis.

---

## Consequences

- v3-nli is the **primary model** for publication; v2-hgnn, v1-hgnn, and baseline
  are ablations in the model-evolution table.
- Graph build time increases by ~0.3s per claim (NLI inference on CPU for N_ev pairs).
  The NLI model is loaded lazily and cached for the full dataset build.
- A separate graph dataset file (`graph_dataset_nli.pt`) is required; the 400d
  dataset (`graph_dataset.pt`) is unchanged and still used for v1-hgnn / v2-hgnn.
- At inference time (demo app), the NLI cross-encoder runs per request for v3-nli.
  Latency is acceptable for a demo (~0.2s per evidence item on CPU).
- `GraphConfig.v2()` encodes the 403d evidence dimension; all pipeline components
  (`train.py`, `evaluate.py`, `predictor.py`) select `v2()` automatically when
  `model == "v3-nli"`.
