# ADR-027: Remove _EC_NEI_MAX — Full Delegation to VerdictHead

**Status:** Accepted  
**Date:** 2026-05-18  
**Builds on:** ADR-014 (VerdictHead learned thresholds)

---

## Context

The `predict()` method in all three HGNN models computed a symbolic verdict using
a three-branch decision tree over the EC aggregation scores:

```python
_EC_DECISIVE = 0.35
_EC_NEI_MAX  = 0.20   # ← REMOVED

if ref > sup and ref > _EC_DECISIVE:
    verdict = "refuted"
elif sup > ref and sup > _EC_DECISIVE:
    verdict = "supported"
elif max(sup, ref) < _EC_NEI_MAX:      # ← REMOVED
    verdict = "not_enough_evidence"
else:
    verdict = VerdictHead.argmax(verdict_logits)
```

The third branch (`_EC_NEI_MAX = 0.20`) forced a `not_enough_evidence` verdict
whenever both EC scores were low, **bypassing the VerdictHead entirely** for these
cases.

This created a **train-inference gap**:

- **During training:** soft EC scores below any threshold still pass through the
  VerdictHead (via the differentiable soft aggregation path). The VerdictHead
  receives gradient supervision on these ambiguous claims and learns to predict
  `supported`, `refuted`, or `not_enough_evidence` based on all available signal.

- **During inference:** the same low-score claims hit the `_EC_NEI_MAX` branch and
  are assigned `not_enough_evidence` unconditionally, contradicting what the VerdictHead
  would predict.

The VerdictHead was trained precisely to handle ambiguous evidence — claims where
neither strong support nor strong refutation is present. Forcing NEI for these cases
discards the trained calibration.

**Example:** A claim with `sup=0.18, ref=0.12` (AVeriTeC, human-annotated
"supported" — weak evidence from mixed-trust sources) was forced to NEI by the
`_EC_NEI_MAX` gate, even though the VerdictHead, having seen thousands of such
cases during training, would predict "supported".

---

## Decision

Remove the `_EC_NEI_MAX` branch from `predict()` in all three models
(`EpistemicHGNN`, `HybridHGNN`, `NLIHybridHGNN`).

The two remaining branches are retained: they represent **EC-decisive** cases where
the symbolic formula provides a clear signal (either score > 0.35 with clear winner).
For all other cases, the VerdictHead decides.

```python
_EC_DECISIVE = 0.35

if ref > sup and ref > _EC_DECISIVE:
    verdict = "refuted"
elif sup > ref and sup > _EC_DECISIVE:
    verdict = "supported"
else:
    # EC is ambiguous — let the trained VerdictHead decide.
    # Covers genuine NEI, conflicting evidence, and weak-evidence cases.
    verdict = VerdictHead.argmax(verdict_logits)
```

The two EC-decisive branches are consistent with the VerdictHead's expected
predictions for those cases (the VerdictHead was trained on the same EC scores),
so no discrepancy is introduced.

---

## Why the EC-decisive threshold mechanism is kept

The EC-decisive branches are retained for two reasons:

1. **Interpretability:** When the symbolic aggregation clearly supports or clearly
   refutes a claim, the verdict should reflect that — not be overridden by the
   VerdictHead's learned calibration. This preserves the epistemic semantics.

2. **Training alignment:** The VerdictHead is supervised on the same EC formula
   output. For claims where EC clearly signals one verdict, the VerdictHead agrees.
   Keeping the branch eliminates no real information — it just makes the decision
   path explicit.

The `_EC_NEI_MAX` branch lacked this alignment: the VerdictHead was **not** trained
to agree with "force NEI when both are low". It was trained to predict the true label
for those claims.

**Note — the threshold value is dynamic, not a fixed constant:**  
`_EC_DECISIVE` is replaced in code by `self.ec_threshold`, which is Optuna-tuned
(search range 0.20–0.60, step 0.05), saved into each model checkpoint, and loaded
at eval/inference time from the checkpoint. The value varies per model:
- v3-nli: 0.25 (Optuna-tuned)
- v2-hgnn: 0.30 (Optuna-tuned)
- Default fallback (pre-hparam-search): 0.35

The *mechanism* (symbolic override when EC is decisive) is the permanent design
decision. The specific threshold is a hyperparameter.

---

## Alternatives Considered

**A. Lower _EC_NEI_MAX to a smaller value (e.g. 0.05):** Reduces cases forced to NEI
but does not eliminate the gap. Any hard threshold creates some train-inference
inconsistency. Rejected in favour of full removal.

**B. Keep _EC_NEI_MAX but re-train with matching inference logic:** Add an explicit
"if both low → NEI" rule during training so the VerdictHead is never asked to handle
these cases. Complicates the training pipeline and reduces supervision for the
hardest AVeriTeC cases. Rejected.

**C. Replace both EC-decisive branches with VerdictHead-only:** Full delegation
to VerdictHead for all cases. Reduces interpretability (the EC signal is
post-hoc in such an architecture). The two remaining branches are cheap to keep and
provide symbolic grounding for the most confident predictions. Deferred.

---

## Consequences

- All three HGNN models use the same updated predict() logic. Behavior change
  is observable only for claims where `max(sup, ref) < 0.20` was previously true —
  these now route to VerdictHead instead of being forced to NEI.
- No retraining required — the VerdictHead was already trained to handle these claims.
  The fix removes an inference-time override, not a training-time bug.
- The `_EC_NEI_MAX` constant is deleted from all three model files. Any re-introduction
  would need to align with ADR-014's principle: learned calibration over symbolic thresholds.
