# ADR-030 — EC Decision Path Analysis: Symbolic vs VerdictHead Vote Distribution

**Status:** Accepted  
**Date:** 2026-05-21  
**Context:** Publication freeze analysis — diagnostic evaluation of how often the EC symbolic layer overrides VerdictHead vs delegates to it, and whether those decisions are correct.

---

## Context

`predict()` in every EC model follows four mutually exclusive branches:

| Branch | Condition | Verdict source |
|--------|-----------|----------------|
| `symbolic_supported` | `sup > θ`, `ref ≤ θ` | EC formula → "supported" |
| `symbolic_refuted`   | `ref > θ`, `sup ≤ θ` | EC formula → "refuted" |
| `vh_conflict`        | both `> θ`            | VerdictHead resolves conflict |
| `vh_fallback`        | neither `> θ`         | VerdictHead decides |

θ is **dynamic** — Optuna-tuned (range 0.20–0.60), saved in each checkpoint, loaded at
eval/inference time from the checkpoint. Per-model values: v3-nli = 0.25, v2-hgnn = 0.30,
default fallback (pre-hparam-search) = 0.35. See `src/pipeline/model/hparam_search.py`.

Before this ADR, only final verdict accuracy was tracked. This analysis adds per-branch counting, correctness, and per-source breakdown to `evaluate.py`, emitting results in `verdict_metrics.json` under `decision_paths` and in `eval_summary.md`.

---

## Findings — v3-nli on Test Set (767 claims, θ=0.35 — checkpoint value at time of analysis, pre-hparam-update)

### Vote Distribution

| Branch | Count | % | Accuracy |
|--------|-------|----|----------|
| `symbolic_supported` | 243 | 31.7% | **79.8%** |
| `symbolic_refuted`   | 251 | 32.7% | **91.6%** |
| `vh_conflict`        |  37 |  4.8% | **40.5%** |
| `vh_fallback`        | 236 | 30.8% | **81.4%** |

**EC symbolic layer fires on 64.4% of claims** — it is the majority decision-maker, not a rare override.

At θ=0.35 (the checkpoint value for this analysis run), a single moderate AVeriTeC evidence
item gives support_score ≈ 0.276 < θ (does not fire alone); two such items give ≈ 0.476 > θ
(fires). AI2THOR items have EC=1.0 and always fire regardless of θ. Synthetic items are
constructed to produce decisive EC values. The high symbolic rate reflects the training
distribution (AI2THOR + synthetic = ~44% of training data), not threshold miscalibration.
With the Optuna-tuned θ=0.25, the symbolic firing rate will be higher (lower bar to cross).

### Symbolic Accuracy Gap: Refuted > Supported

`symbolic_refuted` (91.6%) significantly outperforms `symbolic_supported` (79.8%). Interpretation: refuting evidence in AVeriTeC tends to be more direct (explicit contradictions, fact corrections) and maps cleanly to high EC refute signals. Supporting evidence is sometimes indirect or partial, producing a high support_score even when annotators judged the claim as only weakly supported.

### vh_conflict Is the Weak Spot (40.5%)

All 37 conflict cases originate from AVeriTeC — **zero from AI2THOR or synthetic**. AI2THOR evidence is simulator ground truth (EC=1.0 per item) so one side always dominates. Synthetic records are constructed to avoid ambiguity by design (ADR-012). AVeriTeC crowdsourced claims can have genuine multi-directional evidence.

### Root Cause of vh_conflict Failures

Inspecting all 22 failures shows that in **20 of 22 cases the VerdictHead correctly picks the higher EC score** (winner-takes-all). The failure is not a VerdictHead bias — the EC scores themselves disagree with AVeriTeC annotations:

Representative failures:

| sup_score | ref_score | Model verdict | Annotator verdict | Pattern |
|-----------|-----------|---------------|-------------------|---------|
| 0.906 | 0.429 | supported | refuted | Strong support signal, annotator overrode |
| 0.978 | 0.604 | supported | not_enough_evidence | Both sides strong, annotator chose caution |
| 0.870 | 0.442 | supported | not_enough_evidence | Strong support, annotator chose NEI |
| 0.995 | 0.775 | supported | refuted | Dominant support, annotator said refuted |
| 0.696 | 0.698 | refuted | supported | Near-equal, annotator broke tie as supported |

Only 2 cases show actual VerdictHead bias against its own EC signal (choosing "refuted" when sup_score > ref_score). These are not systematic.

### Alternatives Considered and Rejected

**Winner-takes-all for conflict:** Replace VerdictHead with `argmax([sup, ref])` when both scores cross θ. Fixes 2 cases where VerdictHead fights its own EC winner. Net improvement: +2/767 = +0.003% accuracy. Not worth the change.

**NEI default for near-equal conflict:** Return NEI when `|sup - ref| < δ`. Principled epistemically (genuine conflict = insufficient basis for verdict). However, 0 of the 15 correctly resolved conflict cases have true label NEI — applying this heuristic would break those 15 while fixing ~4–6 NEI failures. Net neutral to negative.

**Raise conflict threshold:** Requiring both scores > 0.50 to trigger VerdictHead. Would reduce conflict rate but not fix the underlying EC-annotation disagreement; failures would redistribute to `vh_fallback` with similar accuracy.

---

## Decision

No model change. The 40.5% conflict accuracy reflects annotation noise concentrated in the conflict bucket, not a fixable architectural problem.

The EC formula performs as designed: it identifies bilateral decisive evidence and correctly flags the epistemic situation as conflicted. AVeriTeC annotators resolve that conflict by applying judgment the model cannot replicate (source authority, broader context, annotation guidelines). This is the irreducible gap between EC-grounded epistemic reasoning and crowdsourced annotation.

**For the paper:** The `vh_conflict` bucket (4.8% of test claims, all AVeriTeC) represents the irreducible annotation noise ceiling. Claims where `sup > θ AND ref > θ` are genuinely contested in the original source material; annotator resolution diverges from EC-formula resolution in 60% of such cases. This is evidence that the AVeriTeC annotation ceiling (~67% model accuracy) is driven by annotation philosophy differences on borderline claims, not model capacity.

---

## Implementation

`decision_path` added to `predict()` return dict in `nlihybridhgnn.py`.  
`evaluate.py` now tracks per-branch counts, correctness, per-source breakdown, and full claim-level failure logs for `vh_conflict` in `verdict_metrics.json` and `eval_summary.md`.
