# Epistemic-FactKG — Research Roadmap

**Current status:** System complete (v3 finalized). BlackboxNLP 2026 benchmark paper submitted (submission #39, July 3 2026, under review). Next target: system paper at ACL 2027.

---

## What Is Done

| Component | Status |
|---|---|
| Schema v3.0 — per-evidence EC, source trust registry | ✅ Complete |
| Data pipeline — AI2THOR + AVeriTeC + synthetic (7,868 records) | ✅ Complete |
| Source trust registry — 183 entries, MBFC-referenced | ✅ Complete |
| 4 GNN variants — baseline, v1-hgnn, v2-hgnn, v3-nli | ✅ Trained + evaluated |
| Hyperparameter search — 30 Optuna trials per model | ✅ Complete |
| Streamlit demo app | ✅ Complete |
| 30 ADRs — all design decisions documented | ✅ Complete |
| BlackboxNLP 2026 paper — evaluation methodology + inversion finding | ✅ Submitted |

### Current Results (val set, 767 claims)

| Model | Verdict Acc | Macro-F1 |
|---|---|---|
| v3-nli | 0.820 | 0.816 |
| baseline | 0.816 | 0.828 |
| v2-hgnn | 0.794 | 0.793 |
| v1-hgnn | 0.761 | 0.751 |

Per-source: AI2THOR ~95–100% · AVeriTeC ~53–67% · Synthetic ~89–100%

**Note:** Baseline outperforms trust-aware models on the full mixed dataset — caused by the vh_fallback shortcut (VerdictHead reads text register when EC < 0.75). This is the architectural flaw being fixed in Phase 1, and it is part of the system paper narrative.

---

## What Is Missing (for system paper)

1. **Architectural fix** — vh_fallback routes to VerdictHead (shortcut path); must auto-output NEE
2. **New synthetic training data** — contrastive pairs across the full ST × EW × IS parameter grid; existing 2,500 synthetic records are evaluation-only (trust-isolating benchmark, cited in BlackboxNLP)
3. **Training set correction** — remove AVeriTeC from training; AVeriTeC becomes zero-shot generalization eval
4. **EC ablation** — ST-only vs ST+EW vs full EC components not yet isolated

**Why no FEVER:** FEVER uses only Wikipedia (constant ST ≈ 0.82), so EC ≈ 0.764 for every record — the EC formula provides no discriminative signal. Adding FEVER tests a stance classifier, not the epistemic framework. The absence of any real-world benchmark capable of evaluating source-trust-aware FV is documented as a finding in the paper.

---

## Phase 1 — Architectural Fix + New Training Data
**Target: July–August 2026**

### 1a. Fix vh_fallback in aggregator

In `src/model/architecture/aggregator.py`, the `vh_fallback` branch (EC < 0.75, neither support/refute score clears threshold) currently delegates to `VerdictHead`. This allows the model to bypass the EC framework by reading text features directly — the shortcut diagnosed in the BlackboxNLP paper.

Fix: when EC < 0.75 and no conflict → output `not_enough_evidence` directly. VerdictHead remains only for `vh_conflict` (both scores high, genuine ambiguity).

```python
# vh_fallback: EC < 0.75, scores ambiguous → NEE (no VerdictHead)
if ec < self.ec_threshold and not conflict:
    return "not_enough_evidence", ec

# vh_conflict: both scores high → VerdictHead resolves conflict
if conflict:
    return self.verdict_head(features), ec
```

### 1b. Generate synthetic-train (contrastive pairs)

The existing 2,500 synthetic records are the evaluation benchmark (cited in BlackboxNLP) — they must not be in the training set. A new synthetic-train set is needed.

**Key design principle:** Training data must use the same contrastive-pair methodology as the evaluation benchmark. If training data is just "diverse claims with varied labels", the model learns text-feature shortcuts (academic-sounding text → supported) rather than the EC formula. Contrastive pairs force the model to use EC because the claim and evidence text are identical — only the EC parameters differ.

**Three types of contrastive pairs:**

| Pair type | What varies | What is fixed | Model must learn |
|---|---|---|---|
| ST pairs | Source (ST=0.30 vs ST=0.90) | Claim + evidence text | ST determines label |
| EW pairs | Evidence type (EW=0.40 vs EW=0.95) | Claim + source (same ST) | EW determines label |
| IS pairs | Inference strength (IS=0.5 vs IS=1.0) | Claim + source + evidence type | IS determines label |

**Generation structure:**

- 600 unique claim templates (varied topics and domains)
- Each claim × 4 ST levels (0.30, 0.55, 0.72, 0.90) × 3 EW levels (0.40, 0.80, 0.95) × 3 IS levels (0.5, 0.8, 1.0)
- EC computed per combination; label derived: EC ≥ 0.75 → SUPPORTED/REFUTED (per stance); EC < 0.75 → NEE
- Full grid = 600 × 36 = 21,600; sample to ~6,000 balanced across EC tiers

**New data split after Phase 1:**

| Split | Source | Records | Role |
|---|---|---|---|
| Train | AI2THOR | 1,800 | Perception-based, max-trust (ST=1.0) anchor |
| Train | Synthetic-train (new) | ~6,000 | Contrastive pairs across ST/EW/IS grid |
| Eval | Synthetic-eval (existing) | 2,500 | Trust-isolating benchmark — cite BlackboxNLP |
| Eval | AVeriTeC (zero-shot) | 3,568 | Generalization gap — not part of training |

**AI2THOR stays in training** — it anchors the high-trust end of the EC spectrum (simulator = ground truth, ST=1.0). ~95–100% accuracy on it validates the architecture when EC is unambiguous.

Deliverable: `data/synthetic/synthetic_train_v1.jsonl` in the existing unified schema.

Add to Justfile:
```just
generate-synthetic-train:
    uv run python scripts/generate_synthetic_train.py --n-claims 600 --output data/synthetic/synthetic_train_v1.jsonl
```

### 1c. Remove AVeriTeC from training set

AVeriTeC labels reflect semantic support judged by crowd workers; EC values are heuristically assigned from the MBFC registry. The two annotation schemes measure fundamentally different things and cannot be reconciled without re-annotation.

Add a `--sources` config flag in `src/pipeline/` to select source subsets. Do not change the schema or existing adapters.

```just
train-corrected:
    uv run python scripts/run_pipeline.py --sources ai2thor,synthetic_train --split train
```

### 1d. Re-run all 4 models

After fix + new training data, re-run all 4 models. Expected outcomes:

| Metric | Expected change | Reason |
|---|---|---|
| AI2THOR accuracy | Stable or higher | Shortcut removed; data quality unchanged |
| Synthetic-eval F1 | Higher | Model trained on contrastive pairs, evaluated the same way |
| AVeriTeC zero-shot | Lower than current 53–67% | Expected — different annotation scheme, not a failure |
| Baseline vs trust-aware gap | Narrows or reverses | Shortcut path closed |

---

## Phase 2 — EC Ablation Study
**Target: August–September 2026**

Modify `src/model/architecture/aggregator.py` to support 3 ablation modes via config flag (no new model code):

| Variant | Formula | What it tests |
|---|---|---|
| ST-only | EC = ST | Does source trust alone predict verdict? |
| ST + EW | EC = 1 − (1 − ST)^EW | Does evidence type add signal over ST alone? |
| Full EC | EC = 1 − (1 − ST)^(EW × IS) | Does IS add signal over ST+EW? |

Run ablation on both the trust-isolating benchmark (synthetic-eval, cite BlackboxNLP) and the corrected training set eval. Expected: each component adds 5–10 F1 points, proving the formula is not over-parameterized.

Add to Justfile:
```just
eval-ablation mode:
    uv run python scripts/run_pipeline.py --ablation {{mode}}
```

**Additional finding to document:** Models trained on flat diverse data (not contrastive pairs) fail to learn IS and EW — they converge to ST-only behavior. Contrastive training is necessary for the full EC formula to be learned. This is a methodology contribution.

---

## Phase 3 — Paper Writing
**Target: November–December 2026**

Wait for BlackboxNLP proceedings (expected November 2026) before writing — need to de-anonymize the self-citation (`anonymous2026system` → real BibTeX key).

**Target venue:** ACL 2027 (deadline ~February 2027)
**Fallback:** EMNLP 2027 (deadline ~June 2027)

### Contributions (3 + 1)

1. **EC framework** — Pramana-grounded formula, source trust registry
2. **Contrastive training methodology** — controlled ST/EW/IS pairs necessary for learning EC; flat data reverts to shortcuts
3. **Controlled evaluation** — trust-isolating benchmark proves trust-aware models outperform baseline when evaluation matches training structure
4. **Benchmark gap finding** — no existing real-world dataset (FEVER: constant ST; AVeriTeC: annotation mismatch) can evaluate source-trust-aware FV — this is a community finding, not a limitation

### Paper structure (8 pages)

1. **Introduction** — epistemic FV motivation, Pramana framing, 4 contributions
2. **Background** — FV benchmarks, shortcuts, source credibility, graph FV
3. **Epistemic Framework** — EC formula, source trust registry, evidence type taxonomy
4. **Dataset and Methodology** — AI2THOR + contrastive synthetic-train; why contrastive pairs are necessary; AVeriTeC annotation mismatch; benchmark gap analysis
5. **Model** — architecture (encoder, heads, symbolic aggregator; vh_fallback → NEE; vh_conflict → VerdictHead)
6. **Experiments** — trust-isolating benchmark (cite BlackboxNLP) + corrected training eval + EC ablation + AVeriTeC zero-shot generalization
7. **Analysis** — contrastive training vs flat training comparison; per-source breakdown; benchmark gap table (FEVER/AVeriTeC limitations)
8. **Conclusion + Limitations**

### Figures to generate

| Figure | Source |
|---|---|
| EC formula curve | Reuse `docs/figures/ec_curve.png` or regenerate |
| Architecture diagram | Reuse from BlackboxNLP `pb_architecture.png` (update vh_fallback branch) |
| EC parameter grid heatmap | Generate: ST × EW → EC value, IS as contour |
| Contrastive pair accuracy chart | Show: contrastive training vs flat training per EC tier |
| Per-source accuracy bar chart | Generate from `out/reports/model/` after Phase 1 retrain |
| Ablation table | After Phase 2 |
| Benchmark gap table | FEVER / AVeriTeC / synthetic — why each can/cannot evaluate EC |

### Submission folder

Create `submissions/acl2027/` following the same structure as `submissions/blackboxnlp2026/`:
```
submissions/acl2027/
├── paper.tex
├── acl.sty
├── acl_natbib.bst
├── references.bib
└── figures/  ← symlink to ../../docs/figures/
```

---

## Phase 4 — Submit
**Target: January–February 2027**

- Double-blind: anonymize authors, anonymize self-citations
- Add `pdf-system` target to Justfile
- Verify: 8 pages, all fonts embedded, A4 paper size

---

## Decision Gate (before Phase 3)

| Question | Condition | Action |
|---|---|---|
| Contrastive training improves synthetic-eval over current? | Phase 1 results | Expected yes — contrastive pairs align train/eval structure |
| Ablation shows each EC component contributes? | Phase 2 results | Expected yes — each adds ~5–10 F1 on trust-isolating benchmark |
| AVeriTeC zero-shot gap explainable? | Annotation mismatch documented | Frame as finding, not failure |
| Flat vs contrastive training comparison shows difference? | Phase 2 extra run | Needed to prove methodology contribution |
| Paper venue decision | Results clarity | ACL 2027 main if clean; EMNLP 2027 if more analysis needed |

---

## Timeline

| Period | Task |
|---|---|
| July 2026 | Fix vh_fallback; generate synthetic-train (contrastive pairs) |
| July–Aug 2026 | Remove AVeriTeC from train; retrain all 4 models |
| Aug–Sep 2026 | EC ablation; flat vs contrastive training comparison; AVeriTeC zero-shot eval |
| Sep–Oct 2026 | BlackboxNLP review response + workshop (Budapest, Oct 28) |
| Nov 2026 | BlackboxNLP proceedings published → de-anonymize self-citation |
| Nov–Dec 2026 | Write system paper draft |
| Jan–Feb 2027 | Submit to ACL 2027 |
