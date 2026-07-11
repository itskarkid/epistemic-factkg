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

**Note:** Baseline outperforms trust-aware models on the full mixed dataset — same leakage dynamic as diagnosed in the BlackboxNLP paper. This is part of the system paper narrative, not a flaw.

---

## What Is Missing (for system paper)

1. **FEVER evaluation** — no adapter exists; no SOTA comparison yet
2. **EC ablation** — ST-only vs ST+EW vs full ST+EW+IS not isolated
3. **Explanation of baseline > trust-aware on full dataset** — addressed in paper narrative

---

## Phase 1 — FEVER Adapter
**Target: July–August 2026**

Build `src/adapters/fever.py` following the pattern in `src/adapters/averitec.py`.

- FEVER verdict mapping: `SUPPORTS → supported`, `REFUTES → refuted`, `NOT ENOUGH INFO → not_enough_evidence`
- Evidence source: Wikipedia → use `wikipedia_web` entry in source trust registry (ST = 0.82)
- Evidence type: testimony (EW = 0.80), IS = 1.0
- Output: `out/data/unified/fever_unified.jsonl` in the existing unified schema

Add to Justfile:
```just
eval-fever:
    uv run python scripts/run_pipeline.py --source fever --split test
```

SOTA comparison (do not re-implement — cite published numbers):
- GEAR: 71.0% FEVER label accuracy
- KGAT: 78.3%
- LOREN: 83.4%
- HeterFC: 80.7%

**Decision gate:** If FEVER macro-F1 ≥ 80% → ACL 2027 main. If 75–80% → EMNLP 2027 findings. If < 75% → pivot to dataset paper framing.

---

## Phase 2 — EC Ablation Study
**Target: August 2026**

Modify `src/model/architecture/aggregator.py` to support 3 ablation modes (config flag, no new model code):

| Variant | Formula |
|---|---|
| ST-only | EC = ST |
| ST + EW | EC = 1 − (1 − ST)^EW |
| Full EC | EC = 1 − (1 − ST)^(EW × IS) |

Run ablation on both the trust-isolating benchmark (cite BlackboxNLP) and FEVER.

Add to Justfile:
```just
eval-ablation mode:
    uv run python scripts/run_pipeline.py --ablation {{mode}}
```

---

## Phase 3 — Paper Writing
**Target: November–December 2026**

Wait for BlackboxNLP proceedings (expected November 2026) before writing — need to de-anonymize the self-citation (`anonymous2026system` → real BibTeX key).

**Target venue:** ACL 2027 (deadline ~February 2027)
**Fallback:** EMNLP 2027 (deadline ~June 2027)

### Paper structure (8 pages)

1. **Introduction** — epistemic FV motivation, Pramana framing, 3 contributions
2. **Background** — FV benchmarks, shortcuts, source credibility, graph FV
3. **Epistemic Framework** — EC formula, source trust registry, evidence types
4. **Dataset** — 3 sources, schema, statistics, trust distribution
5. **Model** — architecture (encoder, heads, symbolic aggregator, hybrid verdict)
6. **Experiments** — FEVER + full dataset + ablation + trust-isolating benchmark (cite BlackboxNLP)
7. **Analysis** — per-source breakdown, mid-tier gap, leakage on mixed data
8. **Conclusion + Limitations**

### Figures to generate

| Figure | Source |
|---|---|
| EC formula curve | Reuse `docs/figures/ec_curve.png` or regenerate |
| Architecture diagram | Reuse from BlackboxNLP `pb_architecture.png` |
| Per-source accuracy bar chart | Generate from `out/reports/model/` |
| FEVER results table | After Phase 1 |
| Ablation table | After Phase 2 |

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

## Timeline

| Period | Task |
|---|---|
| July–Aug 2026 | FEVER adapter + eval all 4 models |
| Aug 2026 | EC ablation study |
| Sep–Oct 2026 | BlackboxNLP review response + workshop (Budapest, Oct 28) |
| Nov 2026 | BlackboxNLP proceedings published → de-anonymize self-citation |
| Nov–Dec 2026 | Write system paper draft |
| Jan–Feb 2027 | Submit to ACL 2027 |
