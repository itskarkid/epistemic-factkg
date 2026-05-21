# Epistemic FactKG

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A fact verification dataset and pipeline grounded in Indian Pramana epistemology. Combines AI2-THOR simulation-state claims and AVeriTeC web-evidence claims into a unified schema for training epistemic graph neural networks. Unlike existing systems that treat all evidence as equivalent, this framework assigns each claim an explicit epistemic category — encoding *how knowledge is obtained*, not just whether the claim is true.

## Pramana Framework

Confidence weights derived from the classical Indian knowledge-source taxonomy:

| Pramana | Category | Confidence | Source |
|---|---|---|---|
| Pratyakṣa | `perception` | 0.95 | AI2-THOR simulation state (closed-world ground truth) |
| Śabda | `testimony` | 0.80 | AVeriTeC web text / PDFs |
| Anupalabdhi | `non_apprehension` | 0.75 | Sensor-confirmed absence of object or state |
| Upamāna | `comparison_analogy` | 0.65 | Numeric or analogy-based claims |
| Anumāna | `inference` | 0.55 | Multi-hop or synthesised reasoning |
| Arthāpatti | `postulation_derivation` | 0.40 | Hypothetical derivation |

When multiple Pramana apply, confidence is combined via diminishing returns:
`combined = 1 − Π(1 − wᵢ)` — implemented in `src/core/claims/labels.py:combine_pramana_weights()`.

## Quick Start

```bash
just init          # Install dependencies

# ── Data pipeline ─────────────────────────────────────────────
just run data      # build → validate → report  (full data pipeline)

# Step by step:
just build         # merge all sources, filter for training, split
just validate      # schema + training-distribution checks
just report        # markdown report + charts from validation output

# ── Model pipeline ────────────────────────────────────────────
just run model     # graph → train → eval → compare  (all registered models)

# Run specific models only:
just run model "" v1-hgnn,baseline

# Step by step for a specific model:
just graph                  # build shared PyG graph dataset (once)
just train                  # train default model (v1-hgnn)
just train baseline         # train a different model
just eval                   # eval default model
just eval baseline          # eval a different model

# Multi-model:
just run model list                     # list all registered models
just run model train v1-hgnn,baseline   # train two models
just run model eval  v1-hgnn,baseline   # eval two models
just run model compare v1-hgnn,baseline # compare report
just compare v1-hgnn baseline           # shorthand comparison
```

## Installation

**Requirements:** Python 3.14+, [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just)

```bash
# macOS
brew install uv just

git clone <repo-url> && cd epistemic-factkg
just init
```

## Data Setup

### AVeriTeC

Download from [fever.ai/task.html](https://fever.ai/task.html) and place under `data/raw/averitec/`:
```
data/raw/averitec/
├── train.json
├── dev.json
└── test.json
```

### AI2-THOR

Claims are generated from simulation — no external download needed:
```bash
just build         # runs KG generation + claim extraction + conversion
just build rebuild=true   # re-simulate AI2-THOR from scratch first
```

Outputs: `out/model/knowledge_graph.ttl` (RDF graph) and `data/raw/ai2thor/claims_all.jsonl`.

## Model Architecture

**EpistemicHGNN** is a neuro-symbolic heterogeneous graph neural network with three prediction heads:

```
EpistemicEncoder  (HeteroConv, config-driven — GraphConfig.v1())
    ↓ evidence embeddings [N_ev, hidden_dim]
H1 StanceHead   → stance logits     [N_ev, 3]      supports / refutes / neutral
H2 ISHead       → IS scalars        [N_ev, 1]      information strength ∈ [0,1]
    ↓ soft symbolic EC aggregation (differentiable during training)
VerdictHead     → verdict logits    [N_claims, 3]
```

**Multi-task loss:** `L = stance_CE + λ₁ · IS_MSE + λ₂ · verdict_CE`

At inference: hard argmax stance → symbolic EC scores → VerdictHead → verdict string.

## Adding a New Model

1. Create `src/model/models/<name>.py` with your `nn.Module` class:
   - Constructor: `__init__(self, graph_config, hidden_dim, heads, dropout)`
   - `forward(data) → {"stance_logits", "is_pred", "verdict_logits"}`
   - `predict(data) → {"stance_pred", "stance_logits", "is_pred", "verdict"}`

2. Register it in `src/model/models/__init__.py`:
   ```python
   from src.model.models.your_model import YourModel
   MODELS = {
       "v1-hgnn": EpistemicHGNN,
       "your-name": YourModel,   # ← add this line
   }
   ```

3. Run it: `just train your-name` / `just eval your-name`

## Output Structure

```
out/
├── data/
│   ├── intermediate/          per-source JSONLs before merge
│   ├── unified/               merged epistemic_factkg.jsonl
│   ├── training/              filtered training JSONL (no postulation_derivation)
│   └── splits/                train/val/test_indices.json
├── model/
│   ├── graphs/                graph_dataset.pt + embed_cache.pkl  (shared)
│   └── <model-name>/
│       └── checkpoints/       best_model.pt
└── reports/
    ├── data/                  validation.json, training_validation.json, summary.md
    │   └── plots/             dataset distribution charts
    └── model/
        └── <model-name>/
            ├── training_history.json
            ├── eval_summary.md         stance/IS/verdict with per-class tables + plots
            └── eval/
                ├── stance_metrics.json
                ├── is_metrics.json
                ├── verdict_metrics.json
                └── plots/              confusion_matrix.png  class_f1.png  per_source_accuracy.png
        comparison_<m1>_vs_<m2>.md      generated by just compare
```

## Project Structure

```
epistemic-factkg/
├── configs/                   Scene + claim generation config
├── data/
│   ├── raw/                   Source datasets (AVeriTeC, AI2-THOR, synthetic)
│   └── registry/              source_trust_registry.jsonl, seed_pool.jsonl
├── docs/
│   ├── adr/                   Architecture Decision Records (001–014)
│   └── research-overview.md
├── src/
│   ├── core/                  Pramana weights, schema, domain logic
│   ├── epistemic/             Source trust registry
│   ├── model/
│   │   ├── models/            Model registry + one file per model class
│   │   ├── architecture/      Encoder, StanceHead, ISHead, VerdictHead, Aggregator
│   │   ├── data/              ClaimGraphBuilder, Featurizer, dataset utilities
│   │   ├── evaluation/        Metrics (accuracy, F1, ECE, RMSE, Pearson), inference
│   │   └── training/          Trainer, TrainConfig
│   ├── pipeline/
│   │   ├── data/              build, generate, validate, split
│   │   └── model/             build_graphs, train, evaluate, report, orchestrate, compare
│   └── utils/                 time, io, logger
├── tests/
├── Justfile
└── pyproject.toml
```

## Key Commands

```bash
just init                       # Install dependencies
just build                      # Build dataset (merge, filter, split)
just validate                   # Validate schema + training distribution
just report                     # Dataset quality report (markdown + charts)
just graph                      # Build shared PyG graph dataset
just train [model]              # Train a model (default: v1-hgnn)
just eval  [model]              # Evaluate on test split
just compare model1 model2      # Side-by-side comparison markdown
just run data                   # Full data pipeline
just run model                  # Full model pipeline (all registered models)
just run model list             # List registered models
just run model train m1,m2      # Train specific models
just run model compare m1,m2    # Compare specific models
just lint                       # Ruff format + lint check
just fix                        # Auto-fix lint issues
just test                       # pytest
just clean                      # Delete all generated outputs (out/)
```

## Documentation

| Document | Description |
|---|---|
| [Research Overview](docs/research-overview.md) | Problem motivation, Pramana inspiration, related work |
| [ADR-001](docs/adr/001-epistemic-framework.md) | Pramana epistemic framework |
| [ADR-002](docs/adr/002-ports-and-adapters-architecture.md) | Ports & Adapters (hexagonal) architecture |
| [ADR-003](docs/adr/003-floorplan-based-train-test-split.md) | Floorplan-based train/dev/test split |
| [ADR-004](docs/adr/004-rdf-as-kg-intermediate-format.md) | RDF/Turtle as KG intermediate format |
| [ADR-005](docs/adr/005-exclude-postulation-derivation-from-training.md) | Exclude postulation_derivation from GNN training |
| [ADR-006](docs/adr/006-dataset-composition-and-generation-strategy.md) | Dataset composition and generation strategy |
| [ADR-007](docs/adr/007-verdict-class-reduction.md) | Reduce verdict classes from 4 to 3 |
| [ADR-008](docs/adr/008-evaluation-protocol.md) | Evaluation protocol |
| [ADR-009](docs/adr/009-source-trust-registry.md) | Source trust registry |
| [ADR-010](docs/adr/010-per-evidence-epistemic-modeling.md) | Per-evidence epistemic modeling |
| [ADR-011](docs/adr/011-evidence-labeling-rules.md) | Per-evidence labeling rules |
| [ADR-012](docs/adr/012-shortcut-leakage-and-synthetic-data-strategy.md) | Shortcut leakage and synthetic data strategy |
| [ADR-013](docs/adr/013-synthetic-pipeline.md) | Synthetic data pipeline |
| [ADR-014](docs/adr/014-verdict-head-learned-thresholds.md) | Learned verdict thresholds (VerdictHead) |
| [ADR-015](docs/adr/015-class-weighted-loss.md) | Class-weighted loss for imbalanced verdicts |
| [ADR-016](docs/adr/016-baseline-hgnn-ablation.md) | Baseline HGNN ablation design |
| [ADR-017](docs/adr/017-is-jitter-synthetic.md) | IS jitter on synthetic data |
| [ADR-018](docs/adr/018-nei-heavy-synthetic-distribution.md) | NEI-heavy synthetic distribution |
| [ADR-019](docs/adr/019-averitec-is-rubric.md) | AVeriTeC IS rubric |
| [ADR-020](docs/adr/020-webarchive-source-trust-resolution.md) | Web Archive source trust resolution |
| [ADR-021](docs/adr/021-is-cap-by-source-trust.md) | IS capped by source trust |
| [ADR-022](docs/adr/022-is-gradient-detach.md) | IS gradient detach for clean regression |
| [ADR-023](docs/adr/023-hybrid-hgnn-v2.md) | HybridHGNN v2 — HybridVerdictHead |
| [ADR-024](docs/adr/024-nli-evidence-augmentation.md) | NLI probs as evidence features (Part 1 retained; Part 2 superseded by ADR-029) |
| [ADR-025](docs/adr/025-averitec-qa-nli-preprocessing.md) | AVeriTeC Q+A NLI pre-processing |
| [ADR-026](docs/adr/026-encoder-residuals-windowed-coevidence.md) | Encoder residuals + windowed co-evidence |
| [ADR-027](docs/adr/027-predict-full-verdicthead-delegation.md) | Full VerdictHead delegation (removed EC_NEI_MAX) |
| [ADR-028](docs/adr/028-generator-direct-v3-output.md) | Generator direct v3.0 output — schema corrections |
| [ADR-029](docs/adr/029-nli-claim-aware-stance-head.md) | v3-nli claim-aware H1 StanceHead on GNN output (supersedes ADR-024 Part 2) |
| [ADR-030](docs/adr/030-ec-decision-path-analysis.md) | EC decision path analysis: symbolic vs VerdictHead vote distribution and vh_conflict failure diagnosis |

## Testing

```bash
just test
# or:
uv run ruff format . && uv run ruff check .
uv run pytest tests/ -v --tb=short
```

## Citation

If you use this work in research, please cite:

```bibtex
@software{epistemic_factkg_2026,
  title  = {Epistemic FactKG: Pramana-grounded Fact Verification},
  author = {Karki, Dheeraj},
  year   = {2026},
  url    = {https://github.com/yourusername/epistemic-factkg}
}
```

Also cite the datasets:

```bibtex
@inproceedings{schlichtkrull2023averitec,
  title     = {AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author    = {Schlichtkrull, Michael and others},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2023}
}

@article{kolve2017ai2thor,
  title   = {AI2-THOR: An Interactive 3D Environment for Visual AI},
  author  = {Kolve, Eric and others},
  journal = {arXiv:1712.05474},
  year    = {2017}
}
```

## License

MIT — see [LICENSE](LICENSE).
