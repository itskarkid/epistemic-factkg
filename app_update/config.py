"""Centralised path & model configuration for app_update.

Reads ``configs/config.yaml`` (the project's single config source).
YAML is parsed once at import time; all paths are resolved as absolute
``pathlib.Path`` objects relative to the project root.

Usage::

    from config import UNIFIED_JSONL, MODEL_NAME, VERDICT_COLORS, ...
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Project root = parent of this file's parent (app_update/../)
ROOT: Path = Path(__file__).parent.parent.resolve()

_cfg: dict = yaml.safe_load(
    (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
)
_app: dict   = _cfg.get("app",   {})
_paths: dict  = _app.get("paths",  {})
_model: dict  = _app.get("model",  {})
_colors: dict = _app.get("colors", {})


def _p(key: str, fallback: str) -> Path:
    """Resolve paths.<key> from config.yaml as an absolute Path."""
    return ROOT / _paths.get(key, fallback)


# ── Pipeline / output paths (used by app tabs) ────────────────────────────────
CONFIG            = ROOT / "configs" / "config.yaml"
REGISTRY          = _p("registry",           "data/registry/source_trust_registry.jsonl")
SEED_POOL         = _p("seed_pool",           "data/registry/seed_pool.jsonl")
UNIFIED_JSONL     = _p("unified_jsonl",       "out/data/unified/epistemic_factkg.jsonl")
TRAINING_JSONL    = _p("training_jsonl",      "out/data/training/epistemic_factkg_training.jsonl")
VALIDATION_JSON   = _p("validation_json",     "out/reports/data/validation.json")
TRAINING_VALIDATION = _p("training_validation", "out/reports/data/training_validation.json")
REPORT_DIR        = _p("report_dir",          "out/reports/data")
SPLITS_DIR        = _p("splits_dir",          "out/data/splits")
GRAPH_DATASET     = _p("graph_dataset",       "out/model/graphs/graph_dataset.pt")
GRAPH_DATASET_NLI = _p("graph_dataset_nli",   "out/model/graphs/graph_dataset_nli.pt")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME: str  = _model.get("name", "v1-hgnn")
CHECKPOINTS_DIR  = ROOT / f"out/model/{MODEL_NAME}/checkpoints"
MODEL_REPORT_DIR = ROOT / f"out/reports/model/{MODEL_NAME}"
RESULTS_DIR      = ROOT / f"out/reports/model/{MODEL_NAME}/eval"

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS: dict         = _colors                         # full colors block
VERDICT_COLORS: dict = _colors.get("verdict", {})      # keyed by verdict label
LAYER_COLORS: dict   = _colors.get("layer",   {})      # keyed by layer name
CHIP_COLORS: dict    = _colors.get("chip",    {})      # keyed by chip variant
