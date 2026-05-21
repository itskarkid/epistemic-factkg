"""AppConfig — single config object for app.

Reads configs/config.yaml for paths and display metadata.
Derives enum-based values directly from src/ (no duplication).

Usage:
    from app.config import get_config, enum_label
    cfg = get_config()
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


# Project root = two levels up from this file (app/config.py → project root)
ROOT: Path = Path(__file__).parent.parent.resolve()


def enum_label(value: str) -> str:
    """Derive a human-readable label from an enum string value.

    "not_enough_evidence" → "Not Enough Evidence"
    "web_text"            → "Web Text"
    "news_media"          → "News Media"
    """
    return value.replace("_", " ").title()


@dataclass(frozen=True)
class AppConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    root:             Path
    unified_jsonl:    Path
    training_jsonl:   Path
    registry_path:    Path
    splits_dir:       Path
    reports_root:     Path
    graph_cache_dir:  Path
    data_report_dir:  Path

    # ── Display — from config.yaml (no label strings; use enum_label() at render time)
    verdict_display:    dict   # {key: {emoji, color, css_class}}
    stance_display:     dict   # {key: {color, css_class}}
    model_descriptions: dict   # {model_key: str}

    # ── EC ─────────────────────────────────────────────────────────────────────
    default_ec_threshold: float

    # ── Tab definitions — icon from YAML; label = key.title() ─────────────────
    tab_defs: tuple   # tuple of dicts [{key, icon}, ...]  (frozen → tuple)

    # ── Derived from src/ — NOT from YAML ──────────────────────────────────────
    model_keys:              tuple   # list(MODELS.keys())
    modality_values:         tuple   # [m.value for m in MODALITY]
    source_type_values:      tuple   # [s.value for s in SOURCE_TYPE]
    modality_evidence_types: dict    # _MODALITY_TO_EVIDENCE_TYPES
    int_to_verdict:          dict    # VERDICT_TO_INT inverted
    int_to_stance:           dict    # STANCE_TO_INT inverted (unique ints only)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and return the singleton AppConfig.

    Reads configs/config.yaml once; derives enum values from src/.
    Cached so repeated calls are free.
    """
    raw = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    app = raw.get("app", {})
    paths = app.get("paths", {})
    display = app.get("display", {})
    models_cfg = app.get("models", {})
    ec_cfg = app.get("ec", {})
    tabs_cfg = app.get("tabs", [])

    def p(key: str, fallback: str) -> Path:
        return ROOT / paths.get(key, fallback)

    # Import from src/ — single source of truth
    from src.model.models import MODELS
    from src.model.data.types import (
        MODALITY,
        SOURCE_TYPE,
        VERDICT_TO_INT,
        STANCE_TO_INT,
        _MODALITY_TO_EVIDENCE_TYPES,
    )

    int_to_verdict = {v: k for k, v in VERDICT_TO_INT.items()}
    # STANCE_TO_INT has duplicate ints (nee and conflicting both → 2); keep first seen
    int_to_stance: dict[int, str] = {}
    for k, v in STANCE_TO_INT.items():
        if v not in int_to_stance:
            int_to_stance[v] = k

    return AppConfig(
        root=ROOT,
        unified_jsonl=p("unified_jsonl",   "out/data/unified/epistemic_factkg.jsonl"),
        training_jsonl=p("training_jsonl",  "out/data/training/epistemic_factkg_training.jsonl"),
        registry_path=p("registry",         "data/registry/source_trust_registry.jsonl"),
        splits_dir=p("splits_dir",          "out/data/splits"),
        reports_root=p("reports_root",      "out/reports/model"),
        graph_cache_dir=p("graph_cache_dir","out/model"),
        data_report_dir=p("data_report_dir","out/reports/data"),
        verdict_display=display.get("verdict", {}),
        stance_display=display.get("stance", {}),
        model_descriptions=models_cfg.get("descriptions", {}),
        default_ec_threshold=float(ec_cfg.get("default_threshold", 0.35)),
        tab_defs=tuple(tabs_cfg),
        model_keys=tuple(MODELS.keys()),
        modality_values=tuple(m.value for m in MODALITY),
        source_type_values=tuple(s.value for s in SOURCE_TYPE),
        modality_evidence_types=dict(_MODALITY_TO_EVIDENCE_TYPES),
        int_to_verdict=int_to_verdict,
        int_to_stance=int_to_stance,
    )
