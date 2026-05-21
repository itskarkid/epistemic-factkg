"""Cached data-loading functions for app_update.

All functions accept explicit Path arguments (derived from AppConfig) so
Streamlit can cache them correctly when paths change between runs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import streamlit as st


# ── Training records ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_records_list(training_jsonl: Path) -> list[dict]:
    """All training JSONL records as a list (for search/filter)."""
    if not training_jsonl.exists():
        return []
    return [
        json.loads(line)
        for line in training_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@st.cache_data(show_spinner=False)
def load_all_records_indexed(training_jsonl: Path) -> dict[str, dict]:
    """Training records keyed by `id`."""
    if not training_jsonl.exists():
        return {}
    result: dict[str, dict] = {}
    with open(training_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                result[r.get("id", str(i))] = r
    return result


@st.cache_data(show_spinner=False)
def load_test_records(training_jsonl: Path, splits_dir: Path) -> list[dict]:
    """Records belonging to the test split."""
    test_idx_path = splits_dir / "test_indices.json"
    if not training_jsonl.exists() or not test_idx_path.exists():
        return []
    indices = set(json.loads(test_idx_path.read_text(encoding="utf-8"))["indices"])
    records: list[dict] = []
    with open(training_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in indices:
                records.append(json.loads(line))
    return records


# ── Dataset stats ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset_stats(data_report_dir: Path, splits_dir: Path) -> dict | None:
    """Load from validation.json + training_validation.json."""
    val_path   = data_report_dir / "validation.json"
    train_path = data_report_dir / "training_validation.json"

    result: dict = {}

    if val_path.exists():
        try:
            val = json.loads(val_path.read_text(encoding="utf-8"))
            summaries = val.get("summaries", [])
            if summaries:
                s = summaries[0]
                result["counts"]         = s.get("counts", {})
                result["coverage"]       = s.get("coverage", {})
                result["distributions"]  = s.get("distributions", {})
                result["schema_errors"]  = s.get("schema_errors_top", {})
                result["logic_warnings"] = s.get("logic_warnings_top", {})
        except Exception:
            pass

    if train_path.exists():
        try:
            result["training"] = json.loads(train_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    splits: dict[str, int] = {}
    for split in ("train", "val", "test"):
        p = splits_dir / f"{split}_indices.json"
        if p.exists():
            try:
                splits[split] = len(json.loads(p.read_text(encoding="utf-8"))["indices"])
            except Exception:
                pass
    result["splits"] = splits

    return result if result else None


# ── Source trust registry ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_registry(registry_path: Path) -> list[dict]:
    if not registry_path.exists():
        return []
    return [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@st.cache_data(show_spinner=False)
def registry_source_types(registry_path: Path) -> list[str]:
    """Unique source_type values from registry, sorted. Used for Verify tab dropdown."""
    entries = load_registry(registry_path)
    types = sorted({e["source_type"] for e in entries if "source_type" in e})
    return types if types else ["unknown"]


@st.cache_data(show_spinner=False)
def source_type_index(registry_path: Path) -> dict[str, str]:
    """Return {source_id: encoder_category} from the registry."""
    from src.model.data.types import _REGISTRY_TYPE_TO_CATEGORY
    return {
        e["source_id"]: _REGISTRY_TYPE_TO_CATEGORY.get(e.get("source_type", "unknown"), "unknown")
        for e in load_registry(registry_path)
        if "source_id" in e
    }


# ── Graph cache ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading graph cache…")
def load_graph_cache(model_key: str, graph_cache_dir: Path) -> dict | None:
    p = graph_cache_dir / f"split_cache_{model_key}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def build_graph_id_map(
    model_key: str,
    training_jsonl: Path,
    splits_dir: Path,
    graph_cache_dir: Path,
) -> dict[str, tuple[str, int]]:
    """Return {claim_id: (split, graph_index)} via sequential scan."""
    cache = load_graph_cache(model_key, graph_cache_dir)
    if not cache or not training_jsonl.exists():
        return {}

    records = [
        json.loads(line)
        for line in training_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    result: dict[str, tuple[str, int]] = {}
    for split in ("train", "val"):
        idx_path = splits_dir / f"{split}_indices.json"
        if not idx_path.exists():
            continue
        split_indices = json.loads(idx_path.read_text(encoding="utf-8"))["indices"]
        graphs = cache.get(split, [])

        graph_pos = 0
        for row_idx in split_indices:
            if graph_pos >= len(graphs):
                break
            if row_idx >= len(records):
                continue
            rec = records[row_idx]
            has_ev = any((e.get("text") or "").strip() for e in rec.get("evidence", []))
            if not has_ev:
                continue
            result[rec.get("id", str(row_idx))] = (split, graph_pos)
            graph_pos += 1

    return result


# ── Per-split dataset distributions ──────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_split_distributions(training_jsonl: Path, splits_dir: Path) -> dict[str, dict]:
    """Per-split (train/val/test) distribution breakdown.

    Returns {split: {total, sources, verdict, modality, stance, evidence_types, structural}}.
    """
    import collections

    if not training_jsonl.exists():
        return {}

    records = [
        json.loads(line)
        for line in training_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    result: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        idx_path = splits_dir / f"{split}_indices.json"
        if not idx_path.exists():
            continue
        try:
            indices = json.loads(idx_path.read_text(encoding="utf-8"))["indices"]
        except Exception:
            continue

        sources:     dict[str, int] = collections.Counter()
        verdicts:    dict[str, int] = collections.Counter()
        modalities:  dict[str, int] = collections.Counter()
        stances:     dict[str, int] = collections.Counter()
        ev_types:    dict[str, int] = collections.Counter()
        n_ev_list:   list[int]      = []
        n_triple_list: list[int]    = []

        for i in indices:
            if i >= len(records):
                continue
            rec = records[i]
            sources[rec.get("provenance", {}).get("dataset", "unknown")] += 1
            verdicts[(rec.get("verdict") or {}).get("label", "unknown")] += 1
            for ev in rec.get("evidence", []):
                modalities[ev.get("modality", "unknown")] += 1
                stances[ev.get("stance", "unknown")] += 1
                for et in ev.get("evidence_types", []):
                    ev_types[et] += 1
            n_ev = len([e for e in rec.get("evidence", []) if (e.get("text") or "").strip()])
            n_ev_list.append(n_ev)
            n_triple_list.append(len(rec.get("claim_triples") or []))

        result[split] = {
            "total":          len(indices),
            "sources":        dict(sources),
            "verdict":        dict(verdicts),
            "modality":       dict(modalities),
            "stance":         dict(stances),
            "evidence_types": dict(ev_types),
            "structural": {
                "avg_evidence": round(sum(n_ev_list) / len(n_ev_list), 2) if n_ev_list else 0,
                "avg_triples":  round(sum(n_triple_list) / len(n_triple_list), 2) if n_triple_list else 0,
            },
        }

    return result


# ── Per-model report files ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_verdict_metrics(model_key: str, reports_root: Path) -> dict | None:
    p = reports_root / model_key / "eval" / "verdict_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_stance_metrics(model_key: str, reports_root: Path) -> dict | None:
    p = reports_root / model_key / "eval" / "stance_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_is_metrics(model_key: str, reports_root: Path) -> dict | None:
    p = reports_root / model_key / "eval" / "is_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_training_history(model_key: str, reports_root: Path) -> dict | list | None:
    p = reports_root / model_key / "training_history.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Predictor ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_predictor(model_key: str, models_root: Path):
    """Load and cache an EpistemicPredictor for the given model key."""
    from app_update.core.predictor import EpistemicPredictor
    try:
        return EpistemicPredictor(model_key, models_root=models_root)
    except FileNotFoundError as e:
        return str(e)
