"""Session state helpers for app_update."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig


def blank_ev() -> dict:
    return {"text": "", "source_type": "unknown", "modality": "web_text"}


def init_state(cfg: "AppConfig") -> None:
    defaults: dict = {
        "evidence_list":      [blank_ev()],
        "last_claim":         "",
        "_random_true_label": None,
        "current_claim_id":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    pending = st.session_state.pop("_pending_claim", None)
    if pending is not None:
        st.session_state["claim_input"] = pending


def load_record_into_state(rec: dict, cfg: "AppConfig") -> None:
    """Push a record's claim + evidence into session state for the Verify tab."""
    from app_update.core.loaders import source_type_index

    st_index = source_type_index(cfg.registry_path)

    old_n = len(st.session_state.get("evidence_list", []))
    for i in range(old_n + 6):
        for pfx in ("ev_", "mod_", "src_"):
            st.session_state.pop(f"{pfx}{i}", None)

    new_evs = [
        {
            "text":        ev.get("text", ""),
            "source_id":   ev.get("source_id", ""),
            "source_type": st_index.get(ev.get("source_id", ""), "unknown"),
            "modality":    ev.get("modality", "web_text"),
        }
        for ev in rec.get("evidence", [])[:4]
    ] or [blank_ev()]

    st.session_state["_pending_claim"]     = rec["claim"]
    st.session_state["last_claim"]         = rec["claim"]
    st.session_state["evidence_list"]      = new_evs
    st.session_state["_random_true_label"] = rec.get("verdict", {}).get("label")
    st.session_state["current_claim_id"]   = rec.get("id")
    for i, ev in enumerate(new_evs):
        st.session_state[f"ev_{i}"]  = ev["text"]
        st.session_state[f"mod_{i}"] = ev["modality"]
        st.session_state[f"src_{i}"] = ev["source_type"]


def load_random_example(cfg: "AppConfig") -> None:
    from app_update.core.loaders import load_test_records
    records = load_test_records(cfg.training_jsonl, cfg.splits_dir)
    if not records:
        st.warning("Test data not found.")
        return
    load_record_into_state(random.choice(records), cfg)


def load_by_id(claim_id: str, cfg: "AppConfig") -> bool:
    """Load record by ID into session state. Returns True on success."""
    from app_update.core.loaders import load_all_records_indexed
    idx_map = load_all_records_indexed(cfg.training_jsonl)
    rec = idx_map.get(claim_id.strip())
    if rec is None:
        return False
    load_record_into_state(rec, cfg)
    return True
