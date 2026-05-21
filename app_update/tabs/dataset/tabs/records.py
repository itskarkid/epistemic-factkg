"""Records sub-tab — search and browse individual claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig

_ICONS = {
    "supported":            "✅",
    "refuted":              "❌",
    "not_enough_evidence":  "~",
    "conflicting_evidence": "⚠️",
}


def render(cfg: "AppConfig") -> None:
    from app_update.core.loaders import load_all_records_list
    from app_update.core.state import load_record_into_state
    from app_update.core.widgets import render_claim_filter_bar, apply_claim_filter

    all_records = load_all_records_list(cfg.training_jsonl)
    if not all_records:
        st.info("Training JSONL not found.")
        return

    filt = render_claim_filter_bar(all_records, key_prefix="rec")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run = st.button("Search", type="primary", key="rec_run", use_container_width=True)
    with col_info:
        st.caption(f"{len(all_records):,} total records")

    if not run:
        return

    filtered = apply_claim_filter(all_records, filt)
    showing  = min(len(filtered), filt.max_results)
    st.caption(f"{len(filtered):,} matching · showing {showing}")

    if not filtered:
        st.info("No records match the current filters.")
        return

    for rec in filtered[:filt.max_results]:
        verdict_label = rec.get("verdict", {}).get("label", "?")
        claim_id      = rec.get("id", "—")
        dataset       = rec.get("provenance", {}).get("dataset", "?")
        structural    = (rec.get("reasoning") or {}).get("structural", "")
        claim_text    = rec.get("claim", "")
        evs           = rec.get("evidence", [])
        n_ev          = len(evs)
        icon          = _ICONS.get(verdict_label, "?")

        is_vals = [
            e.get("inference_strength")
            for e in evs
            if e.get("inference_strength") is not None
        ]
        avg_is = sum(is_vals) / len(is_vals) if is_vals else None

        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            tags = f"`{dataset}`"
            if structural:
                tags += f" · `{structural}`"
            h1.markdown(f"{icon} **{claim_id}**  {tags}")
            h2.markdown(f"`{verdict_label}`")

            st.markdown(claim_text[:130] + ("…" if len(claim_text) > 130 else ""))

            f1, f2 = st.columns([4, 1])
            ev_info = f"{n_ev} evidence"
            if avg_is is not None:
                ev_info += f" · IS avg {avg_is:.2f}"
            f1.caption(ev_info)
            if f2.button("Load in Verify", key=f"rec_load_{claim_id}", use_container_width=True):
                load_record_into_state(rec, cfg)
                st.success("Loaded — switch to Verify tab")
