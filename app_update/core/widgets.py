"""Reusable Streamlit widget components shared across tabs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig


@dataclass
class ClaimFilter:
    """Filter state returned by render_claim_filter_bar()."""
    text_q:      str  = ""
    claim_id_q:  str  = ""
    verdict_q:   str  = ""
    source_q:    str  = ""
    struct_q:    str  = ""
    max_results: int  = 25


def render_claim_filter_bar(
    all_records: list[dict],
    *,
    key_prefix: str = "cf",
    show_max_results: bool = True,
) -> ClaimFilter:
    """Render a reusable claim search/filter widget.

    Returns a ClaimFilter with the current values of all controls.
    The caller is responsible for applying the filter and deciding when to run
    (e.g. checking a search button or reacting to changes).
    """
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            text_q = st.text_input(
                "Claim text contains",
                key=f"{key_prefix}_text",
                placeholder="e.g. Olympic Games",
            )
        with c2:
            claim_id_q = st.text_input(
                "Claim ID (exact or prefix)",
                key=f"{key_prefix}_id",
                placeholder="e.g. averitec-train-0001",
            )
        with c3:
            verdicts = [""] + sorted({
                r.get("verdict", {}).get("label", "")
                for r in all_records if r.get("verdict")
            })
            verdict_q = st.selectbox("Verdict", verdicts, key=f"{key_prefix}_verdict")

        c4, c5, c6 = st.columns(3)
        with c4:
            sources = [""] + sorted({
                r.get("provenance", {}).get("dataset", "")
                for r in all_records if r.get("provenance")
            })
            source_q = st.selectbox("Source dataset", sources, key=f"{key_prefix}_source")
        with c5:
            structures = [""] + sorted({
                (r.get("reasoning") or {}).get("structural") or ""
                for r in all_records
            } - {""})
            struct_q = st.selectbox(
                "Reasoning structure", structures, key=f"{key_prefix}_struct"
            )
        with c6:
            if show_max_results:
                max_results = st.number_input(
                    "Max results", 5, 200, 25, 5, key=f"{key_prefix}_max"
                )
            else:
                max_results = 25

    return ClaimFilter(
        text_q=text_q,
        claim_id_q=claim_id_q,
        verdict_q=verdict_q,
        source_q=source_q,
        struct_q=struct_q,
        max_results=int(max_results),
    )


def apply_claim_filter(records: list[dict], f: ClaimFilter) -> list[dict]:
    """Apply a ClaimFilter to a list of records, returning matching records."""
    out = records
    if f.text_q.strip():
        q = f.text_q.strip().lower()
        out = [r for r in out if q in r.get("claim", "").lower()]
    if f.claim_id_q.strip():
        q = f.claim_id_q.strip().lower()
        out = [r for r in out if r.get("id", "").lower().startswith(q)]
    if f.verdict_q:
        out = [r for r in out if r.get("verdict", {}).get("label") == f.verdict_q]
    if f.source_q:
        out = [r for r in out if r.get("provenance", {}).get("dataset") == f.source_q]
    if f.struct_q:
        out = [r for r in out if (r.get("reasoning") or {}).get("structural") == f.struct_q]
    return out
