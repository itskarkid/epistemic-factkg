"""Dataset tab -- Insights | Claim Search | Schema.

Loads from the unified JSONL (out/data/unified/epistemic_factkg.jsonl).
No raw-source imports; no src/ imports.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Optional

import pandas as pd
import streamlit as st

from config import UNIFIED_JSONL, VERDICT_COLORS

# -- Field documentation (shown in Schema sub-tab) ----------------------------
_SCHEMA_FIELDS: list[dict] = [
    {"field": "schema_version", "type": "string",  "description": "Schema version, e.g. '3.0'"},
    {"field": "id",             "type": "string",  "description": "Unique claim identifier"},
    {"field": "claim",          "type": "string",  "description": "Natural language claim text"},
    {"field": "verdict",        "type": "object",  "description": "label (supported|refuted|not_enough_evidence), justification, derivation_method"},
    {"field": "epistemic",      "type": "object",  "description": "evidence_types_all (list), assignment_method"},
    {"field": "claim_triples",  "type": "array",   "description": "KG triples [[subject_uri, predicate_uri, object], ...]"},
    {"field": "reasoning",      "type": "object",  "description": "structural (one_hop|conjunction|negation|absence), strategy"},
    {"field": "evidence",       "type": "array",   "description": "Per-evidence: evidence_id, text, triples, modality, stance, evidence_types, source_id, inference_strength, source_url"},
    {"field": "provenance",     "type": "object",  "description": "dataset (ai2thor|averitec|synthetic), split, context_id"},
    {"field": "meta",           "type": "object",  "description": "schema_version, created_utc, template_type (synthetic only), is_shortcut_breaking (synthetic only)"},
]

_EV_FIELDS: list[dict] = [
    {"field": "evidence_id",        "type": "string",  "description": "Unique evidence identifier within the claim"},
    {"field": "text",               "type": "string",  "description": "Evidence sentence text"},
    {"field": "triples",            "type": "array",   "description": "KG triples extracted from this evidence"},
    {"field": "triple_source",      "type": "string",  "description": "ground_truth | extracted"},
    {"field": "modality",           "type": "string",  "description": "sensor | web_text | video | audio | image | pdf | web_table | annotator_knowledge"},
    {"field": "stance",             "type": "string",  "description": "supports | refutes | neutral"},
    {"field": "evidence_types",     "type": "array",   "description": "perception | testimony | non_apprehension | comparison_analogy | inference | postulation_derivation"},
    {"field": "source_id",          "type": "string",  "description": "Identifier linking to source trust registry"},
    {"field": "inference_strength", "type": "float",   "description": "IS prior seed [0.1 - 1.0] from ISHead (training label)"},
    {"field": "source_url",         "type": "string?", "description": "Original URL if applicable (nullable)"},
]


# -- Data loading --------------------------------------------------------------

@st.cache_data(show_spinner="Loading unified dataset...")
def _load() -> Optional[list[dict]]:
    if not UNIFIED_JSONL.exists():
        return None
    records: list[dict] = []
    for line in UNIFIED_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        verdict_obj   = r.get("verdict") or {}
        verdict       = (verdict_obj.get("label") if isinstance(verdict_obj, dict) else verdict_obj or "unknown").lower()
        justification = (verdict_obj.get("justification") or "") if isinstance(verdict_obj, dict) else ""
        reasoning_obj = r.get("reasoning") or {}
        reasoning_str = (
            f"{reasoning_obj.get('structural', '')} / {reasoning_obj.get('strategy', '')}"
            if isinstance(reasoning_obj, dict) else str(reasoning_obj)
        ).strip(" /")
        records.append({
            "id":            r.get("id", ""),
            "claim":         r.get("claim", ""),
            "verdict":       verdict,
            "source":        (r.get("provenance") or {}).get("dataset", "--"),
            "split":         (r.get("provenance") or {}).get("split") or "--",
            "justification": justification,
            "reasoning":     reasoning_str,
        })
    return records


# -- Sub-tab: Insights ---------------------------------------------------------

def _render_insights(records: list[dict]) -> None:
    total      = len(records)
    by_source  = Counter(r["source"]  for r in records)
    by_verdict = Counter(r["verdict"] for r in records)

    cols = st.columns(4)
    cols[0].metric("Total Claims", f"{total:,}")
    cols[1].metric("AI2Thor",   f"{by_source.get('ai2thor',   0):,}", "sensor / perception")
    cols[2].metric("Synthetic", f"{by_source.get('synthetic', 0):,}", "NEI-heavy")
    cols[3].metric("AVeritec",  f"{by_source.get('averitec',  0):,}", "web text")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Verdict distribution")
        df_v = pd.DataFrame(
            [{"Verdict": k.replace("_", " ").title(), "Count": v}
             for k, v in sorted(by_verdict.items(), key=lambda x: -x[1])]
        )
        st.bar_chart(df_v.set_index("Verdict"), height=200, use_container_width=True)

    with c2:
        st.markdown("##### Source distribution")
        df_s = pd.DataFrame(
            [{"Source": k.title(), "Count": v}
             for k, v in sorted(by_source.items(), key=lambda x: -x[1])]
        )
        st.bar_chart(df_s.set_index("Source"), height=200, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Verdict x Source")
    rows = [{"Verdict": r["verdict"].replace("_", " ").title(), "Source": r["source"].title()} for r in records]
    pivot = pd.DataFrame(rows).pivot_table(index="Verdict", columns="Source", aggfunc="size", fill_value=0)
    st.dataframe(pivot, use_container_width=True)


# -- Sub-tab: Claim Search -----------------------------------------------------

def _render_claim_search(records: list[dict]) -> None:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 4])
    with f1:
        src = st.selectbox("Source",  ["All"] + sorted({r["source"]  for r in records}), key="ds_src")
    with f2:
        vd  = st.selectbox("Verdict", ["All"] + sorted({r["verdict"] for r in records}), key="ds_vd")
    with f3:
        sp  = st.selectbox("Split",   ["All"] + sorted({r["split"]   for r in records} - {"--"}) + ["--"], key="ds_sp")
    with f4:
        q   = st.text_input("Search claim or ID", placeholder="keyword...", key="ds_q")

    filtered = records
    if src != "All": filtered = [r for r in filtered if r["source"]  == src]
    if vd  != "All": filtered = [r for r in filtered if r["verdict"] == vd]
    if sp  != "All": filtered = [r for r in filtered if r["split"]   == sp]
    if q:
        ql = q.lower()
        filtered = [r for r in filtered if ql in r["claim"].lower() or ql in r["id"].lower()]

    st.caption(f"{len(filtered):,} claims")

    if not filtered:
        st.info("No claims match the current filters.")
        return

    df = pd.DataFrame(filtered)[["id", "claim", "verdict", "source", "reasoning", "justification"]]
    st.dataframe(
        df,
        use_container_width=True,
        height=460,
        column_config={
            "id":            st.column_config.TextColumn("Claim ID",      width="medium"),
            "claim":         st.column_config.TextColumn("Claim",         width="large"),
            "verdict":       st.column_config.TextColumn("Verdict",       width="small"),
            "source":        st.column_config.TextColumn("Source",        width="small"),
            "reasoning":     st.column_config.TextColumn("Reasoning",     width="medium"),
            "justification": st.column_config.TextColumn("Justification", width="large"),
        },
        hide_index=True,
    )


# -- Sub-tab: Schema -----------------------------------------------------------

def _render_schema() -> None:
    st.markdown("##### Unified claim record -- field reference")
    st.caption("Schema v3.0  All fields present in every record unless noted 'synthetic only'.")
    st.dataframe(
        pd.DataFrame(_SCHEMA_FIELDS),
        use_container_width=True, hide_index=True,
        column_config={
            "field":       st.column_config.TextColumn("Field",       width="medium"),
            "type":        st.column_config.TextColumn("Type",        width="small"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
    )
    st.markdown("---")
    st.markdown("##### Evidence sub-object fields")
    st.dataframe(
        pd.DataFrame(_EV_FIELDS),
        use_container_width=True, hide_index=True,
        column_config={
            "field":       st.column_config.TextColumn("Field",       width="medium"),
            "type":        st.column_config.TextColumn("Type",        width="small"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
    )


# -- Entry point ---------------------------------------------------------------

def render() -> None:
    records = _load()

    if records is None:
        st.warning("Unified dataset not found. Run `just build` to generate it.", icon="warning")
        return

    tab_insights, tab_search, tab_schema = st.tabs(["Insights", "Claim Search", "Schema"])

    with tab_insights:
        _render_insights(records)
    with tab_search:
        _render_claim_search(records)
    with tab_schema:
        _render_schema()
