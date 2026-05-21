"""Schema sub-tab — unified v3.0 record schema viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig


def render(cfg: "AppConfig") -> None:
    try:
        from src.epistemic.schema import CLAIM_SCHEMA
        schema = CLAIM_SCHEMA
    except Exception as e:
        st.error(f"Could not load schema: {e}")
        return

    version = schema.get("properties", {}).get("schema_version", {}).get("const", "?")
    st.markdown(f"**Epistemic FactKG Schema**  v{version}")
    st.caption("Single source of truth — src/epistemic/schema.py")

    required = schema.get("required", [])
    props    = schema.get("properties", {})

    rows     = []
    has_desc = False
    for field, defn in props.items():
        ftype = defn.get("type", "—")
        if isinstance(ftype, list):
            ftype = " | ".join(ftype)
        fdesc = defn.get("description", "")[:120]
        if fdesc:
            has_desc = True
        freq = "✓" if field in required else ""
        rows.append({"Field": field, "Type": ftype, "Required": freq, "Description": fdesc})

    import pandas as pd
    df = pd.DataFrame(rows).set_index("Field")
    if not has_desc:
        df = df.drop(columns=["Description"])

    col_json, col_table, = st.columns([4, 2])

    with col_json:
        with st.expander("View full JSON Schema", expanded=True):
            st.json(schema, expanded=2)

    with col_table:
        st.dataframe(df, width='stretch', height=460)

