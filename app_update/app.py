from __future__ import annotations
from turtle import width

import streamlit as st

from style import CSS
from tabs import TABS, render_tab

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Epistemic FactKG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Header (no sidebar) ───────────────────────────────────────────────────
    st.markdown(
        '<div class="page-header">'
        '<span class="page-title">Epistemic Claim Verifier</span>'
        '<span class="page-badge">Pramana · Neuro-Symbolic</span>'
        '</div>',
        unsafe_allow_html=True,
        width='content',
    )
    # ── Tab layout ────────────────────────────────────────────────────────────
    tab_labels = [t.label for t in TABS]
    rendered = st.tabs(tab_labels)

    for tab_ctx, tab_def in zip(rendered, TABS):
        with tab_ctx:
            render_tab(tab_def)


if __name__ == "__main__":
    main()
