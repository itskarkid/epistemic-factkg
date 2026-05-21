"""app entry point — thin Streamlit shell."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config import get_config
from app.style import CSS
from app.tabs import get_tabs


def main() -> None:
    cfg = get_config()

    st.set_page_config(
        page_title="Epistemic FactKG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="page-header">'
        '<span class="page-title">🧠 Epistemic FactKG</span>'
        '<span class="page-badge">Pramana · Neuro-Symbolic</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    tabs = get_tabs(cfg)
    widgets = st.tabs([f"{t.icon} {t.label}" for t in tabs])
    for widget, tdef in zip(widgets, tabs):
        with widget:
            tdef.render(cfg)


if __name__ == "__main__":
    main()
