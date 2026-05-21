"""Dataset tab — data quality, distributions, schema viewer, claim browser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig


def render(cfg: "AppConfig") -> None:
    from app.tabs.dataset.tabs import TABS
    widgets = st.tabs([f"{t.icon} {t.label}" for t in TABS])
    for widget, tdef in zip(widgets, TABS):
        with widget:
            tdef.render(cfg)
