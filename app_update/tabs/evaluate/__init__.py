"""Evaluate tab — single-model and multi-model comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig

from app_update.core.tab import TabDef


def render(cfg: "AppConfig") -> None:
    from app_update.tabs.evaluate.tabs import TABS
    widgets = st.tabs([f"{t.icon} {t.label}" for t in TABS])
    for widget, tdef in zip(widgets, TABS):
        with widget:
            tdef.render(cfg)
