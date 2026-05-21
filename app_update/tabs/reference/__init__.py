"""Reference tab — ADRs, schema docs, pramana framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig


def render(cfg: "AppConfig") -> None:
    st.info("Reference tab — under construction.")
