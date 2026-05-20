"""Tabs registry for the updated app.

Each tab is declared as a ``TabDef`` and added to the ``TABS`` list.
``render_tab()`` dispatches to the correct render function.

To add a new tab:
1. Create ``tabs/<name>.py`` with a ``render()`` function.
2. Import it here and add a ``TabDef`` entry to ``TABS``.
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from typing import Callable

from tabs import dataset as _dataset


@dataclass(frozen=True)
class TabDef:
    """Metadata + render callback for a single tab."""
    key: str            # machine-readable identifier
    label: str          # displayed on the tab button (no emoji)
    description: str    # shown in placeholder until the tab is implemented
    render: Callable[[], None]


# ── Placeholder renderer (used until a tab is implemented) ───────────────────

def _placeholder(name: str, description: str) -> None:
    st.markdown(f"### {name}")
    st.caption(description)
    st.info("Under construction — implementation coming next.")


def _make_placeholder(key: str, description: str) -> Callable[[], None]:
    def _render() -> None:
        _placeholder(key.title(), description)
    _render.__name__ = f"render_{key}"
    return _render


# ── Tab registry (order = display order) ─────────────────────────────────────

TABS: list[TabDef] = [
    TabDef(
        key="dataset",
        label="Dataset",
        description="Data quality, distributions, schema viewer, claim browser.",
        render=_dataset.render,
    ),
    TabDef(
        key="model",
        label="Model",
        description="Checkpoint browser, registry info, cache controls.",
        render=_make_placeholder("model", "Checkpoint browser, registry info, cache controls."),
    ),
    TabDef(
        key="evaluation",
        label="Evaluation",
        description="Batch evaluation on test set. Multi-model selector.",
        render=_make_placeholder("evaluation", "Batch evaluation on test set. Multi-model selector."),
    ),
    TabDef(
        key="report",
        label="Report",
        description="Training loss & accuracy curves, metrics, confusion matrix, plots.",
        render=_make_placeholder("report", "Training loss & accuracy curves, metrics, confusion matrix, plots."),
    ),
    TabDef(
        key="verify",
        label="Verify",
        description="Live claim verification with multi-model selector and evidence cards.",
        render=_make_placeholder("verify", "Live claim verification with multi-model selector and evidence cards."),
    ),
    TabDef(
        key="executor",
        label="Executor",
        description="Run pipeline stages (build · validate · test · train · evaluate · report).",
        render=_make_placeholder("executor", "Run pipeline stages (build · validate · test · train · evaluate · report)."),
    ),
    TabDef(
        key="knowledge",
        label="Knowledge",
        description="Reference, Hypothesis (ADRs), Source Registry.",
        render=_make_placeholder("knowledge", "Reference, Hypothesis (ADRs), Source Registry."),
    ),
]


def render_tab(tab: TabDef) -> None:
    """Call the render function for the given tab."""
    tab.render()


__all__ = ["TabDef", "TABS", "render_tab"]
