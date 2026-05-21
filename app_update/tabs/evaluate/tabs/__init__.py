"""Sub-tab definitions for the Evaluate tab."""

from __future__ import annotations

from app_update.core.tab import TabDef
from app_update.tabs.evaluate.tabs.single  import render as _render_single
from app_update.tabs.evaluate.tabs.compare import render as _render_compare
from app_update.tabs.evaluate.tabs.graphs  import render as _render_graphs

TABS: list[TabDef] = [
    TabDef(key="single",  label="Single Model", icon="🎯", render=_render_single),
    TabDef(key="compare", label="Compare All",  icon="⚖️", render=_render_compare),
    TabDef(key="graphs",  label="Graph Browser", icon="🌐", render=_render_graphs),
]
