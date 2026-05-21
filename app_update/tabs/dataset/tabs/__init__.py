"""Sub-tab definitions for the Dataset tab."""

from __future__ import annotations

from app_update.core.tab import TabDef
from app_update.tabs.dataset.tabs.stats   import render as _render_stats
from app_update.tabs.dataset.tabs.records import render as _render_records
from app_update.tabs.dataset.tabs.schema  import render as _render_schema

TABS: list[TabDef] = [
    TabDef(key="stats",   label="Stats",   icon="📊", render=_render_stats),
    TabDef(key="records", label="Records", icon="🔎", render=_render_records),
    TabDef(key="schema",  label="Schema",  icon="📋", render=_render_schema),
]
