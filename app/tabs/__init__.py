"""Tab registry for app.

get_tabs(cfg) returns ordered list of TabDef from config.yaml tab_defs.
Each tab module exposes def render(cfg: AppConfig) -> None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.tab import TabDef

if TYPE_CHECKING:
    from app.config import AppConfig

import app.tabs.verify     as _verify
import app.tabs.evaluate   as _evaluate
import app.tabs.reports    as _reports
import app.tabs.dataset    as _dataset
import app.tabs.pipeline   as _pipeline
import app.tabs.reference  as _reference
import app.tabs.probe_lab  as _probe_lab

_RENDER = {
    "verify":    _verify.render,
    "evaluate":  _evaluate.render,
    "probe_lab": _probe_lab.render,
    "reports":   _reports.render,
    "dataset":   _dataset.render,
    "pipeline":  _pipeline.render,
    "reference": _reference.render,
}


def get_tabs(cfg: "AppConfig") -> list[TabDef]:
    """Build ordered tab list from config.yaml app.tabs (icon from YAML; label from key)."""
    return [
        TabDef(
            key=t["key"],
            label=t["key"].title(),
            icon=t["icon"],
            render=_RENDER[t["key"]],
        )
        for t in cfg.tab_defs
        if t["key"] in _RENDER
    ]
