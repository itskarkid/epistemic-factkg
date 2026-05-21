"""Tab registry for app_update.

get_tabs(cfg) returns ordered list of TabDef from config.yaml tab_defs.
Each tab module exposes def render(cfg: AppConfig) -> None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app_update.core.tab import TabDef

if TYPE_CHECKING:
    from app_update.config import AppConfig

import app_update.tabs.verify    as _verify
import app_update.tabs.evaluate  as _evaluate
import app_update.tabs.reports   as _reports
import app_update.tabs.dataset   as _dataset
import app_update.tabs.pipeline  as _pipeline
import app_update.tabs.registry  as _registry
import app_update.tabs.reference as _reference

_RENDER = {
    "verify":    _verify.render,
    "evaluate":  _evaluate.render,
    "reports":   _reports.render,
    "dataset":   _dataset.render,
    "pipeline":  _pipeline.render,
    "registry":  _registry.render,
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
