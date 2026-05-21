"""TabDef — reusable tab descriptor used for both top-level and sub-tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.config import AppConfig


@dataclass
class TabDef:
    key:    str
    label:  str
    icon:   str
    render: Callable[["AppConfig"], None]
