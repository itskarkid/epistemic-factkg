"""Style package — collects and combines all CSS files in this folder.

Usage (in app.py):
    from style import CSS
    st.markdown(CSS, unsafe_allow_html=True)

Files are loaded in alphabetical order so later files can override earlier ones.
Add new ``*.css`` files to this folder; they are picked up automatically.
"""

from __future__ import annotations

from pathlib import Path

_STYLE_DIR = Path(__file__).parent

# Collect every .css file in the folder, sorted for deterministic order
_css_parts: list[str] = [
    p.read_text(encoding="utf-8")
    for p in sorted(_STYLE_DIR.glob("*.css"))
]

# Single injectable string ready for st.markdown
CSS: str = "<style>\n" + "\n\n".join(_css_parts) + "\n</style>"

__all__ = ["CSS"]
