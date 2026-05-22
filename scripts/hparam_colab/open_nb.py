"""Open a notebook file with the system default app, WSL-aware."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False


def open_notebook(nb_path: Path) -> None:
    nb_path = nb_path.resolve()
    try:
        if sys.platform == "win32":
            os.startfile(str(nb_path))
        elif _is_wsl():
            win_path = subprocess.check_output(["wslpath", "-w", str(nb_path)]).decode().strip()
            subprocess.run(["explorer.exe", win_path], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(nb_path)], check=False)
        else:
            subprocess.run(["xdg-open", str(nb_path)], check=False)
    except Exception:
        print(f"Could not open notebook automatically. Open it manually:\n  {nb_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    args = ap.parse_args()
    open_notebook(args.notebook)
