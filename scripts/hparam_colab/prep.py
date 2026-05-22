"""Build colab_upload.zip — all files Colab needs to run hparam search."""
from __future__ import annotations

import zipfile
from pathlib import Path

# Files / glob patterns to include in the zip
_INCLUDE: list[str] = [
    "src/**/*.py",
    "configs/**/*",
    "data/registry/*.jsonl",
    "pyproject.toml",
    "README.md",
    "notebooks/hparam_colab.ipynb",
]

# Data files that may be large — included only if they exist
_DATA_FILES: list[str] = [
    "out/data/training/epistemic_factkg_training.jsonl",
    "out/data/splits/train_indices.json",
    "out/data/splits/val_indices.json",
    "out/data/splits/test_indices.json",
]

_OUTPUT = Path("colab_upload.zip")


def build(root: Path | None = None, output: Path | None = None) -> Path:
    """Create colab_upload.zip in *root* and return its path."""
    root   = root   or Path(".")
    output = output or (root / _OUTPUT)

    collected: list[Path] = []

    for pattern in _INCLUDE:
        collected.extend(p for p in root.glob(pattern) if p.is_file())

    for rel in _DATA_FILES:
        p = root / rel
        if p.exists():
            collected.append(p)
        else:
            print(f"  WARN  missing data file (skipped): {rel}")

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in collected:
            arcname = p.relative_to(root)
            zf.write(p, arcname)

    total_mb = output.stat().st_size / 1_048_576
    print(f"Created {output}  ({len(collected)} files, {total_mb:.1f} MB)")
    print("→ Upload this file to your Google Drive root folder.")
    return output


if __name__ == "__main__":
    build()
