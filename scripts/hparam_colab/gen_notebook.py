"""Generate notebooks/hparam_colab.ipynb — the Colab GPU hparam notebook."""
from __future__ import annotations

import json
from pathlib import Path

_NB_PATH = Path("notebooks/hparam_colab.ipynb")

# ── Cell helpers ──────────────────────────────────────────────────────────────

def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ── Notebook cells ────────────────────────────────────────────────────────────

def _cells(models: list[str], n_trials: int) -> list[dict]:
    jsonl = "out/data/training/epistemic_factkg_training.jsonl"
    splits = "out/data/splits"

    cells = [
        _md(
            "# Epistemic FactKG — Hyperparameter Search (Colab GPU)\n"
            "\n"
            "**Runtime → Change runtime type → T4 GPU** before running.\n\n"
            f"Models: {' → '.join(models)}  ·  {n_trials} trials each  \n"
            "Pre-requisite: `colab_upload.zip` uploaded to Google Drive root  \n"
            "(`just colab-prep` on your local machine generates it)."
        ),
        _code(
            "# ── Cell 1: Mount Drive + extract project ────────────────────\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "\n"
            "import zipfile, pathlib, os\n"
            "\n"
            "ZIP     = pathlib.Path('/content/drive/MyDrive/colab_upload.zip')\n"
            "PROJECT = pathlib.Path('/content/epistemic-factkg')\n"
            "\n"
            "if not PROJECT.exists():\n"
            "    print('Extracting...')\n"
            "    PROJECT.mkdir(parents=True)\n"
            "    with zipfile.ZipFile(ZIP) as zf:\n"
            "        zf.extractall(PROJECT)\n"
            "    print('Done.')\n"
            "else:\n"
            "    print('Already extracted.')\n"
            "\n"
            "os.chdir(PROJECT)\n"
            "print('cwd:', os.getcwd())"
        ),
        _code(
            "# ── Cell 2: Install uv + Python 3.14 + dependencies ──────────\n"
            "!pip install uv -q\n"
            "!uv python install 3.14\n"
            "!uv sync --python 3.14 -q\n"
            "\n"
            "# Verify GPU\n"
            "!uv run python -c \""
            "import torch; "
            "print('CUDA:', torch.cuda.is_available(), "
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\""
        ),
        _code(
            "# ── Cell 3: Verify required data files ───────────────────────\n"
            "import pathlib\n"
            "\n"
            "required = [\n"
            f"    '{jsonl}',\n"
            f"    '{splits}/train_indices.json',\n"
            f"    '{splits}/val_indices.json',\n"
            f"    '{splits}/test_indices.json',\n"
            "    'data/registry/source_trust_registry.jsonl',\n"
            "]\n"
            "all_ok = True\n"
            "for p in required:\n"
            "    ok = pathlib.Path(p).exists()\n"
            "    print(('✅' if ok else '❌'), p)\n"
            "    if not ok: all_ok = False\n"
            "\n"
            "for d in ['out/model/graphs', 'configs/hparams']:\n"
            "    pathlib.Path(d).mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "assert all_ok, 'Some data files are missing — re-run just colab-prep and re-upload.'"
        ),
    ]

    for i, model in enumerate(models):
        note = " (downloads NLI model ~250 MB on first run)" if model == "v3-nli" else ""
        cells.append(_code(
            f"# ── Cell {4 + i}: {model} hparam search{note} ────────────────\n"
            f"!uv run python -m src.pipeline.model.hparam_search \\\\\n"
            f"    --model {model} \\\\\n"
            f"    --jsonl {jsonl} \\\\\n"
            f"    --splits-dir {splits} \\\\\n"
            f"    --n-trials {n_trials} \\\\\n"
            f"    --batch-size 32"
        ))

        # After v2-hgnn, insert cache copy cell (v1-hgnn and baseline reuse it)
        if model == "v2-hgnn":
            cells.append(_code(
                f"# ── Cell {4 + i + 1}: Copy graph cache → v1-hgnn + baseline ──\n"
                "# v1-hgnn and baseline use identical GraphConfig.v1() with no NLI.\n"
                "# Reusing the v2-hgnn cache skips ~10 min of graph building per model.\n"
                "import shutil, pathlib\n"
                "\n"
                "src = pathlib.Path('out/model/graphs/split_cache_v2-hgnn.pkl')\n"
                "for m in ['v1-hgnn', 'baseline']:\n"
                "    dst = pathlib.Path(f'out/model/graphs/split_cache_{m}.pkl')\n"
                "    if src.exists() and not dst.exists():\n"
                "        shutil.copy(src, dst)\n"
                "        print(f'Copied cache → {dst}')\n"
                "    elif dst.exists():\n"
                "        print(f'Cache exists: {dst}')\n"
                "    else:\n"
                "        print(f'WARN: source cache not found — graph will be rebuilt')"
            ))

    cells += [
        _code(
            f"# ── Cell {4 + len(models) + 1}: Results summary ─────────────────\n"
            "import json, pathlib\n"
            "\n"
            f"for model in {models!r}:\n"
            "    p = pathlib.Path(f'configs/hparams/{model}_best_hparams.json')\n"
            "    if p.exists():\n"
            "        d = json.loads(p.read_text())\n"
            "        print(f'\\n── {model}  val_loss={d.get(\"_best_val_loss\")}')\n"
            "        for k, v in d.items():\n"
            "            if not k.startswith('_'):\n"
            "                print(f'  {k}: {v}')\n"
            "    else:\n"
            "        print(f'❌  {model}: not found')"
        ),
        _code(
            f"# ── Cell {4 + len(models) + 2}: Save hparam files back to Drive ──\n"
            "import shutil, pathlib\n"
            "\n"
            "OUT = pathlib.Path('/content/drive/MyDrive/epistemic-factkg-hparams')\n"
            "OUT.mkdir(exist_ok=True)\n"
            "\n"
            "for f in pathlib.Path('configs/hparams').glob('*_best_hparams.json'):\n"
            "    shutil.copy(f, OUT / f.name)\n"
            "    print(f'Saved → {OUT / f.name}')\n"
            "\n"
            "print('\\nDone. Copy these files back to your local configs/hparams/ folder.')"
        ),
    ]

    return cells


def build(
    models: list[str] | None = None,
    n_trials: int = 30,
    output: Path | None = None,
) -> Path:
    """Write the Colab notebook and return its path."""
    models = models or ["v3-nli", "v2-hgnn", "v1-hgnn", "baseline"]
    output = output or _NB_PATH

    nb = {
        "cells": _cells(models, n_trials),
        "metadata": {
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {output}")
    return output


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    build(models=args.models, n_trials=args.n_trials)
