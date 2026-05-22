"""Environment-aware hparam search launcher.

Usage (via Justfile):
    just hparam-search v3-nli
    just hparam-search v2-hgnn 50

Usage (direct):
    python scripts/hparam_colab/run.py --model v3-nli --n-trials 30

Behaviour:
    Colab / local GPU  → runs hparam_search directly in this process
    Local CPU-only     → generates the Colab notebook, builds the upload zip,
                         opens the notebook, and prints upload instructions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so src/ imports work when invoked directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.hparam_colab.detect import Env, describe, detect


def _run_search(model: str, n_trials: int) -> None:
    """Run hparam search directly (Colab or local GPU path)."""
    import json
    import pickle
    import torch
    from torch_geometric.loader import DataLoader

    from src.epistemic.registry import load_source_trust_registry
    from src.model.config import GraphConfig
    from src.model.data.builder import ClaimGraphBuilder
    from src.model.data.featurizer import Featurizer
    from src.model.data.types import NUM_STANCE, NUM_VERDICT
    from src.model.models import MODELS
    from src.model.models.nlihybridhgnn import NLIHybridHGNN
    from src.pipeline.model.hparam_search import run_search

    jsonl_path  = _ROOT / "out/data/training/epistemic_factkg_training.jsonl"
    splits_dir  = _ROOT / "out/data/splits"
    cache_path  = _ROOT / f"out/model/graphs/split_cache_{model}.pkl"
    embed_cache = _ROOT / "out/model/graphs/embed_cache.pkl"
    registry_p  = _ROOT / "data/registry/source_trust_registry.jsonl"

    is_nli    = MODELS.get(model) is NLIHybridHGNN
    graph_cfg = GraphConfig.v2() if is_nli else GraphConfig.v1()

    # ── Load or build graph cache ─────────────────────────────────────────
    if cache_path.exists():
        print(f"Loading graph cache: {cache_path}")
        cached       = pickle.loads(cache_path.read_bytes())
        train_graphs = cached["train"]
        val_graphs   = cached["val"]
    else:
        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        def _load_idx(split: str) -> list[int]:
            return json.loads((splits_dir / f"{split}_indices.json").read_text())["indices"]

        featurizer = Featurizer(cache_path=str(embed_cache))
        registry   = load_source_trust_registry(str(registry_p))
        builder    = ClaimGraphBuilder(registry, featurizer, use_nli=is_nli)

        def _build(indices: list[int], split: str) -> list:
            graphs = []
            for idx in indices:
                try:
                    g = builder.build(records[idx])
                    if g is not None:
                        graphs.append(g.data)
                except Exception:
                    pass
            print(f"{split}: {len(graphs)} graphs built")
            return graphs

        train_graphs = _build(_load_idx("train"), "train")
        val_graphs   = _build(_load_idx("val"),   "val")
        featurizer.save_cache()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps({"train": train_graphs, "val": val_graphs}))

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=32, shuffle=False)

    all_stance_y  = torch.cat([g["evidence"].stance_y for g in train_graphs])
    s_counts      = torch.bincount(all_stance_y,  minlength=NUM_STANCE).float().clamp(min=1.0)
    stance_weights = s_counts.sum() / (NUM_STANCE * s_counts)

    all_verdict_y  = torch.cat([g["claim"].y for g in train_graphs])
    v_counts       = torch.bincount(all_verdict_y, minlength=NUM_VERDICT).float().clamp(min=1.0)
    verdict_weights = v_counts.sum() / (NUM_VERDICT * v_counts)

    (_ROOT / "configs/hparams").mkdir(parents=True, exist_ok=True)

    run_search(
        model_key=model,
        graph_cfg=graph_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        stance_weights=stance_weights,
        verdict_weights=verdict_weights,
        n_trials=n_trials,
    )


def _colab_fallback(model: str, n_trials: int) -> None:
    """CPU-only path: generate notebook + zip, then open notebook."""
    from scripts.hparam_colab.gen_notebook import build as gen_nb
    from scripts.hparam_colab.prep import build as gen_zip

    print("\n No CUDA GPU detected — preparing Colab package...\n")

    nb_path  = gen_nb(n_trials=n_trials)
    zip_path = gen_zip(root=_ROOT)

    print(
        f"\n{'─' * 60}\n"
        f" Next steps:\n"
        f"  1. Upload  {zip_path.name}  to your Google Drive root\n"
        f"  2. Open    {nb_path}  in Google Colab\n"
        f"     (colab.research.google.com → File → Upload notebook)\n"
        f"  3. Runtime → Change runtime type → T4 GPU\n"
        f"  4. Runtime → Run all\n"
        f"  5. Results will be saved to Drive: epistemic-factkg-hparams/\n"
        f"     Copy them back to  configs/hparams/\n"
        f"{'─' * 60}\n"
    )

    from scripts.hparam_colab.open_nb import open_notebook
    open_notebook(nb_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Environment-aware hparam search launcher.")
    ap.add_argument("--model",       default="v3-nli",
                    choices=["v3-nli", "v2-hgnn", "v1-hgnn", "baseline"])
    ap.add_argument("--n-trials",    type=int, default=30)
    ap.add_argument("--force-colab", action="store_true",
                    help="Always use the Colab notebook path, even when a local GPU is available.")
    args = ap.parse_args()

    env = detect()
    print(f"[hparam-search] env={env.name}  model={args.model}  trials={args.n_trials}")
    print(f"[hparam-search] {describe(env)}\n")

    if args.force_colab or env == Env.COLAB:
        _colab_fallback(args.model, args.n_trials)
    elif env == Env.LOCAL_GPU:
        _run_search(args.model, args.n_trials)
    else:
        _colab_fallback(args.model, args.n_trials)


if __name__ == "__main__":
    main()
