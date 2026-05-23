#!/usr/bin/env python3
"""Train EpistemicHGNN (multi-head neuro-symbolic) on the graph dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.epistemic.registry import load_source_trust_registry
from src.model.config import GraphConfig
from src.model.data.featurizer import Featurizer
from src.model.data.builder import ClaimGraphBuilder
from src.model.models import MODELS
from src.model.models.nlihybridhgnn import NLIHybridHGNN
from src.model.training.config import TrainConfig
from src.model.training.trainer import Trainer
from src.model.data.types import NUM_STANCE, NUM_VERDICT
from src.pipeline.model.hparam_search import load_best_hparams


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train EpistemicHGNN on the filtered training JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--jsonl", required=True, help="Filtered training JSONL")
    ap.add_argument("--model", default="v1-hgnn", help="Model key from MODELS registry")
    ap.add_argument(
        "--model-name", default="v1-hgnn", help="Display name for logs/reports"
    )
    ap.add_argument("--splits-dir", default="out/data/splits")
    ap.add_argument("--checkpoint-dir", default="out/model/v1-hgnn/checkpoints")
    ap.add_argument("--report-dir", default="out/reports/model/v1-hgnn")
    ap.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    ap.add_argument("--embed-cache", default="out/model/graphs/embed_cache.pkl")
    ap.add_argument(
        "--dataset", default=None, help="Unused — kept for Justfile compatibility"
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--ec-threshold", type=float, default=0.35)
    ap.add_argument(
        "--is-loss-weight", type=float, default=0.5, help="λ₁ for IS regression loss"
    )
    ap.add_argument(
        "--verdict-loss-weight",
        type=float,
        default=1.0,
        help="λ₂ for claim-level verdict loss (VerdictHead calibration)",
    )
    ap.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights for both stance and verdict",
    )
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--run-id", default=None, help="Run ID for versioned output (default: ISO timestamp)")
    return ap


def _load_split_indices(splits_dir: Path, split: str) -> list[int]:
    path = splits_dir / f"{split}_indices.json"
    if not path.exists():
        print(f"Error: {path} not found. Run 'just split' first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))["indices"]


def _build_graphs(
    records: list[dict],
    indices: list[int],
    builder: ClaimGraphBuilder,
    split_name: str,
    verbose: bool,
) -> tuple[list, list[str]]:
    graphs = []
    skipped_ids: list[str] = []
    for idx in indices:
        if idx >= len(records):
            skipped_ids.append(str(idx))
            continue
        try:
            g = builder.build(records[idx])
        except Exception:
            skipped_ids.append(records[idx].get("id", str(idx)))
            continue
        if g is None:
            skipped_ids.append(records[idx].get("id", str(idx)))
            continue
        g.data.claim_idx = torch.tensor([idx], dtype=torch.long)
        graphs.append(g.data)
    if verbose:
        print(f"  {split_name}: {len(graphs)} graphs  ({len(skipped_ids)} skipped)")
    return graphs, skipped_ids


def _write_latest_pointer(base_dir: Path, run_id: str) -> None:
    """Point 'latest' at run_id. Uses a symlink; falls back to latest.txt on Windows."""
    latest = base_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_id)
    except OSError:
        (base_dir / "latest.txt").write_text(run_id, encoding="utf-8")


def main() -> None:
    parser = _build_parser()

    # Apply saved hparams as defaults — CLI args always override them.
    # Parse model arg first (without full validation) to look up the right file.
    _pre, _ = parser.parse_known_args()
    hparams = load_best_hparams(_pre.model)
    if hparams:
        parser.set_defaults(**hparams)
        print(f"Loaded hparams for '{_pre.model}' from configs/hparams/{_pre.model}_best_hparams.json")

    args = parser.parse_args()
    # device: None means auto-detect (TrainConfig handles it, but train.py also needs it)
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if hparams:
        _HPARAM_KEYS = ["hidden_dim", "heads", "dropout", "lr", "weight_decay",
                        "is_loss_weight", "ec_threshold"]
        parts = []
        for k in _HPARAM_KEYS:
            val = getattr(args, k, None)
            if k in hparams and val == hparams[k]:
                src = "hparam"
            elif k in hparams:
                src = f"cli-override (hparam was {hparams[k]})"
            else:
                src = "default"
            parts.append(f"  {k}={val}  [{src}]")
        print("Effective training config:\n" + "\n".join(parts))

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(
            f"Error: {jsonl_path} not found. Run 'just filter' first.", file=sys.stderr
        )
        sys.exit(1)

    splits_dir = Path(args.splits_dir)
    train_indices = _load_split_indices(splits_dir, "train")
    val_indices = _load_split_indices(splits_dir, "val")

    # ── Load records + build graphs ───────────────────────────────────────────
    if args.verbose:
        print("Loading JSONL records...")
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    is_nli = MODELS.get(args.model) is NLIHybridHGNN

    # ── Graph cache: skip JSONL rebuild when cache is newer than JSONL + splits ─
    cache_dir = Path("out/model/graphs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_path = cache_dir / f"split_cache_{args.model}.pkl"

    _split_files = [splits_dir / f"{s}_indices.json" for s in ("train", "val")]
    _split_mtime = max(
        f.stat().st_mtime for f in _split_files if f.exists()
    )
    cache_valid = (
        graph_cache_path.exists()
        and graph_cache_path.stat().st_mtime >= jsonl_path.stat().st_mtime
        and graph_cache_path.stat().st_mtime >= _split_mtime
    )

    if cache_valid:
        if args.verbose:
            print(f"Loading graphs from cache: {graph_cache_path}")
        with open(graph_cache_path, "rb") as _f:
            _cached = pickle.load(_f)
        train_graphs = _cached["train"]
        val_graphs = _cached["val"]
        train_skipped_ids = _cached.get("train_skipped_ids", [])
        val_skipped_ids = _cached.get("val_skipped_ids", [])
    else:
        if not graph_cache_path.exists():
            reason = "cache missing"
        elif graph_cache_path.stat().st_mtime < jsonl_path.stat().st_mtime:
            reason = "JSONL newer than cache"
        else:
            reason = "splits newer than cache"
        if args.verbose:
            print(f"Building graphs... ({reason})")
        featurizer = Featurizer(cache_path=args.embed_cache, device=args.device)
        registry = load_source_trust_registry(args.registry)
        builder = ClaimGraphBuilder(registry, featurizer, use_nli=is_nli)

        train_graphs, train_skipped_ids = _build_graphs(records, train_indices, builder, "train", args.verbose)
        val_graphs, val_skipped_ids = _build_graphs(records, val_indices, builder, "val", args.verbose)
        featurizer.save_cache()

        with open(graph_cache_path, "wb") as _f:
            pickle.dump(
                {
                    "train": train_graphs,
                    "val": val_graphs,
                    "train_skipped_ids": train_skipped_ids,
                    "val_skipped_ids": val_skipped_ids,
                },
                _f,
            )
        if args.verbose:
            print(f"Graph cache saved: {graph_cache_path}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    # ── Class weights (inverse-frequency) ────────────────────────────────────
    stance_weights = None
    verdict_weights = None
    if not args.no_class_weights:
        all_stance_y = torch.cat([g["evidence"].stance_y for g in train_graphs])
        s_counts = (
            torch.bincount(all_stance_y, minlength=NUM_STANCE).float().clamp(min=1.0)
        )
        stance_weights = s_counts.sum() / (NUM_STANCE * s_counts)

        all_verdict_y = torch.cat([g["claim"].y for g in train_graphs])
        v_counts = (
            torch.bincount(all_verdict_y, minlength=NUM_VERDICT).float().clamp(min=1.0)
        )
        verdict_weights = v_counts.sum() / (NUM_VERDICT * v_counts)

        if args.verbose:
            print(
                f"Stance  class weights: {[round(w, 3) for w in stance_weights.tolist()]}"
            )
            print(
                f"Verdict class weights: {[round(w, 3) for w in verdict_weights.tolist()]}"
                f"  (supported / refuted / NEI)"
            )

    # ── Model + trainer ───────────────────────────────────────────────────────
    if args.model not in MODELS:
        print(
            f"Unknown model '{args.model}'. Available: {list(MODELS)}", file=sys.stderr
        )
        sys.exit(1)
    graph_cfg = GraphConfig.v2() if is_nli else GraphConfig.v1()
    model = MODELS[args.model](
        graph_cfg, args.hidden_dim, args.heads, args.dropout, args.ec_threshold
    )
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        is_loss_weight=args.is_loss_weight,
        verdict_loss_weight=args.verdict_loss_weight,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        ec_threshold=args.ec_threshold,
    )
    trainer = Trainer(
        model,
        config,
        stance_class_weights=stance_weights,
        verdict_class_weights=verdict_weights,
    )

    print(
        f"Training {args.model_name}  |  "
        f"train={len(train_graphs)}  val={len(val_graphs)}  "
        f"device={args.device}  epochs={args.epochs}"
    )
    history = trainer.fit(train_loader, val_loader, verbose=True)

    # ── Save history ──────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    base_report_dir = Path(args.report_dir)
    report_dir = base_report_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "data_coverage": {
            "train_total": len(train_indices),
            "train_graphs": len(train_graphs),
            "train_skipped": len(train_skipped_ids),
            "train_skipped_ids": train_skipped_ids,
            "val_total": len(val_indices),
            "val_graphs": len(val_graphs),
            "val_skipped": len(val_skipped_ids),
            "val_skipped_ids": val_skipped_ids,
        },
        "history": history,
    }
    (report_dir / "training_history.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_latest_pointer(base_report_dir, run_id)
    print(f"Best model (val_acc) -> {ckpt_dir}/best_model.pt")
    print(f"Training report      -> {report_dir}/ (run_id={run_id})")


if __name__ == "__main__":
    main()
