#!/usr/bin/env python3
"""Evaluate EpistemicHGNN V1 — stance, IS regression, and symbolic verdict."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.epistemic.registry import load_source_trust_registry
from src.model.data.builder import ClaimGraphBuilder
from src.model.data.featurizer import Featurizer
from src.model.evaluation.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_ece,
    compute_macro_f1,
    compute_pearson_r,
    compute_per_class_metrics,
    compute_per_group_accuracy,
    compute_rmse,
    compute_weighted_f1,
)
from src.model.models import MODELS
from src.model.models.nlihybridhgnn import NLIHybridHGNN
from src.model.config import GraphConfig
from src.model.data.types import NUM_STANCE, NUM_VERDICT, VERDICT_TO_INT
from src.pipeline.model.hparam_search import load_best_hparams

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}
_INT_TO_STANCE = {0: "supports", 1: "refutes", 2: "neutral"}


def _write_eval_plots(
    stance_metrics: dict,
    verdict_metrics: dict,
    plots_dir: Path,
) -> None:
    """Write PNG eval plots to plots_dir: confusion matrix, per-class F1, per-source accuracy."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Confusion matrix heatmap ──────────────────────────────────────────────
    confusion = verdict_metrics["confusion"]  # list[list[int]]
    labels = [_INT_TO_VERDICT[i] for i in range(len(confusion))]
    matrix = np.array(confusion, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Verdict Confusion Matrix")
    plt.colorbar(im, ax=ax)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
            )
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ── Per-class F1 bar chart ────────────────────────────────────────────────
    stance_pc = stance_metrics["per_class"]
    verdict_pc = verdict_metrics["per_class"]
    all_classes = list(stance_pc.keys()) + list(verdict_pc.keys())
    precisions = [m["precision"] for m in stance_pc.values()] + [
        m["precision"] for m in verdict_pc.values()
    ]
    f1s = [m["f1"] for m in stance_pc.values()] + [m["f1"] for m in verdict_pc.values()]

    x = np.arange(len(all_classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, precisions, width, label="Precision", color="steelblue")
    ax.bar(x + width / 2, f1s, width, label="F1", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision & F1 (Stance + Verdict)")
    ax.axvline(x=len(stance_pc) - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.text(
        len(stance_pc) / 2 - 0.5, 1.02, "Stance", ha="center", fontsize=8, color="gray"
    )
    ax.text(
        len(stance_pc) + len(verdict_pc) / 2 - 0.5,
        1.02,
        "Verdict",
        ha="center",
        fontsize=8,
        color="gray",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "class_f1.png", dpi=120)
    plt.close(fig)

    # ── Per-source accuracy bar chart ─────────────────────────────────────────
    per_source = verdict_metrics["per_source"]
    sources = list(per_source.keys())
    accs = [per_source[s]["accuracy"] for s in sources]
    ns = [per_source[s]["support"] for s in sources]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(sources, accs, color=["#4C9BE8", "#E87B4C", "#5CBE6A"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Verdict Accuracy")
    ax.set_title("Per-Source Verdict Accuracy")
    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "per_source_accuracy.png", dpi=120)
    plt.close(fig)


def _write_latest_pointer(base_dir: Path, run_id: str) -> None:
    """Point 'latest' at run_id. Uses a symlink; falls back to latest.txt on Windows."""
    latest = base_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_id)
    except OSError:
        (base_dir / "latest.txt").write_text(run_id, encoding="utf-8")


def _load_test_records(jsonl_path: Path, splits_dir: Path) -> list[dict]:
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    split_file = splits_dir / "test_indices.json"
    if split_file.exists():
        indices = json.loads(split_file.read_text(encoding="utf-8"))["indices"]
        return [records[i] for i in indices if i < len(records)]
    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate EpistemicHGNN (V1) on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--jsonl",
        required=True,
        help="Filtered training JSONL (test records drawn from this)",
    )
    ap.add_argument("--model", default="v1-hgnn", help="Model key from MODELS registry")
    ap.add_argument("--model-name", default="v1-hgnn", help="Display name for reports")
    ap.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    ap.add_argument("--embed-cache", default="out/model/graphs/embed_cache.pkl")
    ap.add_argument("--splits-dir", default="out/data/splits")
    ap.add_argument(
        "--output",
        required=True,
        help="Base directory for eval output (run-id subdir created inside)",
    )
    ap.add_argument("--run-id", default=None, help="Run ID for versioned output (default: ISO timestamp)")
    ap.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dim (default: from best_hparams or 256)")
    ap.add_argument("--heads", type=int, default=None, help="Override attention heads (default: from best_hparams or 4)")
    ap.add_argument("--dropout", type=float, default=None, help="Override dropout (default: from best_hparams or 0.3)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(args.device)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    base_output_dir = Path(args.output)
    output_dir = base_output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    if args.model not in MODELS:
        print(
            f"Unknown model '{args.model}'. Available: {list(MODELS)}", file=sys.stderr
        )
        sys.exit(1)

    best = load_best_hparams(args.model)
    if best:
        print(f"Loaded best hparams from configs/hparams/{args.model}_best_hparams.json")

    ckpt_raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt_raw.get("model_state_dict", ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw
    ec_threshold = ckpt_raw.get("ec_threshold", 0.35) if isinstance(ckpt_raw, dict) else 0.35

    def _resolve(cli_val, hparam_val, default):
        if cli_val is not None:
            return cli_val, "cli"
        if hparam_val is not None:
            return hparam_val, "hparam"
        return default, "default"

    hidden_dim, hd_src = _resolve(args.hidden_dim, best.get("hidden_dim") if best else None, 256)
    heads,      h_src  = _resolve(args.heads,      best.get("heads")      if best else None, 4)
    dropout,    d_src  = _resolve(args.dropout,     best.get("dropout")    if best else None, 0.3)
    print(f"Model arch: hidden_dim={hidden_dim}[{hd_src}]  heads={heads}[{h_src}]  "
          f"dropout={dropout}[{d_src}]  ec_threshold={ec_threshold}[checkpoint]")

    is_nli = MODELS.get(args.model) is NLIHybridHGNN
    graph_cfg = GraphConfig.v2() if is_nli else GraphConfig.v1()
    model = MODELS[args.model](
        graph_cfg, hidden_dim, heads, dropout, ec_threshold
    )
    model.load_state_dict(state)
    model.to(device).eval()

    # ── Load test records ─────────────────────────────────────────────────────
    test_records = _load_test_records(Path(args.jsonl), Path(args.splits_dir))
    registry = load_source_trust_registry(args.registry)
    featurizer = Featurizer(cache_path=args.embed_cache, device=args.device)
    builder = ClaimGraphBuilder(registry, featurizer, use_nli=is_nli)

    # ── Accumulators ─────────────────────────────────────────────────────────
    stance_pred_list:  list[torch.Tensor] = []
    stance_label_list: list[torch.Tensor] = []
    stance_logit_list: list[torch.Tensor] = []
    is_pred_list:      list[torch.Tensor] = []
    is_label_list:     list[torch.Tensor] = []
    verdict_preds:  list[int] = []
    verdict_trues:  list[int] = []
    source_labels:  list[str] = []
    decision_paths: list[str] = []
    claim_ids:      list[str] = []
    support_scores: list[float] = []
    refute_scores:  list[float] = []

    skipped_ids: list[str] = []
    with torch.no_grad():
        for record in test_records:
            try:
                g = builder.build(record)
            except Exception:
                skipped_ids.append(record.get("id", "unknown"))
                continue

            if g is None:
                skipped_ids.append(record.get("id", "unknown"))
                continue

            data = g.data.to(device)
            out = model.predict(data)

            stance_pred_list.append(out["stance_pred"].cpu())
            stance_label_list.append(data["evidence"].stance_y.cpu())
            stance_logit_list.append(out["stance_logits"].cpu())
            is_pred_list.append(out["is_pred"].view(-1).cpu())
            is_label_list.append(data["evidence"].is_y.view(-1).cpu())

            v_pred = VERDICT_TO_INT.get(out["verdict"], 2)
            v_true = g.label
            verdict_preds.append(v_pred)
            verdict_trues.append(v_true)
            source_labels.append(g.dataset)
            decision_paths.append(out.get("decision_path", "vh_fallback"))
            claim_ids.append(record.get("id", "unknown"))
            support_scores.append(float(out.get("support_score", 0.0)))
            refute_scores.append(float(out.get("refute_score", 0.0)))

    if not verdict_preds:
        print("No test records evaluated — check splits-dir and jsonl path.")
        return

    # ── Concatenate evidence-level tensors ────────────────────────────────────
    stance_preds   = torch.cat(stance_pred_list)
    stance_labels  = torch.cat(stance_label_list)
    stance_logits  = torch.cat(stance_logit_list)
    is_preds       = torch.cat(is_pred_list)
    is_labels      = torch.cat(is_label_list)
    verdict_pred_t = torch.tensor(verdict_preds, dtype=torch.long)
    verdict_true_t = torch.tensor(verdict_trues, dtype=torch.long)

    # ── Stance metrics ────────────────────────────────────────────────────────
    stance_per_class_raw = compute_per_class_metrics(stance_preds, stance_labels, NUM_STANCE)
    stance_metrics = {
        "accuracy": round(compute_accuracy(stance_preds, stance_labels), 4),
        "macro_f1": round(compute_macro_f1(stance_preds, stance_labels, NUM_STANCE), 4),
        "ece": compute_ece(stance_logits, stance_labels),
        "n_evidence": int(stance_preds.numel()),
        "per_class": {_INT_TO_STANCE[k]: v for k, v in stance_per_class_raw.items()},
    }

    # ── IS metrics ────────────────────────────────────────────────────────────
    is_metrics = {
        "rmse": compute_rmse(is_preds, is_labels),
        "pearson_r": compute_pearson_r(is_preds, is_labels),
        "n_evidence": int(is_preds.numel()),
        "pred_mean": round(is_preds.mean().item(), 4),
        "true_mean": round(is_labels.mean().item(), 4),
    }

    # ── Verdict metrics ───────────────────────────────────────────────────────
    verdict_per_class_raw = compute_per_class_metrics(verdict_pred_t, verdict_true_t, NUM_VERDICT)
    verdict_metrics = {
        "accuracy": round(compute_accuracy(verdict_pred_t, verdict_true_t), 4),
        "macro_f1": round(compute_macro_f1(verdict_pred_t, verdict_true_t, NUM_VERDICT), 4),
        "weighted_f1": round(compute_weighted_f1(verdict_pred_t, verdict_true_t, NUM_VERDICT), 4),
        "n_claims": len(verdict_preds),
        "skipped": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "per_class": {_INT_TO_VERDICT[k]: v for k, v in verdict_per_class_raw.items()},
        "confusion": compute_confusion_matrix(verdict_pred_t, verdict_true_t, NUM_VERDICT),
        "per_source": compute_per_group_accuracy(verdict_pred_t, verdict_true_t, source_labels),
    }

    # ── Decision path analysis ────────────────────────────────────────────────
    _PATH_KEYS = ["symbolic_supported", "symbolic_refuted", "vh_conflict", "vh_fallback"]

    def _path_stats(
        preds: list[int],
        trues: list[int],
        paths: list[str],
        sources: list[str],
        ids: list[str],
        sup_scores: list[float],
        ref_scores: list[float],
    ) -> dict:
        overall: dict[str, dict] = {k: {"count": 0, "correct": 0} for k in _PATH_KEYS}
        per_src: dict[str, dict[str, dict]] = {}
        conflict_failures: list[dict] = []
        conflict_correct:  list[dict] = []

        for pred, true, path, src, cid, sup, ref in zip(
            preds, trues, paths, sources, ids, sup_scores, ref_scores
        ):
            if path not in overall:
                overall[path] = {"count": 0, "correct": 0}
            overall[path]["count"] += 1
            if pred == true:
                overall[path]["correct"] += 1
            if src not in per_src:
                per_src[src] = {k: {"count": 0, "correct": 0} for k in _PATH_KEYS}
            if path not in per_src[src]:
                per_src[src][path] = {"count": 0, "correct": 0}
            per_src[src][path]["count"] += 1
            if pred == true:
                per_src[src][path]["correct"] += 1

            if path == "vh_conflict":
                entry = {
                    "id": cid,
                    "source": src,
                    "true": _INT_TO_VERDICT.get(true, str(true)),
                    "pred": _INT_TO_VERDICT.get(pred, str(pred)),
                    "support_score": round(sup, 4),
                    "refute_score": round(ref, 4),
                }
                if pred != true:
                    conflict_failures.append(entry)
                else:
                    conflict_correct.append(entry)

        for bucket in overall.values():
            bucket["accuracy"] = round(bucket["correct"] / bucket["count"], 4) if bucket["count"] else None
        for src_buckets in per_src.values():
            for bucket in src_buckets.values():
                bucket["accuracy"] = round(bucket["correct"] / bucket["count"], 4) if bucket["count"] else None
        n_symbolic = overall["symbolic_supported"]["count"] + overall["symbolic_refuted"]["count"]
        n_vh = overall["vh_conflict"]["count"] + overall["vh_fallback"]["count"]
        n_total = n_symbolic + n_vh
        return {
            "overall": overall,
            "per_source": per_src,
            "vh_conflict_failures": conflict_failures,
            "vh_conflict_correct":  conflict_correct,
            "summary": {
                "symbolic_votes": n_symbolic,
                "verdict_head_votes": n_vh,
                "total": n_total,
                "symbolic_pct": round(100 * n_symbolic / n_total, 1) if n_total else 0.0,
            },
        }

    decision_path_stats = _path_stats(
        verdict_preds, verdict_trues, decision_paths, source_labels,
        claim_ids, support_scores, refute_scores,
    )
    verdict_metrics["decision_paths"] = decision_path_stats

    # ── Write output files ────────────────────────────────────────────────────
    (output_dir / "stance_metrics.json").write_text(
        json.dumps(stance_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "is_metrics.json").write_text(json.dumps(is_metrics, indent=2), encoding="utf-8")
    (output_dir / "verdict_metrics.json").write_text(
        json.dumps(verdict_metrics, indent=2), encoding="utf-8"
    )

    # ── Write eval_summary.md ─────────────────────────────────────────────────
    _now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _pc_rows(per_class: dict) -> str:
        return "\n".join(
            f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
            for cls, m in per_class.items()
        )

    def _confusion_table(confusion: list) -> str:
        labels = [_INT_TO_VERDICT[i] for i in range(len(confusion))]
        header = "| True \\ Pred | " + " | ".join(labels) + " |"
        sep = "|" + "---|" * (len(labels) + 1)
        rows = [
            f"| {labels[i]} | "
            + " | ".join(str(confusion[i][j]) for j in range(len(labels)))
            + " |"
            for i in range(len(labels))
        ]
        return "\n".join([header, sep] + rows)

    def _source_rows(per_source: dict) -> str:
        return "\n".join(
            f"| {src} | {m['accuracy']:.3f} | {m['support']} |"
            for src, m in per_source.items()
        )

    def _decision_path_rows(overall: dict) -> str:
        rows = []
        for path in _PATH_KEYS:
            m = overall.get(path, {})
            cnt = m.get("count", 0)
            correct = m.get("correct", 0)
            acc = f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—"
            rows.append(f"| {path} | {cnt} | {correct} | {acc} |")
        return "\n".join(rows)

    def _decision_path_source_rows(per_src: dict, path: str) -> str:
        rows = []
        for src, buckets in per_src.items():
            m = buckets.get(path, {"count": 0, "correct": 0, "accuracy": None})
            cnt = m.get("count", 0)
            acc = f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—"
            rows.append(f"| {src} | {cnt} | {acc} |")
        return "\n".join(rows)

    plots_dir = output_dir / "plots"
    _write_eval_plots(stance_metrics, verdict_metrics, plots_dir)

    hparams_src = f"configs/hparams/{args.model}_best_hparams.json" if best else "defaults"
    eval_md = (
        f"# Evaluation Summary\n\n"
        f"**Model:** {args.model_name}  \n"
        f"**Generated:** {_now}  \n"
        f"**Hparams:** hidden_dim={hidden_dim}, heads={heads}, dropout={dropout} (source: {hparams_src})\n\n"
        f"---\n\n"
        f"## Stance Classification\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Accuracy | {stance_metrics['accuracy']:.4f} |\n"
        f"| Macro F1 | {stance_metrics['macro_f1']:.4f} |\n"
        f"| ECE | {stance_metrics['ece']:.4f} |\n"
        f"| N Evidence | {stance_metrics['n_evidence']} |\n\n"
        f"### Per-Class Breakdown\n\n"
        f"| Class | Precision | Recall | F1 | N |\n|-------|-----------|--------|----|---|\n"
        f"{_pc_rows(stance_metrics['per_class'])}\n\n"
        f"## Information Score (IS)\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| RMSE | {is_metrics['rmse']:.4f} |\n"
        f"| Pearson r | {is_metrics['pearson_r']:.4f} |\n"
        f"| Pred Mean | {is_metrics['pred_mean']:.4f} |\n"
        f"| True Mean | {is_metrics['true_mean']:.4f} |\n\n"
        f"## Verdict\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Accuracy | {verdict_metrics['accuracy']:.4f} |\n"
        f"| Macro F1 | {verdict_metrics['macro_f1']:.4f} |\n"
        f"| Weighted F1 | {verdict_metrics['weighted_f1']:.4f} |\n"
        f"| N Claims | {verdict_metrics['n_claims']} |\n\n"
        f"## Data Coverage\n\n"
        f"| Split | Total | Evaluated | Skipped |\n"
        f"|-------|-------|-----------|---------|\n"
        f"| val | {verdict_metrics['n_claims'] + verdict_metrics['skipped']} "
        f"| {verdict_metrics['n_claims']} | {verdict_metrics['skipped']} |\n\n"
        f"_Full list of skipped claim IDs available in `verdict_metrics.json`._\n\n"
        f"### Per-Class Breakdown\n\n"
        f"| Class | Precision | Recall | F1 | N |\n|-------|-----------|--------|----|---|\n"
        f"{_pc_rows(verdict_metrics['per_class'])}\n\n"
        f"### Confusion Matrix\n\n"
        f"{_confusion_table(verdict_metrics['confusion'])}\n\n"
        f"### Per-Source Verdict Accuracy\n\n"
        f"| Source | Accuracy | N |\n|--------|----------|---|\n"
        f"{_source_rows(verdict_metrics['per_source'])}\n\n"
        f"---\n\n"
        f"## Decision Path Analysis\n\n"
        f"_Symbolic_: EC score crossed threshold θ={ec_threshold} → override.  "
        f"_VerdictHead_: EC weak or conflicting → learned head decides.\n\n"
        f"**Summary:** {decision_path_stats['summary']['symbolic_votes']} symbolic overrides "
        f"({decision_path_stats['summary']['symbolic_pct']}%)  ·  "
        f"{decision_path_stats['summary']['verdict_head_votes']} VerdictHead decisions  "
        f"(out of {decision_path_stats['summary']['total']} total)\n\n"
        f"| Decision Path | Count | Correct | Accuracy |\n"
        f"|---------------|-------|---------|----------|\n"
        f"{_decision_path_rows(decision_path_stats['overall'])}\n\n"
        f"#### Symbolic Supported — per source\n\n"
        f"| Source | Count | Accuracy |\n|--------|-------|----------|\n"
        f"{_decision_path_source_rows(decision_path_stats['per_source'], 'symbolic_supported')}\n\n"
        f"#### Symbolic Refuted — per source\n\n"
        f"| Source | Count | Accuracy |\n|--------|-------|----------|\n"
        f"{_decision_path_source_rows(decision_path_stats['per_source'], 'symbolic_refuted')}\n\n"
        f"#### VerdictHead Fallback — per source\n\n"
        f"| Source | Count | Accuracy |\n|--------|-------|----------|\n"
        f"{_decision_path_source_rows(decision_path_stats['per_source'], 'vh_fallback')}\n\n"
        f"#### vh_conflict Failures ({len(decision_path_stats['vh_conflict_failures'])} wrong / "
        f"{decision_path_stats['overall'].get('vh_conflict', {}).get('count', 0)} total)\n\n"
        f"Both EC scores crossed θ but VerdictHead predicted incorrectly.\n\n"
        f"| ID | Source | True | Pred | sup_score | ref_score |\n"
        f"|----|--------|------|------|-----------|-----------|\n"
        + "".join(
            f"| {r['id']} | {r['source']} | {r['true']} | {r['pred']} "
            f"| {r['support_score']:.3f} | {r['refute_score']:.3f} |\n"
            for r in decision_path_stats["vh_conflict_failures"]
        )
        + f"\n---\n\n"
        f"## Plots\n\n"
        f"![Confusion Matrix](plots/confusion_matrix.png)\n\n"
        f"![Per-Class F1](plots/class_f1.png)\n\n"
        f"![Per-Source Accuracy](plots/per_source_accuracy.png)\n"
    )
    (output_dir / "eval_summary.md").write_text(eval_md, encoding="utf-8")

    _write_latest_pointer(base_output_dir, run_id)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(
        f"{args.model_name} — {len(verdict_preds)} claims  ({int(stance_preds.numel())} evidence items)"
    )
    print()
    print("H1 Stance")
    print(
        f"  accuracy  {stance_metrics['accuracy']:.4f}   macro_f1  {stance_metrics['macro_f1']:.4f}   ECE  {stance_metrics['ece']:.4f}"
    )
    for stance, m in stance_metrics["per_class"].items():
        print(
            f"  {stance:<8}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
    print("H2 Inference Strength")
    print(
        f"  RMSE  {is_metrics['rmse']:.4f}   Pearson r  {is_metrics['pearson_r']:.4f}"
    )
    print(
        f"  pred_mean={is_metrics['pred_mean']:.3f}  true_mean={is_metrics['true_mean']:.3f}"
    )
    print()
    print("Symbolic Verdict")
    print(
        f"  accuracy  {verdict_metrics['accuracy']:.4f}   macro_f1  {verdict_metrics['macro_f1']:.4f}   weighted_f1  {verdict_metrics['weighted_f1']:.4f}"
    )
    for verdict, m in verdict_metrics["per_class"].items():
        print(
            f"  {verdict:<22}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
    print("Per-source verdict accuracy:")
    for src, m in verdict_metrics["per_source"].items():
        print(f"  {src:<16}  acc={m['accuracy']:.3f}  n={m['support']}")
    print()
    print("Decision Path Analysis")
    dp_s = decision_path_stats["summary"]
    print(f"  symbolic overrides: {dp_s['symbolic_votes']} ({dp_s['symbolic_pct']}%)  "
          f"verdict_head: {dp_s['verdict_head_votes']}")
    for path in _PATH_KEYS:
        m = decision_path_stats["overall"].get(path, {})
        cnt = m.get("count", 0)
        if cnt == 0:
            continue
        acc_str = f"acc={m['accuracy']:.3f}" if m.get("accuracy") is not None else "acc=—"
        print(f"  {path:<24}  n={cnt:<4}  correct={m['correct']:<4}  {acc_str}")
    print(f"\nResults -> {output_dir}/ (run_id={run_id})")
    print("=" * 60)


if __name__ == "__main__":
    main()
