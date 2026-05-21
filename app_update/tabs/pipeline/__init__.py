"""Pipeline tab — run data and model pipeline stages from the UI."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status(root: Path, rel: str | None) -> tuple[str, bool]:
    """Return (badge, exists) for a relative output path."""
    if rel is None:
        return "🔄", False
    p = root / rel
    if p.is_dir():
        exists = any(p.iterdir())
    else:
        exists = p.exists()
    return ("✅" if exists else "⬜"), exists


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    return result.returncode, result.stdout


def _show_result(label: str, rc: int, out: str) -> None:
    if rc == 0:
        st.success(f"{label} completed.")
    else:
        st.error(f"{label} failed (exit {rc}).")
    if out.strip():
        with st.expander("Output", expanded=(rc != 0)):
            st.code(out, language=None)


def _stage(
    root: Path,
    key: str,
    label: str,
    desc: str,
    output: str | None,
    *,
    extra_widget=None,
) -> bool:
    """Render one pipeline stage card. Returns True when Run was clicked."""
    badge, _ = _status(root, output)
    with st.container(border=True):
        h, b = st.columns([5, 1])
        h.markdown(f"{badge} **{label}**  \n{desc}")
        if extra_widget:
            extra_widget()
        run = b.button("Run", key=f"pl_run_{key}", use_container_width=True)
        if output and (root / output).exists() and not (root / output).is_dir():
            st.caption(f"`{output}`")
    return run


# ── Data pipeline ─────────────────────────────────────────────────────────────

def _render_data_pipeline(cfg: "AppConfig") -> None:
    root = cfg.root

    st.markdown("Build and validate the training dataset. Run stages in order.")

    # Quick-run all
    if st.button("▶ Run full data pipeline", key="dp_run_all"):
        for lbl, cmd in [
            ("Build Training Data", ["just", "build"]),
            ("Validate",            ["just", "validate"]),
            ("Generate Report",     ["just", "report"]),
        ]:
            with st.spinner(f"Running {lbl}…"):
                rc, out = _run(cmd, root)
            _show_result(lbl, rc, out)
            if rc != 0:
                break
        st.rerun()

    st.divider()

    # ── Enrich registry ───────────────────────────────────────────────────────
    if _stage(root, "enrich-registry",
              "Enrich Registry",
              "Scan AVeriTeC URLs and enrich source trust values in registry.",
              "data/registry/source_trust_registry.jsonl"):
        with st.spinner("Running Enrich Registry…"):
            rc, out = _run(["just", "enrich-registry"], root)
        _show_result("Enrich Registry", rc, out)
        st.rerun()

    # ── Build ─────────────────────────────────────────────────────────────────
    rebuild = st.checkbox("Re-simulate AI2THOR before building", value=False, key="dp_rebuild")
    if _stage(root, "build",
              "Build Training Data",
              "Merge AI2THOR + AVeriTeC + synthetic; filter; create splits.",
              "out/data/training/epistemic_factkg_training.jsonl"):
        cmd = ["just", "build", "rebuild=true"] if rebuild else ["just", "build"]
        with st.spinner("Running Build Training Data…"):
            rc, out = _run(cmd, root)
        _show_result("Build Training Data", rc, out)
        st.rerun()

    # ── Validate ──────────────────────────────────────────────────────────────
    if _stage(root, "validate",
              "Validate",
              "Validate unified JSONL schema and Pramana distribution (ADR-012).",
              "out/reports/data/validation.json"):
        with st.spinner("Running Validate…"):
            rc, out = _run(["just", "validate"], root)
        _show_result("Validate", rc, out)
        st.rerun()

    # ── Report ────────────────────────────────────────────────────────────────
    if _stage(root, "report",
              "Generate Report",
              "Produce dataset quality report (markdown + charts).",
              "out/reports/data"):
        with st.spinner("Running Generate Report…"):
            rc, out = _run(["just", "report"], root)
        _show_result("Generate Report", rc, out)
        st.rerun()


# ── Model pipeline ────────────────────────────────────────────────────────────

def _render_model_pipeline(cfg: "AppConfig") -> None:
    root = cfg.root

    st.markdown("Build graphs, tune hyperparameters, train, evaluate, and compare models.")

    model_keys = list(cfg.model_keys)

    c1, c2, c3 = st.columns([2, 2, 1])
    model_key  = c1.selectbox("Model", model_keys, key="mp_model")
    model_key2 = c2.selectbox("Compare with", model_keys, key="mp_model2",
                               help="Used only for the Compare step.")
    n_trials   = c3.number_input("HP trials", min_value=5, max_value=500, value=30, step=5,
                                  key="mp_trials", help="Trials for hyperparameter search.")

    # Quick-run full pipeline for one model
    if st.button(f"▶ Run full model pipeline for {model_key}", key="mp_run_all"):
        for lbl, cmd in [
            ("Build Graphs",    ["just", "graph"]),
            (f"Train {model_key}", ["just", "train", model_key]),
            (f"Eval {model_key}",  ["just", "eval",  model_key]),
        ]:
            with st.spinner(f"Running {lbl}…"):
                rc, out = _run(cmd, root)
            _show_result(lbl, rc, out)
            if rc != 0:
                break
        st.rerun()

    st.divider()

    # ── Build graphs ──────────────────────────────────────────────────────────
    if _stage(root, "graph",
              "Build Graphs",
              "Build PyG HeteroData graph dataset from training JSONL.",
              "out/model/graphs/graph_dataset.pt"):
        with st.spinner("Running Build Graphs…"):
            rc, out = _run(["just", "graph"], root)
        _show_result("Build Graphs", rc, out)
        st.rerun()

    # ── Build NLI graphs ──────────────────────────────────────────────────────
    if _stage(root, "graph-nli",
              "Build NLI Graphs",
              "Build NLI-enhanced graph dataset (required for v3-nli model).",
              "out/model/graphs/graph_dataset_nli.pt"):
        with st.spinner("Running Build NLI Graphs…"):
            rc, out = _run(["just", "graph-nli"], root)
        _show_result("Build NLI Graphs", rc, out)
        st.rerun()

    # ── Hyperparameter search ─────────────────────────────────────────────────
    if _stage(root, "hparam-search",
              f"Hyperparameter Search — {model_key}",
              f"Optuna search ({n_trials} trials). Saves best params to configs/hparams/.",
              "configs/hparams/best_hparams.json"):
        with st.spinner(f"Running Hyperparameter Search for {model_key}…"):
            rc, out = _run(["just", "hparam-search", model_key, str(int(n_trials))], root)
        _show_result(f"Hyperparameter Search — {model_key}", rc, out)
        st.rerun()

    # ── Train ─────────────────────────────────────────────────────────────────
    if _stage(root, "train",
              f"Train — {model_key}",
              f"Train {model_key} to convergence. Saves checkpoint to out/model/{model_key}/.",
              f"out/model/{model_key}/checkpoints/best_model.pt"):
        with st.spinner(f"Training {model_key}…"):
            rc, out = _run(["just", "train", model_key], root)
        _show_result(f"Train — {model_key}", rc, out)
        st.rerun()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if _stage(root, "eval",
              f"Evaluate — {model_key}",
              f"Evaluate {model_key} on test set; generate metrics and confusion plots.",
              f"out/reports/model/{model_key}/eval/verdict_metrics.json"):
        with st.spinner(f"Evaluating {model_key}…"):
            rc, out = _run(["just", "eval", model_key], root)
        _show_result(f"Evaluate — {model_key}", rc, out)
        st.rerun()

    # ── Compare ───────────────────────────────────────────────────────────────
    cmp_out = f"out/reports/model/comparison_{model_key}_vs_{model_key2}.md"
    if _stage(root, "compare",
              f"Compare — {model_key}  vs  {model_key2}",
              "Generate side-by-side comparison report (markdown with Δ metrics).",
              cmp_out):
        with st.spinner(f"Comparing {model_key} vs {model_key2}…"):
            rc, out = _run(["just", "compare", model_key, model_key2], root)
        _show_result(f"Compare — {model_key} vs {model_key2}", rc, out)
        st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render(cfg: "AppConfig") -> None:
    t_data, t_model = st.tabs(["Data Pipeline", "Model Pipeline"])
    with t_data:
        _render_data_pipeline(cfg)
    with t_model:
        _render_model_pipeline(cfg)
