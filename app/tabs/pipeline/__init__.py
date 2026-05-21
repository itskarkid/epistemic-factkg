"""Pipeline tab — GitLab-runner-style stage grid."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(root: Path, rel: str | None) -> bool:
    if rel is None:
        return False
    p = root / rel
    return (any(p.iterdir()) if p.is_dir() else p.exists())


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    r = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    return r.returncode, r.stdout


def _console(label: str, rc: int, out: str) -> None:
    if rc == 0:
        st.success(f"`{label}` passed")
    else:
        st.error(f"`{label}` failed (exit {rc})")
    if out.strip():
        with st.expander("Console", expanded=(rc != 0)):
            st.code(out, language="bash")


# ── Job grid ──────────────────────────────────────────────────────────────────

def _job_grid(
    root: Path,
    jobs: list[tuple[str, str, str | None]],
    prefix: str,
) -> str | None:
    """Render compact horizontal job chips with ▶ run buttons. Returns clicked key."""
    n = len(jobs)
    col_widths = []
    for i in range(n):
        col_widths.append(1.6)
        if i < n - 1:
            col_widths.append(0.2)
    cols = st.columns(col_widths)

    clicked: str | None = None
    col_i = 0
    for i, (key, label, out) in enumerate(jobs):
        ok            = _ok(root, out)
        badge         = "✅" if ok else "⬜"
        border_color  = "#16a34a" if ok else "#cbd5e1"
        bg            = "#f0fdf4" if ok else "#f8fafc"
        text_color    = "#15803d" if ok else "#475569"

        with cols[col_i]:
            st.markdown(
                f'<div style="border:1px solid {border_color};border-radius:6px;'
                f'background:{bg};padding:6px 8px;text-align:center;margin-bottom:4px">'
                f'<span style="font-size:0.9rem">{badge}</span><br>'
                f'<code style="font-size:0.70rem;color:{text_color};font-weight:600">{label}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("▶", key=f"{prefix}_{key}", width='stretch',
                         help=f"Run: {label}"):
                clicked = key

        col_i += 1
        if i < n - 1:
            with cols[col_i]:
                st.markdown(
                    '<div style="text-align:center;padding-top:14px;'
                    'color:#94a3b8;font-size:0.85rem">→</div>',
                    unsafe_allow_html=True,
                )
            col_i += 1

    return clicked


# ── Data pipeline ─────────────────────────────────────────────────────────────

_DATA_JOBS: list[tuple[str, str, str | None]] = [
    ("enrich",   "enrich-registry", "data/registry/source_trust_registry.jsonl"),
    ("build",    "build",           "out/data/training/epistemic_factkg_training.jsonl"),
    ("validate", "validate",        "out/reports/data/validation.json"),
    ("report",   "report",          "out/reports/data"),
]


def _render_data_pipeline(cfg: "AppConfig") -> None:
    root = cfg.root

    c_title, c_btn = st.columns([5, 1])
    c_title.markdown("**Data Pipeline**")
    run_all = c_btn.button("▶ All", key="dp_run_all", width='stretch')

    clicked = _job_grid(root, _DATA_JOBS, "dp")

    # Options shown always (read by handlers below)
    rebuild = st.checkbox("Re-simulate AI2THOR before build", value=False, key="dp_rebuild")

    st.divider()

    if run_all:
        for label, cmd in [
            ("enrich-registry", ["just", "enrich-registry"]),
            ("build",           ["just", "build", "rebuild=true"] if rebuild else ["just", "build"]),
            ("validate",        ["just", "validate"]),
            ("report",          ["just", "report"]),
        ]:
            with st.spinner(f"Running {label}…"):
                rc, out = _run(cmd, root)
            _console(label, rc, out)
            if rc != 0:
                break
        st.rerun()

    elif clicked == "enrich":
        with st.spinner("Running enrich-registry…"):
            rc, out = _run(["just", "enrich-registry"], root)
        _console("enrich-registry", rc, out)
        st.rerun()

    elif clicked == "build":
        cmd = ["just", "build", "rebuild=true"] if rebuild else ["just", "build"]
        with st.spinner("Running build…"):
            rc, out = _run(cmd, root)
        _console("build", rc, out)
        st.rerun()

    elif clicked == "validate":
        with st.spinner("Running validate…"):
            rc, out = _run(["just", "validate"], root)
        _console("validate", rc, out)
        st.rerun()

    elif clicked == "report":
        with st.spinner("Running report…"):
            rc, out = _run(["just", "report"], root)
        _console("report", rc, out)
        st.rerun()


# ── Model pipeline ────────────────────────────────────────────────────────────

def _render_model_pipeline(cfg: "AppConfig") -> None:
    from itertools import combinations

    root       = cfg.root
    model_keys = list(cfg.model_keys)

    c_sel, c_trials = st.columns([4, 1])
    selected = c_sel.multiselect(
        "Models", model_keys, default=[model_keys[0]], key="mp_models",
        help="Select 1–4 models. Compare step runs all pairs automatically.",
    )
    if not selected:
        st.info("Select at least one model above.")
        return
    n_trials = int(c_trials.number_input("HP trials", min_value=5, max_value=500,
                                          value=30, step=5, key="mp_trials"))

    # Show only one train/eval chip per selected model; shared graph jobs are single
    shared_jobs: list[tuple[str, str, str | None]] = [
        ("graph",     "graph",     "out/model/graphs/graph_dataset.pt"),
        ("graph_nli", "graph-nli", "out/model/graphs/graph_dataset_nli.pt"),
        ("hpsearch",  "hp-search", "configs/hparams/best_hparams.json"),
    ]
    per_model_jobs: list[tuple[str, str, str | None]] = []
    for mk in selected:
        per_model_jobs.append((f"train_{mk}", f"train·{mk}",
                                f"out/model/{mk}/checkpoints/best_model.pt"))
        per_model_jobs.append((f"eval_{mk}",  f"eval·{mk}",
                                f"out/reports/model/{mk}/eval/verdict_metrics.json"))

    compare_pairs = list(combinations(selected, 2))
    cmp_jobs: list[tuple[str, str, str | None]] = [
        (f"cmp_{a}_{b}", f"cmp·{a}·{b}",
         f"out/reports/model/comparison_{a}_vs_{b}.md")
        for a, b in compare_pairs
    ]

    all_jobs = shared_jobs + per_model_jobs + cmp_jobs

    sel_str = "+".join(selected)
    c_title, c_btn = st.columns([5, 1])
    c_title.markdown(f"**Model Pipeline** — `{sel_str}`")
    run_core = c_btn.button("▶ Core", key="mp_run_core", width='stretch',
                             help="graph → train all → eval all")

    clicked = _job_grid(root, all_jobs, "mp")

    st.divider()

    if run_core:
        steps = [("graph", ["just", "graph"])]
        for mk in selected:
            steps.append((f"train {mk}", ["just", "train", mk]))
            steps.append((f"eval {mk}",  ["just", "eval",  mk]))
        for label, cmd in steps:
            with st.spinner(f"Running {label}…"):
                rc, out = _run(cmd, root)
            _console(label, rc, out)
            if rc != 0:
                break
        st.rerun()

    elif clicked == "graph":
        with st.spinner("Building graphs…"):
            rc, out = _run(["just", "graph"], root)
        _console("graph", rc, out)
        st.rerun()

    elif clicked == "graph_nli":
        with st.spinner("Building NLI graphs…"):
            rc, out = _run(["just", "graph-nli"], root)
        _console("graph-nli", rc, out)
        st.rerun()

    elif clicked == "hpsearch":
        for mk in selected:
            with st.spinner(f"HP search — {mk} ({n_trials} trials)…"):
                rc, out = _run(["just", "hparam-search", mk, str(n_trials)], root)
            _console(f"hp-search {mk}", rc, out)
            if rc != 0:
                break
        st.rerun()

    elif clicked and clicked.startswith("train_"):
        mk = clicked[len("train_"):]
        with st.spinner(f"Training {mk}…"):
            rc, out = _run(["just", "train", mk], root)
        _console(f"train {mk}", rc, out)
        st.rerun()

    elif clicked and clicked.startswith("eval_"):
        mk = clicked[len("eval_"):]
        with st.spinner(f"Evaluating {mk}…"):
            rc, out = _run(["just", "eval", mk], root)
        _console(f"eval {mk}", rc, out)
        st.rerun()

    elif clicked and clicked.startswith("cmp_"):
        # cmp_{a}_{b} — we stored actual pairs
        for a, b in compare_pairs:
            if clicked == f"cmp_{a}_{b}":
                with st.spinner(f"Comparing {a} vs {b}…"):
                    rc, out = _run(["just", "compare", a, b], root)
                _console(f"compare {a} vs {b}", rc, out)
                st.rerun()
                break


# ── Main render ───────────────────────────────────────────────────────────────

def render(cfg: "AppConfig") -> None:
    t_data, t_model = st.tabs(["Data Pipeline", "Model Pipeline"])
    with t_data:
        _render_data_pipeline(cfg)
    with t_model:
        _render_model_pipeline(cfg)
