"""Compare-all-models evaluation sub-tab."""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig

# Reuse helpers from single.py — no duplication
from app.tabs.evaluate.tabs.single import (
    _build_eval_export,
    _compute_metrics,
    _failure_pattern,
    _render_metrics,
    _render_predictions,
)

_VERDICT_ORDER = ["supported", "refuted", "not_enough_evidence"]


def render(cfg: "AppConfig") -> None:
    from app.core.loaders import get_predictor, load_test_records

    records = load_test_records(cfg.training_jsonl, cfg.splits_dir)
    if not records:
        st.warning("Test data not found.")
        return

    st.caption(
        f"Run all {len(cfg.model_keys)} models on the same random sample "
        "and compare results side-by-side."
    )
    st.caption(f"{len(records)} test records available")

    c_n, c_seed = st.columns([3, 2])
    with c_n:
        n = st.slider("Samples", 5, min(len(records), 200), 20, 5, key="cmp_n")
    with c_seed:
        fixed = st.checkbox("Fixed seed", value=True, key="cmp_fixed")
        seed  = int(st.number_input("Seed", value=42, step=1,
                                    label_visibility="collapsed",
                                    key="cmp_seed")) if fixed else None

    if st.button("▶ Run All Models", type="primary", key="cmp_run"):
        rng    = random.Random(seed)
        sample = rng.sample(records, min(n, len(records)))

        all_rows: list[dict] = []
        predictors: dict = {}
        total = len(cfg.model_keys) * len(sample)
        step  = 0
        prog  = st.progress(0.0, text="Loading…")

        for mk in cfg.model_keys:
            pred = get_predictor(mk, cfg.graph_cache_dir, cfg.registry_path)
            predictors[mk] = pred
            for rec in sample:
                step += 1
                prog.progress(step / total, text=f"{mk} · {step}/{total}")
                true   = rec["verdict"]["label"]
                source = rec.get("provenance", {}).get("dataset", "?")
                if isinstance(pred, str):
                    all_rows.append({"model": mk, "claim": rec["claim"],
                                     "true": true, "pred": None,
                                     "result": None, "source": source,
                                     "error": pred})
                    continue
                try:
                    out = pred.predict_from_record(rec)
                    all_rows.append({"model": mk, "claim": rec["claim"],
                                     "true": true, "pred": out["verdict"],
                                     "result": out, "source": source})
                except Exception as exc:
                    all_rows.append({"model": mk, "claim": rec["claim"],
                                     "true": true, "pred": None,
                                     "result": None, "source": source,
                                     "error": str(exc)})

        prog.empty()
        st.session_state["cmp_rows"]       = all_rows
        st.session_state["cmp_predictors"] = predictors

    rows = st.session_state.get("cmp_rows")
    if not rows:
        return

    models   = list(cfg.model_keys)
    vd       = cfg.verdict_display

    # ── Summary metrics table ─────────────────────────────────────────────────
    import pandas as pd

    metric_rows = []
    for mk in models:
        m_rows = [r for r in rows if r["model"] == mk]
        m = _compute_metrics(m_rows)
        if m:
            metric_rows.append({
                "Model":    mk,
                "Accuracy": f"{m['acc']:.1%}",
                "Macro F1": f"{m['macro_f1']:.3f}",
                f"{vd['supported']['emoji']} F1":           f"{m['f1s'].get('supported', 0):.3f}",
                f"{vd['refuted']['emoji']} F1":             f"{m['f1s'].get('refuted', 0):.3f}",
                f"{vd['not_enough_evidence']['emoji']} F1": f"{m['f1s'].get('not_enough_evidence', 0):.3f}",
                "n":        f"{m['n_valid']}/{m['n_total']}",
            })

    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows).set_index("Model"),
                     width='stretch')

    st.divider()

    # ── Per-model detailed view (same UX as Single) ───────────────────────────
    st.markdown("**Inspect predictions per model**")
    inspect_model = st.selectbox(
        "Model", models, key="cmp_inspect_model",
        format_func=lambda k: f"{k} — {cfg.model_descriptions.get(k, '')}",
    )

    model_rows = [r for r in rows if r["model"] == inspect_model]
    if not model_rows:
        return

    metrics = _compute_metrics(model_rows)
    if metrics:
        _render_metrics(metrics, inspect_model, cfg)

    valid_rows = [r for r in model_rows if r.get("pred") is not None]
    if valid_rows:
        json_data = _build_eval_export(model_rows)
        c_dl, c_pat = st.columns([2, 5])
        with c_dl:
            st.download_button(
                "⬇ Download JSON", json_data,
                file_name=f"eval_{inspect_model}.json", mime="application/json",
                key="cmp_dl_btn",
            )
        with c_pat:
            patterns = Counter(_failure_pattern(r) for r in model_rows)
            pat_str = "  ·  ".join(
                f"**{k}** ×{v}" for k, v in patterns.most_common()
            )
            st.caption(pat_str)

    st.divider()

    # Re-use cached predictor (loaded during run; fall back to fresh load on reload)
    predictors = st.session_state.get("cmp_predictors", {})
    pred_obj   = predictors.get(inspect_model)
    if pred_obj is None or isinstance(pred_obj, str):
        pred_obj = get_predictor(inspect_model, cfg.graph_cache_dir, cfg.registry_path)

    if isinstance(pred_obj, str):
        st.warning(f"Model not loaded — prediction details unavailable ({pred_obj})")
        return

    _render_predictions(model_rows, cfg, pred_obj, inspect_model)
