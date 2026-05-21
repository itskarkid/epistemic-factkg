"""Compare-all-models evaluation sub-tab."""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig

_VERDICT_ORDER = ["supported", "refuted", "not_enough_evidence"]


def _compute_metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["pred"] is not None]
    if not valid:
        return {}
    correct = sum(r["pred"] == r["true"] for r in valid)
    acc = correct / len(valid)
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for r in valid:
        t, p = r["true"], r["pred"]
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = {}
    for lbl in _VERDICT_ORDER:
        pr = tp[lbl] / (tp[lbl] + fp[lbl]) if tp[lbl] + fp[lbl] else 0.0
        re = tp[lbl] / (tp[lbl] + fn[lbl]) if tp[lbl] + fn[lbl] else 0.0
        f1s[lbl] = 2 * pr * re / (pr + re) if pr + re else 0.0
    return {"acc": acc, "macro_f1": sum(f1s.values()) / len(f1s),
            "f1s": f1s, "n_valid": len(valid), "n_total": len(rows)}


def render(cfg: "AppConfig") -> None:
    from app_update.core.loaders import load_test_records, get_predictor

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
        total = len(cfg.model_keys) * len(sample)
        step  = 0
        prog  = st.progress(0.0, text="Loading…")

        for mk in cfg.model_keys:
            pred = get_predictor(mk, cfg.graph_cache_dir)
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
        st.session_state["cmp_rows"] = all_rows

    rows = st.session_state.get("cmp_rows")
    if not rows:
        return

    models = list(cfg.model_keys)
    vd = cfg.verdict_display

    # ── Metrics table ─────────────────────────────────────────────────────────
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
                     use_container_width=True)

    st.divider()

    # ── Side-by-side prediction browser ──────────────────────────────────────
    st.markdown("**Side-by-side predictions**")
    inspect_idx = st.number_input(
        "Record index", min_value=0,
        max_value=max(0, len(sample) - 1) if (sample := records[:n]) else 0,
        value=0, key="cmp_idx",
    )

    unique_claims = list(dict.fromkeys(r["claim"] for r in rows if r["model"] == models[0]))
    if inspect_idx < len(unique_claims):
        claim = unique_claims[inspect_idx]
        st.markdown(f"**Claim:** {claim}")

        cols = st.columns(len(models))
        for col, mk in zip(cols, models):
            row = next((r for r in rows if r["model"] == mk and r["claim"] == claim), None)
            if row is None:
                col.caption(f"{mk}: no result")
                continue
            pred   = row.get("pred") or "—"
            true   = row.get("true") or "—"
            ok     = pred == true
            result = row.get("result") or {}
            with col:
                st.markdown(f"**{mk}**")
                st.markdown(
                    f"{'✅' if ok else '❌'} `{pred}`  "
                    f"_(true: `{true}`)_"
                )
                if result:
                    sup = result.get("support_score", 0.0)
                    ref = result.get("refute_score", 0.0)
                    st.caption(f"sup: {sup:.3f}  ref: {ref:.3f}")
