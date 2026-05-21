"""Single-model evaluation sub-tab."""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig

# Verdict class ordering for F1 display
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
    macro_f1 = sum(f1s.values()) / len(f1s)
    return {"acc": acc, "macro_f1": macro_f1, "f1s": f1s,
            "n_valid": len(valid), "n_total": len(rows)}


def _render_metrics(metrics: dict, model_key: str, cfg: "AppConfig") -> None:
    with st.container(border=True):
        st.markdown(f"**{model_key}** — {cfg.model_descriptions.get(model_key, '')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",  f"{metrics['acc']:.1%}")
        c2.metric("Macro F1",  f"{metrics['macro_f1']:.3f}")
        c3.metric("Evaluated", f"{metrics['n_valid']}/{metrics['n_total']}")
        f1s = metrics["f1s"]
        fc1, fc2, fc3 = st.columns(3)
        vd = cfg.verdict_display
        fc1.metric(f"{vd['supported']['emoji']} F1",            f"{f1s.get('supported', 0):.3f}")
        fc2.metric(f"{vd['refuted']['emoji']} F1",              f"{f1s.get('refuted', 0):.3f}")
        fc3.metric(f"{vd['not_enough_evidence']['emoji']} F1",  f"{f1s.get('not_enough_evidence', 0):.3f}")


def _render_predictions(rows: list[dict], cfg: "AppConfig", key_prefix: str) -> None:
    st.markdown("**Predictions**")
    vd = cfg.verdict_display
    for idx, row in enumerate(rows[:50]):
        if row["pred"] is None:
            continue
        ok      = row["pred"] == row["true"]
        t_emoji = vd.get(row["true"],  {}).get("emoji", "?")
        p_emoji = vd.get(row["pred"],  {}).get("emoji", "?")
        snip    = row["claim"][:80] + ("…" if len(row["claim"]) > 80 else "")
        result  = row.get("result") or {}
        with st.expander(
            f"{'✅' if ok else '❌'}  {row['source']}  ·  {snip}  "
            f"[{t_emoji}→{p_emoji}]",
            expanded=False,
        ):
            if result:
                sup = result.get("support_score", 0.0)
                ref = result.get("refute_score",  0.0)
                vp  = result.get("verdict_probs", [0, 0, 0])
                c1, c2, c3 = st.columns(3)
                c1.metric("Support", f"{sup:.3f}")
                c2.metric("Refute",  f"{ref:.3f}")
                c3.metric("EC",      "✓" if result.get("has_ec") else "✗")

                st.caption(f"Verdict probs — sup: {vp[0]:.3f}  ref: {vp[1]:.3f}  nei: {vp[2]:.3f}")

                bd = result.get("evidence_breakdown") or []
                if bd:
                    import pandas as pd
                    ev_rows = [
                        {
                            "Stance":   ev.get("stance", "—"),
                            "IS":       f"{ev.get('is_score', 0):.3f}",
                            "ST":       f"{ev.get('source_trust', 0):.3f}",
                            "EC":       f"{ev.get('ec_score', 0):.3f}",
                            "Pramana":  ev.get("pramana", "—"),
                            "Text":     (ev.get("text") or "")[:100],
                        }
                        for ev in bd
                    ]
                    st.dataframe(pd.DataFrame(ev_rows), use_container_width=True,
                                 hide_index=True)
            else:
                st.error(row.get("error", "no result"))


def render(cfg: "AppConfig") -> None:
    from app_update.core.loaders import load_test_records, get_predictor

    records = load_test_records(cfg.training_jsonl, cfg.splits_dir)
    if not records:
        st.warning("Test data not found.")
        return

    st.caption(f"{len(records)} test records available")

    # Model selector
    model_key = st.selectbox(
        "Model", cfg.model_keys, key="eval_single_model",
        format_func=lambda k: f"{k} — {cfg.model_descriptions.get(k, '')}",
    )

    c_n, c_seed = st.columns([3, 2])
    with c_n:
        n = st.slider("Samples", 5, min(len(records), 500), 20, 5, key="eval_n")
    with c_seed:
        fixed = st.checkbox("Fixed seed", value=False, key="eval_fixed")
        seed  = int(st.number_input("Seed", value=42, step=1,
                                    label_visibility="collapsed",
                                    key="eval_seed")) if fixed else None

    if st.button("▶ Run", type="primary", key="eval_run"):
        rng    = random.Random(seed)
        sample = rng.sample(records, min(n, len(records)))
        pred   = get_predictor(model_key, cfg.graph_cache_dir)

        if isinstance(pred, str):
            st.error(f"Could not load model: {pred}")
            return

        prog  = st.progress(0.0, text="Running…")
        rows: list[dict] = []
        for i, rec in enumerate(sample):
            prog.progress((i + 1) / len(sample), text=f"{i + 1}/{len(sample)}")
            true   = rec["verdict"]["label"]
            source = rec.get("provenance", {}).get("dataset", "?")
            try:
                out = pred.predict_from_record(rec)
                rows.append({"model": model_key, "claim": rec["claim"],
                             "true": true, "pred": out["verdict"],
                             "result": out, "source": source})
            except Exception as exc:
                rows.append({"model": model_key, "claim": rec["claim"],
                             "true": true, "pred": None,
                             "result": None, "source": source, "error": str(exc)})
        prog.empty()
        st.session_state["eval_rows"] = rows

    rows = st.session_state.get("eval_rows")
    if not rows:
        return

    # Filter to selected model only (session may have compare rows too)
    rows = [r for r in rows if r.get("model") == model_key]
    if not rows:
        return

    metrics = _compute_metrics(rows)
    if metrics:
        _render_metrics(metrics, model_key, cfg)

    st.divider()
    _render_predictions(rows, cfg, key_prefix="eval")
