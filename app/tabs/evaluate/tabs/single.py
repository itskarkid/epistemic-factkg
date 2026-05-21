"""Single-model evaluation sub-tab."""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig

_VERDICT_ORDER = ["supported", "refuted", "not_enough_evidence"]

_PLACEHOLDER_TEXTS = frozenset([
    "no sensor evidence found for this object type.",
    "no evidence found.",
])
_QA_PREFIXES = ("q:", "did ", "does ", "is ", "are ", "was ", "were ", "have ", "has ", "can ", "could ", "would ")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _failure_pattern(row: dict) -> str:
    if row.get("pred") is None:
        return "error"
    if row["pred"] == row["true"]:
        return "correct"
    result = row.get("result") or {}
    has_ec = result.get("has_ec", False)
    sup    = result.get("support_score", 0.0)
    ref    = result.get("refute_score",  0.0)
    if has_ec:
        if ref > 0.35 and row["pred"] != "refuted":
            return "ec_override_fix"
        if sup > 0.35 and row["pred"] != "supported":
            return "ec_override_fix"
        if max(sup, ref) < 0.20:
            return "all_neutral_ec"
    bd = (result.get("evidence_breakdown") or [])
    if any((ev.get("text") or "").strip().lower() in _PLACEHOLDER_TEXTS for ev in bd):
        return "placeholder_ev"
    if any((ev.get("text") or "").lower().lstrip().startswith(_QA_PREFIXES) for ev in bd):
        return "qa_format_ev"
    return "genuine_error"


def _build_eval_export(rows: list[dict]) -> str:
    export = []
    for row in rows:
        result = row.get("result") or {}
        probs  = result.get("verdict_probs") or [0.0, 0.0, 0.0]
        sup    = result.get("support_score", 0.0)
        ref    = result.get("refute_score",  0.0)
        bd     = result.get("evidence_breakdown") or []
        pred   = row.get("pred")
        true   = row.get("true", "")
        has_ph = any((ev.get("text") or "").strip().lower() in _PLACEHOLDER_TEXTS for ev in bd)
        has_qa = any((ev.get("text") or "").lower().lstrip().startswith(_QA_PREFIXES) for ev in bd)
        ec_dis = (ref > 0.35 and pred != "refuted") or (sup > 0.35 and pred != "supported")
        export.append({
            "model":              row.get("model", ""),
            "source_dataset":     row.get("source", ""),
            "claim":              row.get("claim", ""),
            "true_label":         true,
            "predicted_label":    pred,
            "correct":            (pred == true) if pred is not None else None,
            "support_score":      round(sup, 4),
            "refute_score":       round(ref, 4),
            "verdict_probs":      {
                "supported": round(probs[0], 4),
                "refuted":   round(probs[1], 4),
                "nei":       round(probs[2], 4),
            },
            "has_placeholder_ev": has_ph,
            "has_qa_ev":          has_qa,
            "ec_disagrees":       ec_dis,
            "failure_pattern":    _failure_pattern(row),
            "evidence":           [
                {
                    "text":              ev.get("text", ""),
                    "stance":            ev.get("stance"),
                    "support_confidence": ev.get("support_confidence"),
                    "refute_confidence":  ev.get("refute_confidence"),
                    "is_score":          ev.get("is_score"),
                    "source_trust":      ev.get("source_trust"),
                    "evidence_weight":   ev.get("evidence_weight"),
                    "ec_score":          ev.get("ec_score"),
                    "source_type":       ev.get("source_type"),
                    "nli_probs":         ev.get("nli_probs"),
                }
                for ev in bd
            ],
        })
    return json.dumps(export, indent=2, ensure_ascii=False)


# ── Metrics ───────────────────────────────────────────────────────────────────

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
        fc1.metric(f"{vd['supported']['emoji']} F1",           f"{f1s.get('supported', 0):.3f}")
        fc2.metric(f"{vd['refuted']['emoji']} F1",             f"{f1s.get('refuted', 0):.3f}")
        fc3.metric(f"{vd['not_enough_evidence']['emoji']} F1", f"{f1s.get('not_enough_evidence', 0):.3f}")


# ── Predictions ───────────────────────────────────────────────────────────────

def _render_predictions(rows: list[dict], cfg: "AppConfig", predictor, model_key: str) -> None:
    from app.tabs.verify import _render_result_tabs

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
            f"{'✅' if ok else '❌'}  {row['source']}  ·  {snip}  [{t_emoji}→{p_emoji}]",
            expanded=False,
        ):
            if result and result.get("evidence_breakdown"):
                # ensure claim_text falls back to row["claim"] for eval rows
                if not result.get("claim_text"):
                    result = {**result, "claim_text": row.get("claim", "")}
                _render_result_tabs(result, predictor, model_key)
            else:
                st.error(row.get("error", "no result"))


# ── Main render ───────────────────────────────────────────────────────────────

def render(cfg: "AppConfig") -> None:
    from app.core.loaders import load_test_records, get_predictor

    records = load_test_records(cfg.training_jsonl, cfg.splits_dir)
    if not records:
        st.warning("Test data not found.")
        return

    st.caption(f"{len(records)} test records available")

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
        pred   = get_predictor(model_key, cfg.graph_cache_dir, cfg.registry_path)

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
        st.session_state["eval_rows"]    = rows
        st.session_state["eval_model"]   = model_key
        st.session_state["eval_pred_obj"] = pred

    rows = st.session_state.get("eval_rows")
    if not rows:
        return

    rows = [r for r in rows if r.get("model") == model_key]
    if not rows:
        return

    metrics = _compute_metrics(rows)
    if metrics:
        _render_metrics(metrics, model_key, cfg)

    # Download + failure pattern
    valid_rows = [r for r in rows if r.get("pred") is not None]
    if valid_rows:
        json_data = _build_eval_export(rows)
        c_dl, c_pat = st.columns([2, 5])
        with c_dl:
            st.download_button(
                "⬇ Download JSON", json_data,
                file_name="eval_results.json", mime="application/json",
                key="eval_dl_btn",
            )
        with c_pat:
            patterns = Counter(_failure_pattern(r) for r in rows)
            pat_str = "  ·  ".join(
                f"**{k}** ×{v}" for k, v in patterns.most_common()
            )
            st.caption(pat_str)

    st.divider()

    # Re-use cached predictor for sub-tabs
    pred_obj = st.session_state.get("eval_pred_obj")
    if pred_obj is None or isinstance(pred_obj, str):
        from app.core.loaders import get_predictor
        pred_obj = get_predictor(model_key, cfg.graph_cache_dir, cfg.registry_path)

    if isinstance(pred_obj, str):
        st.warning(f"Model not loaded — prediction details unavailable ({pred_obj})")
        return

    _render_predictions(rows, cfg, pred_obj, model_key)
