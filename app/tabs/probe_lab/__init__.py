"""Probe Lab tab — run all epistemic probes, test hypotheses, compare models."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig

_PROBES_PATH = Path("data/probes/epistemic_probes.jsonl")

_HYPOTHESES = [
    {
        "id":    "H1",
        "title": "Source Trust changes verdict",
        "desc":  "Identical evidence text from Nature (ST=0.90) vs Twitter (ST=0.35) → different verdict",
        "probes": ["probe_001", "probe_002", "probe_003", "probe_004"],
        "pairs":  [("probe_001", "probe_002"), ("probe_003", "probe_004")],
        "pair_metric": "support_score",
    },
    {
        "id":    "H2",
        "title": "Epistemic priority: quality > quantity",
        "desc":  "One high-trust source beats multiple low-trust sources; combined weak evidence stays below θ",
        "probes": ["probe_007", "probe_008", "probe_009", "probe_016", "probe_018"],
        "pairs":  [],
        "pair_metric": None,
    },
    {
        "id":    "H3",
        "title": "Shortcut-breaking: stance alone ≠ verdict",
        "desc":  "Supporting stance + low-trust source → NEI, not SUPPORTED",
        "probes": ["probe_005", "probe_006"],
        "pairs":  [],
        "pair_metric": None,
    },
    {
        "id":    "H4",
        "title": "IS cap on low-trust sources",
        "desc":  "Confident-sounding text from social media (ST=0.35) → IS capped → EC stays low → NEI",
        "probes": ["probe_017"],
        "pairs":  [],
        "pair_metric": None,
    },
    {
        "id":    "H5",
        "title": "Evidence type (EW) affects epistemic confidence",
        "desc":  "Testimony (EW=0.80) gives higher EC than inference (EW=0.55) at the same source trust",
        "probes": ["probe_019", "probe_020", "probe_012", "probe_013"],
        "pairs":  [("probe_020", "probe_019"), ("probe_012", "probe_013")],
        "pair_metric": "support_score",
    },
]


@lru_cache(maxsize=1)
def _load_probes() -> list[dict]:
    if not _PROBES_PATH.exists():
        return []
    probes = []
    for line in _PROBES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            probes.append(json.loads(line))
    return probes


def _probe_as_record(probe: dict) -> dict:
    return {
        "id":       probe["id"],
        "claim":    probe["claim"],
        "verdict":  {"label": probe["expected_verdict"]},
        "evidence": probe["evidence"],
    }


def _verdict_icon(verdict: str, expected: str) -> str:
    if verdict == expected:
        return "✅"
    return "❌"


def _run_probes(probes: list[dict], pred, progress_bar) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for i, probe in enumerate(probes):
        progress_bar.progress(
            (i + 1) / len(probes),
            text=f"Running {probe['id']} ({i + 1}/{len(probes)})…",
        )
        try:
            results[probe["id"]] = pred.predict_from_record(_probe_as_record(probe))
        except Exception as exc:
            results[probe["id"]] = {"_error": str(exc), "verdict": "error"}
    return results


def _render_hypothesis_summary(probes_by_id: dict, results: dict[str, dict]) -> None:
    """Tab A: per-hypothesis pass/fail table with EC delta for paired probes."""
    total_pass = sum(
        1 for p in probes_by_id.values()
        if results.get(p["id"], {}).get("verdict") == p["expected_verdict"]
    )
    total = len(probes_by_id)

    col_a, col_b = st.columns([1, 3])
    col_a.metric("Overall Pass Rate", f"{total_pass} / {total}")
    col_b.caption(
        "Each hypothesis tests an isolated epistemic property. "
        "✅ = model verdict matches expected.  "
        "Δ = difference in support_score between paired probes — quantifies the ST or EW effect."
    )

    st.divider()

    for hyp in _HYPOTHESES:
        h_probes = [probes_by_id[pid] for pid in hyp["probes"] if pid in probes_by_id]
        h_pass = sum(
            1 for p in h_probes
            if results.get(p["id"], {}).get("verdict") == p["expected_verdict"]
        )
        h_total = len(h_probes)
        icon = "✅" if h_pass == h_total else ("⚠️" if h_pass > 0 else "❌")

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{icon} {hyp['id']} — {hyp['title']}**")
            c1.caption(hyp["desc"])
            c2.metric("Pass", f"{h_pass}/{h_total}")

            # Per-probe rows
            rows = []
            for p in h_probes:
                r = results.get(p["id"], {})
                if "_error" in r:
                    rows.append({
                        "Probe":      p["id"],
                        "Expected":   p["expected_verdict"],
                        "Got":        f"ERROR: {r['_error'][:40]}",
                        "Result":     "❌",
                        "EC":         "—",
                        "Layer":      "—",
                        "VH pred":    "—",
                        "sup_score":  "—",
                        "ref_score":  "—",
                    })
                else:
                    ec_dec  = r.get("ec_decision",  "—")
                    fl      = r.get("final_layer",  "—")
                    vh_pred = r.get("vh_pred") or "—"
                    layer_icon = "🔬" if fl == "ec_symbolic" else ("🧠" if fl == "verdicthead" else "")
                    rows.append({
                        "Probe":     p["id"],
                        "Expected":  p["expected_verdict"],
                        "Got":       r.get("verdict", "—"),
                        "Result":    _verdict_icon(r.get("verdict", ""), p["expected_verdict"]),
                        "EC":        ec_dec,
                        "Layer":     f"{layer_icon} {fl}",
                        "VH pred":   vh_pred,
                        "sup_score": f"{r.get('support_score', 0):.3f}",
                        "ref_score": f"{r.get('refute_score', 0):.3f}",
                    })
            st.dataframe(rows, hide_index=True, width='stretch')

            # EC delta for paired probes
            for pid_a, pid_b in hyp["pairs"]:
                if pid_a not in results or pid_b not in results:
                    continue
                metric = hyp["pair_metric"]
                if metric is None:
                    continue
                val_a = results[pid_a].get(metric, 0.0)
                val_b = results[pid_b].get(metric, 0.0)
                delta = val_a - val_b
                pa = probes_by_id.get(pid_a, {})
                pb = probes_by_id.get(pid_b, {})

                d1, d2, d3, d4 = st.columns(4)
                d1.metric(
                    f"{pid_a}",
                    f"{val_a:.3f}",
                    help=pa.get("name", ""),
                )
                d2.metric(
                    f"{pid_b}",
                    f"{val_b:.3f}",
                    help=pb.get("name", ""),
                )
                d3.metric(
                    f"Δ {metric.replace('_score','')}",
                    f"{delta:+.3f}",
                    delta=round(delta, 3),
                    delta_color="normal",
                )
                d4.caption(
                    f"`{pid_a}` vs `{pid_b}` — "
                    + ("ST effect" if hyp["id"] == "H1" else "EW effect")
                )


def _render_per_probe_explorer(probes: list[dict], probes_by_id: dict, results: dict, pred, model_key: str) -> None:
    """Tab B: select a probe, show full detail tabs."""
    from app.tabs.verify import _render_result_tabs

    category_labels = {
        "st_contrast":       "Source Trust Contrast",
        "shortcut_breaking": "Shortcut-Breaking",
        "multi_evidence":    "Multi-Evidence",
        "conflicting":       "Conflicting Evidence",
        "is_text_contrast":  "IS Text Contrast",
        "sensor":            "Sensor Observation",
        "source_gradient":   "Source Gradient",
        "evidence_types":    "Evidence Types (EW)",
    }

    options = [f"{p['id']} — {p['name']}" for p in probes]
    selected = st.selectbox(
        "Select probe",
        options,
        key="probe_lab_selected",
        label_visibility="collapsed",
    )
    probe = probes[options.index(selected)]
    result = results.get(probe["id"])

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.caption(f"**Category:** {category_labels.get(probe['category'], probe['category'])}")
        expected = probe["expected_verdict"]
        actual = result.get("verdict", "—") if result and "_error" not in result else "error"
        icon = _verdict_icon(actual, expected) if actual != "error" else "❌"
        c2.caption(f"**Expected:** `{expected}`")
        c3.caption(f"**Got:** {icon} `{actual}`")
        st.info(probe["expected_behavior"])

    if result is None:
        st.info("Run probes to see results.")
        return
    if "_error" in result:
        st.error(f"Prediction error: {result['_error']}")
        return

    if result.get("has_ec"):
        sup = result.get("support_score", 0.0)
        ref = result.get("refute_score", 0.0)
        thr = result.get("ec_threshold", 0.35)
        m1, m2, m3 = st.columns(3)
        m1.metric("Support Score", f"{sup:.4f}")
        m2.metric("Refute Score", f"{ref:.4f}")
        m3.metric("θ (threshold)", f"{thr:.2f}")

    _render_result_tabs(result, pred, model_key)


def _render_model_comparison(probes: list[dict], comparison_results: dict[str, dict[str, dict]]) -> None:
    """Tab C: probe × model matrix showing verdict + pass/fail."""
    from app.config import enum_label
    model_keys = list(comparison_results.keys())
    if not model_keys:
        st.info("Run comparison to see results.")
        return

    header = ["Probe", "Expected"] + model_keys
    rows = []
    for probe in probes:
        pid = probe["id"]
        exp = probe["expected_verdict"]
        row = {"Probe": pid, "Expected": exp}
        for mk in model_keys:
            r = comparison_results[mk].get(pid, {})
            verdict = r.get("verdict", "—")
            icon = _verdict_icon(verdict, exp)
            row[mk] = f"{icon} {verdict}"
        rows.append(row)
    st.dataframe(rows, hide_index=True, width='stretch')

    # Summary row: pass rate per model
    st.caption("Pass rates per model:")
    cols = st.columns(len(model_keys))
    for col, mk in zip(cols, model_keys):
        passed = sum(
            1 for p in probes
            if comparison_results[mk].get(p["id"], {}).get("verdict") == p["expected_verdict"]
        )
        col.metric(mk, f"{passed}/{len(probes)}")


def render(cfg: "AppConfig") -> None:
    from app.core.loaders import get_predictor

    probes = _load_probes()
    if not probes:
        st.warning("No probe data found. Expected: `data/probes/epistemic_probes.jsonl`")
        return

    probes_by_id = {p["id"]: p for p in probes}
    all_model_keys = list(cfg.model_keys)

    st.caption(
        "Run all 20 epistemic probes through one or more models. "
        "Tests 5 hypotheses about source trust, IS cap, and evidence type effects."
    )

    # ── Model selector + run button ───────────────────────────────────────────
    with st.container(border=True):
        col_models, col_btn = st.columns([3, 1])
        selected_models = col_models.multiselect(
            "Models to run",
            all_model_keys,
            default=all_model_keys[:1],
            key="probe_lab_models",
        )
        run_clicked = col_btn.button(
            "▶ Run Probes", type="primary", key="probe_lab_run", width='stretch'
        )

    if run_clicked and selected_models:
        new_results: dict[str, dict] = {}
        outer_bar = st.progress(0.0, text="Starting…")
        for mi, mk in enumerate(selected_models):
            outer_bar.progress(mi / len(selected_models), text=f"Loading {mk}…")
            pred = get_predictor(mk, cfg.graph_cache_dir, cfg.registry_path)
            if isinstance(pred, str):
                st.warning(f"Skipping {mk}: {pred}")
                continue
            inner_bar = st.progress(0.0, text=f"Running {mk}…")
            new_results[mk] = _run_probes(probes, pred, inner_bar)
            inner_bar.empty()
        outer_bar.empty()
        st.session_state["probe_lab_all_results"] = new_results
        st.rerun()

    all_results: dict[str, dict] = st.session_state.get("probe_lab_all_results", {})
    run_models = list(all_results.keys())

    if not all_results:
        st.info("Select one or more models and click **▶ Run Probes** to start.")
        return

    # ── Per-model pass rate header ────────────────────────────────────────────
    pass_cols = st.columns(len(run_models))
    for col, mk in zip(pass_cols, run_models):
        n_pass = sum(
            1 for p in probes
            if all_results[mk].get(p["id"], {}).get("verdict") == p["expected_verdict"]
        )
        col.metric(mk, f"{n_pass} / {len(probes)}", help="Probes passed")

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_labels = ["📊 Hypothesis Summary", "🔬 Per-Probe Explorer"]
    if len(run_models) >= 2:
        tab_labels.append("⚖️ Model Comparison")

    tabs = st.tabs(tab_labels)

    # Tab A — Hypothesis summary (model picker when multiple models ran)
    with tabs[0]:
        if len(run_models) > 1:
            hyp_model = st.selectbox(
                "Show hypothesis detail for",
                run_models,
                key="probe_lab_hyp_model",
            )
        else:
            hyp_model = run_models[0]
        _render_hypothesis_summary(probes_by_id, all_results[hyp_model])

    # Tab B — Per-probe explorer (model picker when multiple models ran)
    with tabs[1]:
        if len(run_models) > 1:
            explorer_model = st.selectbox(
                "Model",
                run_models,
                key="probe_lab_explorer_model",
            )
        else:
            explorer_model = run_models[0]
        pred = get_predictor(explorer_model, cfg.graph_cache_dir, cfg.registry_path)
        if isinstance(pred, str):
            st.error(f"Model not available: {pred}")
        else:
            _render_per_probe_explorer(
                probes, probes_by_id, all_results[explorer_model], pred, explorer_model
            )

    # Tab C — Model comparison (only when ≥ 2 models)
    if len(run_models) >= 2:
        with tabs[2]:
            _render_model_comparison(probes, all_results)
