"""Verify tab — live claim verification."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig

_MAX_EV = 4
_PROBES_PATH = Path("data/probes/epistemic_probes.jsonl")

_CATEGORY_LABELS = {
    "st_contrast":       "Source Trust Contrast",
    "shortcut_breaking": "Shortcut-Breaking",
    "multi_evidence":    "Multi-Evidence",
    "conflicting":       "Conflicting Evidence",
    "is_text_contrast":  "IS Text Contrast",
    "sensor":            "Sensor Observation",
    "source_gradient":   "Source Gradient",
    "evidence_types":    "Evidence Types (EW)",
}


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
_STANCE_EMOJI = {
    "supports":             "✅",
    "refutes":              "❌",
    "not_enough_evidence":  "~",
}
_BAR_W = 16


def _bar(pct: float) -> str:
    n = round(pct / 100 * _BAR_W)
    return "█" * n + "░" * (_BAR_W - n)


def _render_result_tabs(result: dict, predictor, model_key: str) -> None:
    """Render the 5 result sub-tabs for a single prediction result."""
    from app.config import enum_label
    from app.core.ui import (
        render_decision_path, render_arch_pipeline_bar,
        render_layerwise, render_debug_view,
        build_model_computation_dot, build_pyvis_html,
    )

    t_ev, t_flow, t_layer, t_graph, t_debug = st.tabs([
        "Evidence", "Computation Flow", "Layerwise", "Claim Graph", "Debug",
    ])

    with t_ev:
        render_arch_pipeline_bar(model_key, result.get("has_ec", False))
        render_decision_path(predictor.decision_path_info(result))
        st.caption("Evidence breakdown")
        breakdown = result.get("evidence_breakdown", [])
        for i, ev in enumerate(breakdown):
            stance  = ev.get("stance", "not_enough_evidence")
            s_emoji = _STANCE_EMOJI.get(stance, "~")
            ec      = ev.get("ec_score", 0.0)
            with st.expander(
                f"{s_emoji} Evidence {i + 1}  ·  {enum_label(stance)}  ·  EC {ec:.3f}",
                expanded=i == 0,
            ):
                st.markdown(ev.get("text_short") or ev.get("text", ""))
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("IS",           f"{ev.get('is_score',           0):.3f}")
                m2.metric("Source Trust", f"{ev.get('source_trust',       0):.3f}")
                m3.metric("Ev. Weight",   f"{ev.get('evidence_weight',    0):.3f}")
                m4.metric("Sup. Conf.",   f"{ev.get('support_confidence', 0):.3f}")
                m5.metric("Ref. Conf.",   f"{ev.get('refute_confidence',  0):.3f}")
                st.caption(
                    f"Modality: {enum_label(ev.get('modality', ''))}  ·  "
                    f"Source: {enum_label(ev.get('source_type', ''))}  ·  "
                    f"ID: `{ev.get('source_id', '')}`"
                )
                nli = ev.get("nli_probs")
                if nli:
                    st.caption(
                        f"NLI  entail {nli['entailment']:.3f}  ·  "
                        f"contra {nli['contradiction']:.3f}  ·  "
                        f"neutral {nli['neutral']:.3f}"
                    )

    with t_flow:
        st.caption(
            "**Computation DAG** — L0 Raw inputs → L1 Encoder → L2 H1 StanceHead → "
            "L3 H2 IS Head → L4 EC Formula → L5 Aggregation → L6 Decision → L7 Verdict"
        )
        try:
            dot = build_model_computation_dot(result, model_key)
            st.graphviz_chart(dot, width='stretch')
            with st.expander("DOT source", expanded=False):
                st.code(dot, language=None)
        except Exception as e:
            st.warning(f"Computation flow unavailable: {e}")

    with t_layer:
        render_layerwise(predictor.evidence_table(result), result=result,
                         true_label=st.session_state.get("verify_true_label"))

    with t_graph:
        hetero_data = result.get("hetero_data")
        ev_texts    = [ev.get("text", "") for ev in result.get("evidence_breakdown", [])]
        claim_text  = result.get("claim_text", "")
        if hetero_data is not None:
            st.caption(
                "**Blue** = CLAIM  ·  **green** = supports  ·  **red** = refutes  ·  **gray** = neutral  \n"
                "**Solid blue** edges = has_evidence  ·  **orange dashed** = connected_to  ·  "
                "**gray dashed** = co_evidence"
            )
            try:
                html = build_pyvis_html(hetero_data, claim_text, ev_texts)
                st.iframe(html, height=560)
            except Exception as e:
                st.warning(f"Interactive graph unavailable: {e}")
        else:
            try:
                st.graphviz_chart(predictor.result_dot(result), width='stretch')
            except Exception as e:
                st.warning(f"Claim graph unavailable: {e}")
        with st.expander("DOT fallback view", expanded=False):
            try:
                st.graphviz_chart(predictor.result_dot(result), width='stretch')
            except Exception as e:
                st.warning(f"DOT unavailable: {e}")

    with t_debug:
        render_debug_view(result)


def render(cfg: "AppConfig") -> None:
    from app.config import enum_label
    from app.core.loaders import get_predictor
    from app.core.state import init_state, load_by_id

    # Must run before any widget renders — transfers _pending_claim → claim_input
    init_state(cfg)

    model_keys = list(cfg.model_keys)

    # ── Model selector ────────────────────────────────────────────────────────
    selected_models: list[str] = st.multiselect(
        "Models",
        model_keys,
        default=[model_keys[0]],
        key="verify_models",
        help="Select 1–3 models. Multiple models show side-by-side tab results.",
    )
    # if len(selected_models) > 3:
    #     st.warning("Select at most 3 models for side-by-side comparison.")
    #     selected_models = selected_models[:3]
    # if not selected_models:
    #     st.info("Select at least one model above.")
    #     return

    # ── Epistemic probe selector ──────────────────────────────────────────────
    probes = _load_probes()
    if probes:
        with st.expander("Epistemic Probes — load a hand-crafted test case", expanded=False):
            by_cat: dict[str, list[dict]] = {}
            for p in probes:
                by_cat.setdefault(p["category"], []).append(p)

            probe_options = ["— select a probe —"]
            probe_map: dict[str, dict] = {}
            for cat, cat_probes in by_cat.items():
                cat_label = _CATEGORY_LABELS.get(cat, cat)
                for p in cat_probes:
                    label = f"[{cat_label}]  {p['id']} — {p['name']}"
                    probe_options.append(label)
                    probe_map[label] = p

            selected_label = st.selectbox(
                "Select probe",
                probe_options,
                key="probe_selector",
                label_visibility="collapsed",
            )

            if selected_label != "— select a probe —":
                probe = probe_map[selected_label]
                st.info(f"**Expected:** {probe['expected_behavior']}")
                st.caption(
                    f"Expected verdict: `{probe['expected_verdict']}`  ·  "
                    f"Category: `{probe['category']}`"
                )
                if st.button("Load Probe", key="probe_load_btn", type="secondary"):
                    from app.core.state import load_record_into_state
                    probe_rec = {
                        "claim":   probe["claim"],
                        "verdict": {"label": probe["expected_verdict"]},
                        "evidence": probe["evidence"],
                        "id":      probe["id"],
                    }
                    load_record_into_state(probe_rec, cfg)
                    st.session_state["_probe_loaded"]      = probe["id"]
                    st.session_state["_random_true_label"] = probe["expected_verdict"]
                    st.rerun()

    # ── Claim-load controls ───────────────────────────────────────────────────
    sources = st.multiselect(
        "Sample from",
        ["test", "val", "probe"],
        default=["test"],
        key="verify_sources",
        help="Random draws from the union of selected pools.",
    )

    c_rand, c_id, c_load = st.columns([1, 4, 1])

    rand_clicked = c_rand.button("Random", key="verify_rand", width='stretch')
    cid = c_id.text_input(
        "Claim ID", label_visibility="collapsed",
        placeholder="Claim ID", key="verify_cid",
    )
    load_clicked = c_load.button("Load", key="verify_load", width='stretch')

    if rand_clicked:
        from app.core.state import load_random_from_sources
        load_random_from_sources(sources or ["test"], cfg)
        st.rerun()

    if load_clicked:
        if cid.strip():
            if not load_by_id(cid.strip(), cfg):
                st.warning(f"ID '{cid.strip()}' not found.")
            else:
                st.rerun()

    # ── Claim ─────────────────────────────────────────────────────────────────
    loaded_id = st.session_state.get("current_claim_id")
    rec_source = st.session_state.get("_loaded_source")
    if loaded_id:
        src_label = f"  ·  Source: `{rec_source}`" if rec_source else ""
        st.caption(f"ID: `{loaded_id}`{src_label}")

    claim = st.text_area(
        "Claim",
        value=st.session_state.get("last_claim", ""),
        height=80,
        placeholder="Enter the claim to verify…",
        key="claim_input",
    )

    # ── Evidence items ────────────────────────────────────────────────────────
    src_types = list(cfg.source_type_values)

    ev_list: list[dict] = st.session_state.setdefault(
        "evidence_list",
        [{"text": "", "modality": "web_text", "source_type": "unknown"}],
    )
    n = len(ev_list)
    st.caption(f"Evidence — {n} item{'s' if n != 1 else ''}  ·  max {_MAX_EV}")

    to_remove: int | None = None

    for i, ev in enumerate(ev_list):
        with st.container(border=True):
            h1, h2 = st.columns([6, 1])
            h1.markdown(f"**#{i + 1}**")
            if n > 1 and h2.button("✕", key=f"rm_{i}"):
                to_remove = i
                continue

            text = st.text_area(
                "Text",
                value=st.session_state.get(f"ev_{i}", ev.get("text", "")),
                height=72,
                key=f"ev_{i}",
                label_visibility="collapsed",
                placeholder="Paste evidence text…",
            )

            c1, c2 = st.columns(2)
            mod_vals = list(cfg.modality_values)
            ev_mod = ev.get("modality", "web_text")
            modality = c1.selectbox(
                "Modality",
                mod_vals,
                index=mod_vals.index(ev_mod) if ev_mod in mod_vals else 0,
                format_func=enum_label,
                key=f"mod_{i}",
            )
            ev_src = ev.get("source_type", "unknown")
            source_type = c2.selectbox(
                "Source Type",
                src_types,
                index=src_types.index(ev_src) if ev_src in src_types else src_types.index("unknown"),
                format_func=enum_label,
                key=f"src_{i}",
            )
            updated = {
                "text":        text,
                "modality":    modality,
                "source_type": source_type,
            }
            if ev.get("source_id"):
                updated["source_id"] = ev["source_id"]
                st.caption(f"Source ID: `{ev['source_id']}`")
            ev_list[i] = updated

    if to_remove is not None:
        ev_list.pop(to_remove)
        st.session_state["evidence_list"] = ev_list
        st.rerun()

    st.session_state["evidence_list"] = ev_list

    # ── Action buttons ────────────────────────────────────────────────────────
    ba, _, br = st.columns([1, 2, 1])
    if ba.button("＋ Add Evidence", disabled=n >= _MAX_EV,
                 width='stretch', key="verify_add_ev"):
        ev_list.append({"text": "", "modality": "web_text", "source_type": "unknown"})
        st.session_state["evidence_list"] = ev_list
        st.rerun()

    run = br.button("🔍 Verify", type="primary", width='stretch', key="verify_run")

    # ── Inference ─────────────────────────────────────────────────────────────
    if run:
        if not claim.strip():
            st.warning("Enter a claim before verifying.")
            st.stop()
        filled = [e for e in ev_list if (e.get("text") or "").strip()]
        if not filled:
            st.warning("Add at least one non-empty evidence item.")
            st.stop()

        st.session_state["last_claim"] = claim

        results: dict[str, dict | str] = {}
        with st.spinner(f"Running {', '.join(selected_models)}…"):
            for mk in selected_models:
                pred = get_predictor(mk, cfg.graph_cache_dir, cfg.registry_path)
                if isinstance(pred, str):
                    results[mk] = pred
                else:
                    try:
                        results[mk] = pred.predict(claim.strip(), filled)
                    except Exception as exc:
                        results[mk] = str(exc)

        st.session_state["verify_results"]     = results
        st.session_state["verify_models_used"] = selected_models
        st.session_state["verify_true_label"]  = st.session_state.get("_random_true_label")

    # ── Results ───────────────────────────────────────────────────────────────
    results     = st.session_state.get("verify_results")
    models_used = st.session_state.get("verify_models_used", [])

    if not results:
        return

    st.divider()
    true_label = st.session_state.get("verify_true_label")

    if len(models_used) == 1:
        mk     = models_used[0]
        result = results.get(mk)
        if result is None:
            return
        if isinstance(result, str):
            st.error(f"Model error ({mk}): {result}")
            return

        predictor = get_predictor(mk, cfg.graph_cache_dir, cfg.registry_path)
        if isinstance(predictor, str):
            st.error(f"Model not available: {predictor}")
            return

        verdict = result["verdict"]
        vd      = cfg.verdict_display.get(verdict, {})
        emoji   = vd.get("emoji", "❓")

        vc, pc = st.columns([1, 2])
        with vc:
            with st.container(border=True):
                st.markdown(f"## {emoji} {enum_label(verdict)}")
                st.caption(f"Model: `{mk}`")
                if result.get("has_ec"):
                    sup = result.get("support_score", 0.0)
                    ref = result.get("refute_score",  0.0)
                    thr = result.get("ec_threshold",  0.35)
                    st.caption(f"EC  sup {sup:.2f}  ·  ref {ref:.2f}  ·  θ {thr:.2f}")
                if true_label:
                    match = true_label == verdict
                    st.caption(f"{'✅' if match else '❌'} True: **{enum_label(true_label)}**")

        with pc:
            vprobs = result.get("verdict_probs", [])
            if len(vprobs) == 3:
                with st.container(border=True):
                    st.caption("Verdict probabilities")
                    for idx, prob in enumerate(vprobs):
                        lbl = enum_label(cfg.int_to_verdict.get(idx, str(idx)))
                        pct = prob * 100
                        st.markdown(f"`{lbl:<22}` `{_bar(pct)}` **{pct:.1f}%**")

        _render_result_tabs(result, predictor, mk)

    else:
        model_tabs = st.tabs(models_used)
        for tab, mk in zip(model_tabs, models_used):
            with tab:
                result = results.get(mk)
                if result is None:
                    st.info("No result.")
                    continue
                if isinstance(result, str):
                    st.error(f"Error: {result}")
                    continue

                predictor = get_predictor(mk, cfg.graph_cache_dir, cfg.registry_path)
                if isinstance(predictor, str):
                    st.error(f"Model not available: {predictor}")
                    continue

                verdict = result["verdict"]
                vd      = cfg.verdict_display.get(verdict, {})
                emoji   = vd.get("emoji", "❓")

                vc, pc = st.columns([1, 2])
                with vc:
                    with st.container(border=True):
                        st.markdown(f"## {emoji} {enum_label(verdict)}")
                        st.caption(f"Model: `{mk}`")
                        if result.get("has_ec"):
                            sup = result.get("support_score", 0.0)
                            ref = result.get("refute_score",  0.0)
                            thr = result.get("ec_threshold",  0.35)
                            st.caption(f"EC  sup {sup:.2f}  ·  ref {ref:.2f}  ·  θ {thr:.2f}")
                        if true_label:
                            match = true_label == verdict
                            st.caption(f"{'✅' if match else '❌'} True: **{enum_label(true_label)}**")

                with pc:
                    vprobs = result.get("verdict_probs", [])
                    if len(vprobs) == 3:
                        with st.container(border=True):
                            st.caption("Verdict probabilities")
                            for idx, prob in enumerate(vprobs):
                                lbl = enum_label(cfg.int_to_verdict.get(idx, str(idx)))
                                pct = prob * 100
                                st.markdown(f"`{lbl:<22}` `{_bar(pct)}` **{pct:.1f}%**")

                _render_result_tabs(result, predictor, mk)
