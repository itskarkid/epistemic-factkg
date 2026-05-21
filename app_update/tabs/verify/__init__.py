"""Verify tab — live claim verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app_update.config import AppConfig

_MAX_EV = 4
_STANCE_EMOJI = {
    "supports":             "✅",
    "refutes":              "❌",
    "not_enough_evidence":  "~",
}
_BAR_W = 16


def _bar(pct: float) -> str:
    n = round(pct / 100 * _BAR_W)
    return "█" * n + "░" * (_BAR_W - n)


def render(cfg: "AppConfig") -> None:
    from app_update.config import enum_label
    from app_update.core.loaders import get_predictor, registry_source_types
    from app_update.core.state import load_random_example, load_by_id

    src_types = registry_source_types(cfg.registry_path)

    # ── Top controls ──────────────────────────────────────────────────────
    c_model, c_rand, c_form = st.columns([2, 1, 2])

    with c_model:
        model_key = st.selectbox(
            "Model",
            cfg.model_keys,
            format_func=lambda k: cfg.model_descriptions.get(k, k),
            key="verify_model",
        )

    with c_rand:
        st.write("")
        if st.button("🎲 Example", use_container_width=True, key="verify_rand"):
            load_random_example(cfg)
            st.rerun()

    with c_form:
        with st.form("verify_load_id", clear_on_submit=True):
            fi, fb = st.columns([3, 1])
            cid = fi.text_input("Load by ID", label_visibility="collapsed", placeholder="Claim ID")
            if fb.form_submit_button("Load") and cid.strip():
                if not load_by_id(cid.strip(), cfg):
                    st.warning(f"ID '{cid.strip()}' not found.")
                else:
                    st.rerun()

    # ── Claim ─────────────────────────────────────────────────────────────
    claim = st.text_area(
        "Claim",
        value=st.session_state.get("last_claim", ""),
        height=80,
        placeholder="Enter the claim to verify…",
        key="claim_input",
    )

    # ── Evidence items ────────────────────────────────────────────────────
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

            c1, c2, c3 = st.columns(3)
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
            url = c3.text_input(
                "URL (optional)",
                value=ev.get("url", ""),
                key=f"url_{i}",
                placeholder="https://…",
            )
            ev_list[i] = {
                "text":        text,
                "modality":    modality,
                "source_type": source_type,
                "url":         url or None,
            }

    if to_remove is not None:
        ev_list.pop(to_remove)
        st.session_state["evidence_list"] = ev_list
        st.rerun()

    st.session_state["evidence_list"] = ev_list

    # ── Action buttons ────────────────────────────────────────────────────
    ba, _, br = st.columns([1, 2, 1])
    if ba.button("＋ Add Evidence", disabled=n >= _MAX_EV, use_container_width=True):
        ev_list.append({"text": "", "modality": "web_text", "source_type": "unknown"})
        st.session_state["evidence_list"] = ev_list
        st.rerun()

    run = br.button("🔍 Verify", type="primary", use_container_width=True)

    # ── Inference ─────────────────────────────────────────────────────────
    if run:
        if not claim.strip():
            st.warning("Enter a claim before verifying.")
            st.stop()
        filled = [e for e in ev_list if (e.get("text") or "").strip()]
        if not filled:
            st.warning("Add at least one non-empty evidence item.")
            st.stop()

        st.session_state["last_claim"] = claim

        predictor = get_predictor(model_key, cfg.graph_cache_dir)
        if isinstance(predictor, str):
            st.error(f"Model not loaded: {predictor}")
            st.stop()

        with st.spinner("Running inference…"):
            try:
                result = predictor.predict(claim, filled)
                st.session_state["verify_result"]     = result
                st.session_state["verify_model_used"] = model_key
                st.session_state["verify_true_label"] = st.session_state.get("_random_true_label")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

    # ── Results ───────────────────────────────────────────────────────────
    result = st.session_state.get("verify_result")
    if result is None:
        return

    st.divider()

    verdict    = result["verdict"]
    vd         = cfg.verdict_display.get(verdict, {})
    emoji      = vd.get("emoji", "❓")
    true_label = st.session_state.get("verify_true_label")
    model_used = st.session_state.get("verify_model_used", model_key)

    # Verdict header
    vc, pc = st.columns([1, 2])
    with vc:
        with st.container(border=True):
            st.markdown(f"## {emoji} {enum_label(verdict)}")
            st.caption(f"Model: `{model_used}`")
            if result.get("has_ec"):
                sup = result.get("support_score", 0.0)
                ref = result.get("refute_score",  0.0)
                thr = result.get("ec_threshold",  0.35)
                st.caption(f"EC  sup {sup:.2f}  ·  ref {ref:.2f}  ·  θ {thr:.2f}")
            if true_label:
                match = true_label == verdict
                st.caption(f"{'✅' if match else '❌'} True label: **{enum_label(true_label)}**")

    with pc:
        vprobs = result.get("verdict_probs", [])
        if len(vprobs) == 3:
            with st.container(border=True):
                st.caption("Verdict probabilities")
                for idx, prob in enumerate(vprobs):
                    lbl = enum_label(cfg.int_to_verdict.get(idx, str(idx)))
                    pct = prob * 100
                    st.markdown(f"`{lbl:<22}` `{_bar(pct)}` **{pct:.1f}%**")

    # Result sub-tabs
    t_ev, t_flow, t_layer, t_graph = st.tabs([
        "Evidence", "Computation Flow", "Layerwise", "Claim Graph",
    ])

    with t_ev:
        from app_update.core.ui import render_decision_path
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
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("IS",           f"{ev.get('is_score',          0):.3f}")
                m2.metric("Source Trust", f"{ev.get('source_trust',      0):.3f}")
                m3.metric("Ev. Weight",   f"{ev.get('evidence_weight',   0):.3f}")
                m4.metric("Stance Conf.", f"{ev.get('stance_confidence', 0):.3f}")
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
        try:
            raw_out = result.get("_raw_out")
            dot = predictor.arc_block_definition(raw_out).to_dot()
            st.graphviz_chart(dot)
        except Exception as e:
            st.warning(f"Computation flow unavailable: {e}")

    with t_layer:
        from app_update.core.ui import render_layerwise
        render_layerwise(predictor.evidence_table(result), result=result, true_label=true_label)

    with t_graph:
        try:
            st.graphviz_chart(predictor.result_dot(result))
        except Exception as e:
            st.warning(f"Claim graph unavailable: {e}")
