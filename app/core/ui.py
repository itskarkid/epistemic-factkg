"""Thin Streamlit rendering layer for model outputs.

All data is produced by src/ methods (decision_path_info, evidence_table,
build_pyvis_html is the only exception — pyvis is a visualization library).
These functions consume dicts/lists and render Streamlit widgets.
"""

from __future__ import annotations

import streamlit as st


def render_decision_path(path_info: dict) -> None:
    """Render a two-layer EC vs VerdictHead decision callout."""
    has_ec      = path_info.get("has_ec", False)
    ec_decision = path_info.get("ec_decision")
    final_layer = path_info.get("final_layer", "verdicthead")
    vh_pred     = path_info.get("vh_pred")
    sup         = path_info.get("support_score", 0.0)
    ref         = path_info.get("refute_score",  0.0)
    thr         = path_info.get("ec_threshold",  0.35)

    if not has_ec:
        st.info("**Baseline (No EC)** — No epistemic confidence formula.")
        return

    scores = f"`sup={sup:.3f}` · `ref={ref:.3f}` · `θ={thr:.2f}`"

    if final_layer == "ec_symbolic":
        st.success(
            f"**🔬 EC Layer → {ec_decision}**  \n{scores}"
        )
        if vh_pred:
            verdict = path_info.get("verdict", ec_decision)
            if vh_pred == verdict:
                st.caption(f"VerdictHead also predicted: `{vh_pred}` ✓")
            else:
                st.caption(f"VerdictHead predicted: `{vh_pred}` (overridden by EC symbolic decision)")
    elif ec_decision == "conflicted":
        st.warning(
            f"**🧠 VerdictHead → conflict resolved**  \n"
            f"EC conflicted — both sides crossed θ ({scores})."
        )
    else:
        st.warning(
            f"**🧠 VerdictHead → decided**  \n"
            f"EC deferred — both sides below θ ({scores})."
        )


_STANCE_EMOJI = {
    "supports":            "🟢",
    "refutes":             "🔴",
    "not_enough_evidence": "⚪",
}

_VERDICT_EMOJI = {
    "supported":            "✅",
    "refuted":              "❌",
    "not_enough_evidence":  "〰️",
    "conflicting_evidence": "⚡",
}

_VERDICT_LABELS = ["supported", "refuted", "not_enough_evidence"]


def render_layerwise(
    rows: list[dict],
    result: dict | None = None,
    true_label: str | None = None,
) -> None:
    """Render step-by-step reasoning trace from evidence_table() rows + optional full result dict.

    Pass result= to show the EC aggregation and verdict probability section.
    """
    from app.config import enum_label

    if not rows:
        st.caption("No evidence items.")
        return

    # ── True label comparison ──────────────────────────────────────────────
    if result is not None and true_label is not None:
        verdict  = result.get("verdict", "")
        v_emoji  = _VERDICT_EMOJI.get(verdict, "❓")
        t_emoji  = _VERDICT_EMOJI.get(true_label, "❓")
        match    = verdict == true_label
        st.markdown(
            f"{'✅' if match else '❌'} &nbsp; "
            f"True: **{t_emoji} {enum_label(true_label)}** &nbsp;·&nbsp; "
            f"Predicted: **{v_emoji} {enum_label(verdict)}**",
            unsafe_allow_html=True,
        )
        st.divider()

    has_ec = (result or {}).get("has_ec", False)

    # ── Per-evidence layer table ───────────────────────────────────────────
    for row in rows:
        preview = row["text_short"][:100] + ("…" if len(row["text_short"]) > 100 else "")
        st.markdown(f"**Evidence {row['idx'] + 1}**  ·  *{preview}*")

        md_rows: list[tuple[str, str]] = []

        if row.get("nli_probs"):
            nli = row["nli_probs"]
            e, c, n = nli["entailment"], nli["contradiction"], nli["neutral"]
            md_rows.append((
                "① NLI input features",
                f"entail `{e:.1%}` &nbsp; contra `{c:.1%}` &nbsp; neutral `{n:.1%}`  *(offline DeBERTa)*",
            ))

        stance   = row["stance"]
        s_emoji  = _STANCE_EMOJI.get(stance, "⚪")
        sup_conf = row.get("support_confidence", 0.0)
        ref_conf = row.get("refute_confidence",  0.0)
        md_rows.append(("② Stance (H1)", f"{s_emoji} **{enum_label(stance)}** &nbsp; p_sup `{sup_conf:.3f}` · p_ref `{ref_conf:.3f}`"))
        md_rows.append(("③ IS (H2)",     f"`{row['is_score']:.3f}`"))

        if has_ec:
            ec_v     = row["ec_score"]
            sup_contribution = round(ec_v * sup_conf, 4)
            ref_contribution = round(ec_v * ref_conf, 4)
            md_rows.append((
                "④ EC formula",
                f"EC=`{ec_v:.3f}` &nbsp; ST `{row['source_trust']:.2f}` · EW `{row['evidence_weight']:.2f}`(type) · IS `{row['is_score']:.3f}`  \n"
                f"soft\\_support\\_contribution = EC × p\\_sup = `{ec_v:.3f}` × `{sup_conf:.3f}` = `{sup_contribution:.4f}`  \n"
                f"soft\\_refute\\_contribution  = EC × p\\_ref = `{ec_v:.3f}` × `{ref_conf:.3f}` = `{ref_contribution:.4f}`",
            ))

        md_rows.append((
            "⑤ Source",
            f"{enum_label(row['modality'])} · {enum_label(row['source_type'])}",
        ))

        st.markdown(
            "| Layer | Value |\n|---|---|\n"
            + "".join(f"| {label} | {value} |\n" for label, value in md_rows)
        )
        if row["idx"] < len(rows) - 1:
            st.markdown("")

    # ── Aggregation + verdict section ──────────────────────────────────────
    if result is None:
        return

    st.divider()
    verdict = result.get("verdict", "")

    if has_ec:
        sup = result.get("support_score", 0.0)
        ref = result.get("refute_score",  0.0)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**⑥ EC aggregation** &nbsp; support `{sup:.3f}` / refute `{ref:.3f}`")
            st.progress(min(sup, 1.0), text=f"🟢 support {sup:.3f}")
            st.progress(min(ref, 1.0), text=f"🔴 refute  {ref:.3f}")
        with c2:
            probs = result.get("verdict_probs", [])
            st.markdown("**⑦ Verdict head**")
            for lbl, p in zip(_VERDICT_LABELS, probs):
                emoji = _VERDICT_EMOJI.get(lbl, "❓")
                st.progress(p, text=f"{emoji} {enum_label(lbl)} {p:.0%}")
    else:
        probs = result.get("verdict_probs", [])
        st.markdown("**④ Verdict head**")
        for lbl, p in zip(_VERDICT_LABELS, probs):
            emoji = _VERDICT_EMOJI.get(lbl, "❓")
            st.progress(p, text=f"{emoji} {enum_label(lbl)} {p:.0%}")

    v_emoji = _VERDICT_EMOJI.get(verdict, "❓")
    st.markdown(f"**Final verdict: {v_emoji} {enum_label(verdict).upper()}**")


# ── Architecture pipeline bar ──────────────────────────────────────────────────

def render_arch_pipeline_bar(model_key: str, has_ec: bool) -> None:
    """Render a compact colored pipeline bar for the given model."""
    stages = [
        ("`① Input`",       "#6b7280"),
        ("`② GNN Encoder`", "#1d4ed8"),
    ]
    stages.append(("`③ H1+H2`", "#b45309"))
    if has_ec:
        stages.append(("`④⑤ EC`",      "#15803d"))
        stages.append(("`⑥ VerdictHead`", "#991b1b"))
    else:
        stages.append(("`④ VerdictMLP`", "#991b1b"))

    arrow = " → "
    parts = []
    for label, color in stages:
        parts.append(
            f'<span style="background:{color};color:#fff;padding:2px 9px;'
            f'border-radius:10px;font-size:0.72rem;font-weight:600;white-space:nowrap">'
            f'{label.strip("`")}</span>'
        )
    bar = (
        '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;'
        'padding:8px 12px;background:#f8fafc;border:1px solid #e5e7eb;'
        'border-radius:8px;margin-bottom:8px">'
        + (' <span style="color:#9ca3af">→</span> '.join(parts))
        + '</div>'
    )
    st.markdown(bar, unsafe_allow_html=True)


# ── Model computation flow (DOT) ───────────────────────────────────────────────

def build_model_computation_dot(result: dict, model_name: str) -> str:
    """Build a detailed L0–L7 Graphviz DOT string for the inference pipeline.

    Ported from app/_ui.py:build_model_computation_dot with an enhanced
    decision node that shows all values + highlights the active branch.
    """
    bd      = result.get("evidence_breakdown", [])
    has_ec  = result.get("has_ec", False)
    verdict = result.get("verdict", "not_enough_evidence")
    sup     = result.get("support_score", 0.0)
    ref     = result.get("refute_score",  0.0)
    is_nli  = (model_name == "v3-nli")
    n       = len(bd)

    v_fill = {"supported": "#166534", "refuted": "#991b1b", "not_enough_evidence": "#92400e"}
    s_fill = {"supports": "#d1fae5", "refutes": "#fee2e2", "neutral": "#f3f4f6"}
    s_fc   = {"supports": "#14532d", "refutes": "#7f1d1d", "neutral": "#374151"}

    def _esc(s: str) -> str:
        return s.replace('"', "'").replace("\n", " ").replace("\\", "/")

    lines = [
        "digraph GNN_Flow {",
        '  graph [bgcolor="#f8fafc", rankdir=TB, pad="0.5", nodesep=1.0, ranksep=0.7,',
        f'         label="Model Computation Flow — {model_name}", labelloc=t,',
        '         fontname="Arial", fontsize=11, center=true]',
        '  node  [fontname="Arial", fontsize=9, style=filled, margin="0.12,0.06"]',
        '  edge  [fontname="Arial", fontsize=8, color="#6b7280"]',
        "",
    ]

    # L0: INPUTS
    lines += ['  { rank=same']
    for i, ev in enumerate(bd):
        text_s  = _esc((ev.get("text_short") or ev.get("text", ""))[:40])
        src     = ev.get("source_type", "?")
        st_v    = ev.get("source_trust", 0.0)
        nli_p   = ev.get("nli_probs") if is_nli else None
        if nli_p:
            nli_line = f"\\nNLI feat: ent={nli_p['entailment']:.3f} con={nli_p['contradiction']:.3f}"
        else:
            nli_line = ""
        lines.append(
            f'    ev_in_{i} [label="EV {i+1} input\\n{text_s}\\n'
            f'ST={st_v:.3f}  src={src}{nli_line}",'
            f' shape=rect, fillcolor="#f1f5f9", fontcolor="#1e293b"]'
        )
    lines += ['  }', '']

    # L1: GNN ENCODER (always GAT at runtime; NLI probs are input features for v3-nli, not a runtime step)
    lines += ['  { rank=same']
    for i in range(n):
        lines.append(
            f'    ev_enc_{i} [label="GNN Encoder\\nHeteroConv GAT",'
            f' shape=rect, fillcolor="#dbeafe", fontcolor="#1e40af"]'
        )
    lines += ['  }', '']

    # L2: H1 STANCE HEAD — all models predict stance from GNN encoder output
    lines += ['  { rank=same']
    for i, ev in enumerate(bd):
        stance   = ev.get("stance", "neutral")
        sup_conf = ev.get("support_confidence", 0.0)
        ref_conf = ev.get("refute_confidence",  0.0)
        lbl = f"H1: StanceHead\\n→ {stance}\\np_sup={sup_conf:.3f}  p_ref={ref_conf:.3f}"
        lines.append(
            f'    ev_h1_{i} [label="{lbl}", shape=rect,'
            f' fillcolor="{s_fill.get(stance, "#f3f4f6")}",'
            f' fontcolor="{s_fc.get(stance, "#374151")}"]'
        )
    lines += ['  }', '']

    # L3: H2 IS HEAD
    lines += ['  { rank=same']
    for i, ev in enumerate(bd):
        is_v = ev.get("is_score", 0.0)
        lines.append(
            f'    ev_h2_{i} [label="H2: ISHead\\nIS={is_v:.4f}",'
            f' shape=rect, fillcolor="#fef9c3", fontcolor="#713f12"]'
        )
    lines += ['  }', '']

    # Source Registry node (shared, outside rank groups) — feeds ST to EC formula
    if has_ec:
        lines.append(
            '  sr [label="Source Registry\\nsource_id → ST lookup\\n(graph-build time)",'
            ' shape=cylinder, fillcolor="#f1f5f9", fontcolor="#374151"]'
        )
        lines.append('')

    # L4: EC FORMULA — uses ST (registry) + EW (evidence type, pre-computed) + IS (H2)
    # H1 stance probs do NOT feed EC; they weight the aggregation in L5.
    if has_ec:
        lines += ['  { rank=same']
        for i, ev in enumerate(bd):
            is_v = ev.get("is_score", 0.0)
            ew_v = ev.get("evidence_weight", 0.0)
            st_v = ev.get("source_trust", 0.0)
            ec_v = ev.get("ec_score", 0.0)
            lines.append(
                f'    ev_ec_{i} [label="EC Formula\\n'
                f'ST={st_v:.3f}  EW={ew_v:.3f}(type)  IS={is_v:.3f}\\n'
                f'1-(1-{st_v:.3f})^({ew_v:.3f}×{is_v:.3f}) = {ec_v:.4f}",'
                f' shape=rect, fillcolor="#f3e8ff", fontcolor="#6b21a8"]'
            )
        lines += ['  }', '']

    # L5: AGGREGATION — soft formula: 1 - ∏(1 - EC_i × p_stance_i)
    # support uses p_sup per evidence; refute uses p_ref per evidence.
    if has_ec:
        sup_ev = [ev for ev in bd if ev.get("stance") == "supports"]
        ref_ev = [ev for ev in bd if ev.get("stance") == "refutes"]
        if sup_ev:
            sup_terms = " × ".join(
                f"(1-{ev.get('ec_score',0):.3f}×{ev.get('support_confidence',0):.3f})"
                for ev in sup_ev
            )
            sup_line = f"sup = 1 − {sup_terms} = {sup:.4f}"
        else:
            sup_line = "sup = 0.0000  (no supporters)"
        if ref_ev:
            ref_terms = " × ".join(
                f"(1-{ev.get('ec_score',0):.3f}×{ev.get('refute_confidence',0):.3f})"
                for ev in ref_ev
            )
            ref_line = f"ref = 1 − {ref_terms} = {ref:.4f}"
        else:
            ref_line = "ref = 0.0000  (no refuters)"
        lines.append(
            f'  agg [label="Soft Aggregation  1-∏(1-EC_i×p_stance_i)\\n'
            f'{sup_line}\\n'
            f'{ref_line}",'
            f' shape=diamond, fillcolor="#eff6ff", fontcolor="#1d4ed8", penwidth=2, fontsize=10]'
        )
        lines.append('')

    # L6: DECISION — show all values + highlight active branch
    if has_ec:
        thr = result.get("ec_threshold", 0.35)
        if sup > thr and ref > thr:
            dec_fill, dec_fc = "#fef3c7", "#92400e"
            dec_lbl = (
                f"DECISION  sup={sup:.3f}  ref={ref:.3f}  θ={thr}\\n"
                f"► sup > θ AND ref > θ  → CONFLICTING\\n"
                f"  sup > θ only         → SUPPORTED\\n"
                f"  ref > θ only         → REFUTED\\n"
                f"  neither > θ          → VerdictHead"
            )
        elif sup > thr:
            dec_fill, dec_fc = "#d1fae5", "#14532d"
            dec_lbl = (
                f"DECISION  sup={sup:.3f}  ref={ref:.3f}  θ={thr}\\n"
                f"  sup > θ AND ref > θ  → CONFLICTING\\n"
                f"► sup > θ only         → SUPPORTED\\n"
                f"  ref > θ only         → REFUTED\\n"
                f"  neither > θ          → VerdictHead"
            )
        elif ref > thr:
            dec_fill, dec_fc = "#fee2e2", "#7f1d1d"
            dec_lbl = (
                f"DECISION  sup={sup:.3f}  ref={ref:.3f}  θ={thr}\\n"
                f"  sup > θ AND ref > θ  → CONFLICTING\\n"
                f"  sup > θ only         → SUPPORTED\\n"
                f"► ref > θ only         → REFUTED\\n"
                f"  neither > θ          → VerdictHead"
            )
        else:
            dec_fill, dec_fc = "#f3f4f6", "#374151"
            dec_lbl = (
                f"DECISION  sup={sup:.3f}  ref={ref:.3f}  θ={thr}\\n"
                f"  sup > θ AND ref > θ  → CONFLICTING\\n"
                f"  sup > θ only         → SUPPORTED\\n"
                f"  ref > θ only         → REFUTED\\n"
                f"► neither > θ          → VerdictHead"
            )
        lines.append(
            f'  decision [label="{dec_lbl}", shape=rect,'
            f' fillcolor="{dec_fill}", fontcolor="{dec_fc}", penwidth=3]'
        )
        lines.append('')

    # L7: VERDICT
    vp = result.get("verdict_probs") or [0.0, 0.0, 0.0]
    vp_labels = ["supported", "refuted", "not_enough_evidence"]
    vp_str = "  ".join(f"{lb[:3]}={p:.3f}" for lb, p in zip(vp_labels, vp))
    verdict_upper = verdict.replace("_", " ").upper()
    lines.append(
        f'  verdict [label="VerdictHead\\n{vp_str}\\n→ {verdict_upper}",'
        f' shape=rect, fillcolor="{v_fill.get(verdict, "#1d4ed8")}",'
        f' fontcolor="white", fontsize=11, penwidth=2]'
    )
    lines.append('')

    # Edges
    for i in range(n):
        lines.append(f'  ev_in_{i} -> ev_enc_{i}')
        lines.append(f'  ev_enc_{i} -> ev_h1_{i}')
        lines.append(f'  ev_enc_{i} -> ev_h2_{i}')

        if has_ec:
            ew_v   = bd[i].get("evidence_weight", 0.0)
            st_v   = bd[i].get("source_trust", 0.0)
            stance = bd[i].get("stance", "neutral")
            # EC formula receives: EW from Input (evidence type weight), IS from H2, ST from registry
            lines += [
                f'  ev_in_{i} -> ev_ec_{i} [label="EW={ew_v:.3f}(type)", style=dashed, color="#9ca3af"]',
                f'  ev_h2_{i} -> ev_ec_{i} [label="IS"]',
                f'  sr -> ev_ec_{i} [label="ST={st_v:.3f}", style=dashed, color="#6b7280"]',
            ]
            # H1 stance probs weight EC in the aggregation — connect H1 directly to agg
            lines.append(f'  ev_h1_{i} -> agg [label="p_stance", color="#b45309", style=dashed]')
            ec_i = bd[i].get("ec_score", 0.0)
            lines.append(f'  ev_ec_{i} -> agg [label="EC_i={ec_i:.3f}"]')
        else:
            # baseline: H1/H2 are training-only supervision; verdict comes from claim_emb
            lines += [
                f'  ev_h1_{i} [style="filled,dashed"]',
                f'  ev_h2_{i} [style="filled,dashed"]',
            ]
    if has_ec:
        lines += ['  agg -> decision', '  decision -> verdict']
    else:
        # baseline: show that verdict comes from claim_emb, not evidence heads
        lines.append(
            '  claim_emb [label="claim_emb\\n(from GNN encoder)", shape=rect,'
            ' fillcolor="#bfdbfe", fontcolor="#1e3a8a"]'
        )
        lines += ['  claim_emb -> verdict [label="VerdictMLP"]']
    lines.append('')
    lines.append('}')
    return "\n".join(lines)


# ── Debug view ─────────────────────────────────────────────────────────────────

def render_debug_view(result: dict, claim_text: str | None = None) -> None:
    """Render raw evidence breakdown with full per-evidence property tables."""
    from app.config import enum_label

    bd     = result.get("evidence_breakdown", [])
    has_ec = result.get("has_ec", False)
    claim  = claim_text or result.get("claim_text", "")

    st.markdown("**Claim**")
    if claim:
        st.markdown(f"> {claim}")

    claim_triples = result.get("claim_triples") or []
    if claim_triples:
        st.markdown("*Claim triples:*")
        st.code(str(claim_triples), language=None)

    st.markdown("**Evidence Items**")
    for i, ev in enumerate(bd):
        stance = ev.get("stance", "not_enough_evidence")
        text_short = (ev.get("text_short") or ev.get("text", ""))[:60]
        with st.expander(f"Evidence {i + 1} · {stance} · {text_short}", expanded=False):
            st.markdown(ev.get("text", ""))
            triples = ev.get("triples") or []
            if triples:
                st.markdown("*Triples:*")
                st.code(str(triples), language=None)
            st.divider()
            ec_val = ev.get("ec_score", 0.0)
            prop_rows: list[tuple[str, str]] = [
                ("Modality",        f"`{enum_label(ev.get('modality', ''))}`"),
                ("Source type",     f"`{enum_label(ev.get('source_type', ''))}`"),
                ("Source trust",    f"`{ev.get('source_trust', 0):.3f}`"),
                ("Evidence weight", f"`{ev.get('evidence_weight', 0):.3f}`"),
                ("IS score",        f"`{ev.get('is_score', 0):.3f}`"),
            ]
            if has_ec:
                prop_rows.append(("EC score", f"`{ec_val:.3f}`"))
            prop_rows += [
                ("Stance",              f"`{stance}`"),
                ("Support confidence",  f"`{ev.get('support_confidence', 0):.3f}`"),
                ("Refute confidence",   f"`{ev.get('refute_confidence',  0):.3f}`"),
            ]
            nli = ev.get("nli_probs")
            if nli:
                prop_rows += [
                    ("NLI entailment",    f"`{nli['entailment']:.3f}`"),
                    ("NLI contradiction", f"`{nli['contradiction']:.3f}`"),
                    ("NLI neutral",       f"`{nli['neutral']:.3f}`"),
                ]
            st.markdown(
                "| Property | Value |\n|---|---|\n"
                + "".join(f"| {p} | {v} |\n" for p, v in prop_rows)
            )

    st.markdown("**Verdict Summary**")
    verdict = result.get("verdict", "")
    vd = {}
    try:
        from app.config import get_config
        vd = get_config().verdict_display.get(verdict, {})
    except Exception:
        pass
    emoji = vd.get("emoji", "❓")
    st.markdown(f"## {emoji} {enum_label(verdict)}")
    if has_ec:
        sup = result.get("support_score", 0.0)
        ref = result.get("refute_score",  0.0)
        st.progress(min(sup, 1.0), text=f"🟢 support  {sup:.3f}")
        st.progress(min(ref, 1.0), text=f"🔴 refute   {ref:.3f}")


def build_pyvis_html(
    hetero_data,
    claim_text: str = "",
    ev_texts: list[str] | None = None,
    height: str = "540px",
) -> str:
    """Build an interactive pyvis HTML string from a HeteroData graph.

    Uses a temp file to avoid depending on generate_html() / get_network_html()
    which differ across pyvis versions.
    """
    import os
    import tempfile

    from pyvis.network import Network

    from src.model.data.types import EdgeType, NodeType

    net = Network(
        height=height, width="100%", directed=True,
        bgcolor="#f8fafc", font_color="#1e293b",
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {"gravitationalConstant": -8000, "springLength": 140},
        "stabilization": {"iterations": 150}
      },
      "edges": {"smooth": {"type": "dynamic"}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)

    claim_store = hetero_data[NodeType.CLAIM]
    ev_store    = hetero_data[NodeType.EVIDENCE]

    claim_snip = (claim_text[:60] + "…") if len(claim_text) > 60 else claim_text
    net.add_node(
        "C0", label=f"CLAIM\n{claim_snip[:30]}",
        title=f"<b>CLAIM</b><br>{claim_snip}<br>x: [{claim_store.x.shape[1]} dims]",
        color={"background": "#1d4ed8", "border": "#1e40af",
               "highlight": {"background": "#2563eb"}},
        font={"color": "white", "size": 13, "bold": True},
        size=40, shape="box",
    )

    _STANCE_COLOR = {
        "supports":            {"background": "#16a34a", "border": "#15803d"},
        "refutes":             {"background": "#dc2626", "border": "#b91c1c"},
        "not_enough_evidence": {"background": "#6b7280", "border": "#4b5563"},
    }
    _INT_TO_STANCE = {0: "supports", 1: "refutes", 2: "not_enough_evidence"}

    n_ev = ev_store.x.shape[0]
    for i in range(n_ev):
        stance_int = int(ev_store.stance_y[i].item()) if hasattr(ev_store, "stance_y") else 2
        stance_lbl = _INT_TO_STANCE.get(stance_int, "not_enough_evidence")
        color      = _STANCE_COLOR.get(stance_lbl, _STANCE_COLOR["not_enough_evidence"])
        st_val     = float(ev_store.st[i].item()) if hasattr(ev_store, "st") else 0.0
        ew_val     = float(ev_store.ew[i].item()) if hasattr(ev_store, "ew") else 0.0
        ev_snip    = ((ev_texts[i][:60] + "…") if ev_texts and i < len(ev_texts) else "")
        net.add_node(
            f"E{i}", label=f"EV{i}\n{stance_lbl[:3]}",
            title=f"<b>EV{i}</b><br>stance: {stance_lbl}<br>ST={st_val:.2f} EW={ew_val:.2f}<br>{ev_snip}",
            color=color, font={"color": "white", "size": 11},
            size=25, shape="ellipse",
        )

    # claim → evidence
    try:
        ei = hetero_data[NodeType.CLAIM, EdgeType.HAS_EVIDENCE, NodeType.EVIDENCE].edge_index
        for j in range(ei.shape[1]):
            net.add_edge("C0", f"E{int(ei[1, j].item())}", color="#3b82f6", width=2)
    except Exception:
        pass

    # evidence → claim (connected_to)
    try:
        ei = hetero_data[NodeType.EVIDENCE, EdgeType.CONNECTED_TO, NodeType.CLAIM].edge_index
        for j in range(ei.shape[1]):
            net.add_edge(f"E{int(ei[0, j].item())}", "C0",
                         color="#f97316", dashes=True, width=1)
    except Exception:
        pass

    # evidence ↔ evidence (co-evidence)
    try:
        ei = hetero_data[NodeType.EVIDENCE, EdgeType.CO_EVIDENCE, NodeType.EVIDENCE].edge_index
        for j in range(ei.shape[1]):
            src, dst = int(ei[0, j].item()), int(ei[1, j].item())
            if src < dst:
                net.add_edge(f"E{src}", f"E{dst}", color="#9ca3af", dashes=True, width=1)
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        tmp = f.name
    try:
        net.save_graph(tmp)
        with open(tmp, encoding="utf-8") as f:
            html = f.read()
    finally:
        os.unlink(tmp)

    # Inject fit-to-extent after stabilization so graph is centered, not corner-aligned
    fit_js = """
<script>
(function waitForNetwork() {
  if (typeof network !== 'undefined') {
    network.once('stabilizationIterationsDone', function () {
      network.fit({ animation: { duration: 600, easingFunction: 'easeOutQuad' } });
    });
    // fallback: also fit after a short delay in case event already fired
    setTimeout(function () { network.fit(); }, 800);
  } else {
    setTimeout(waitForNetwork, 50);
  }
})();
</script>
"""
    html = html.replace("</body>", fit_js + "</body>")
    return html
