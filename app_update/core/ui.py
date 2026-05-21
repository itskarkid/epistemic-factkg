"""Thin Streamlit rendering layer for model outputs.

All data is produced by src/ methods (decision_path_info, evidence_table,
build_pyvis_html is the only exception — pyvis is a visualization library).
These functions consume dicts/lists and render Streamlit widgets.
"""

from __future__ import annotations

import streamlit as st


_PATH_STYLE = {
    "symbolic_override": ("info",    "Symbolic Override"),
    "conflicting":       ("warning", "Conflicting Evidence"),
    "weak_ec":           ("info",    "Weak EC — Neural Verdict"),
    "baseline":          ("info",    "Baseline (No EC)"),
}


def render_decision_path(path_info: dict) -> None:
    """Render an EC decision path callout from decision_path_info() output."""
    path    = path_info.get("path", "baseline")
    reason  = path_info.get("override_reason", "")
    has_ec  = path_info.get("has_ec", False)
    kind, label = _PATH_STYLE.get(path, ("info", path))

    if not has_ec:
        st.info(f"**{label}** — {reason}")
        return

    sup = path_info.get("support_score", 0.0)
    ref = path_info.get("refute_score",  0.0)
    thr = path_info.get("ec_threshold",  0.35)

    msg = f"**{label}** — {reason}  \n`sup={sup:.3f}` · `ref={ref:.3f}` · `θ={thr:.2f}`"
    getattr(st, kind)(msg)


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
    from app_update.config import enum_label

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
                "① NLI cross-encoder",
                f"entail `{e:.1%}` &nbsp; contra `{c:.1%}` &nbsp; neutral `{n:.1%}`",
            ))

        stance     = row["stance"]
        s_emoji    = _STANCE_EMOJI.get(stance, "⚪")
        conf       = row["stance_confidence"]
        md_rows.append(("② Stance (H1)", f"{s_emoji} **{enum_label(stance)}** &nbsp; conf `{conf:.0%}`"))
        md_rows.append(("③ IS (H2)",     f"`{row['is_score']:.3f}`"))

        if has_ec:
            md_rows.append((
                "④ EC formula",
                f"`{row['ec_score']:.3f}` &nbsp; ST `{row['source_trust']:.2f}` · EW `{row['evidence_weight']:.2f}` · IS `{row['is_score']:.3f}`",
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
    return html
