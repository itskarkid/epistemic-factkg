"""ArcBlock — self-describing computation graph for model visualization.

Models and sub-components implement arc_block_definition() to produce a
renderable Graphviz DOT string. App layer calls .to_dot() on the returned
CompositeArcBlock and passes the result to st.graphviz_chart() — zero model
knowledge required in the app layer.

Free functions decision_path_info(), evidence_table(), and result_dot() convert
a build_prediction_payload() result dict into renderable data; models delegate
to these rather than duplicating the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArcBlock:
    """One computation stage in a model's pipeline."""

    name: str
    detail: str
    node_id: str = ""
    live_annotations: dict[str, str] = field(default_factory=dict)
    color: str = "#e2e8f0"

    def to_dot_node(self) -> str:
        label = f"{self.name}\\n{self.detail}"
        if self.live_annotations:
            ann = "\\n".join(f"{k}: {v}" for k, v in self.live_annotations.items())
            label += f"\\n──────────\\n{ann}"
        nid = self._nid()
        return (
            f'  {nid} [label="{label}", style="filled,rounded", '
            f'fillcolor="{self.color}", shape=box, fontsize=10];'
        )

    def _nid(self) -> str:
        return self.node_id or (
            self.name.replace(" ", "_")
                     .replace("(", "")
                     .replace(")", "")
                     .replace("-", "_")
        )


@dataclass
class CompositeArcBlock:
    """Ordered list of ArcBlocks → renderable Graphviz DOT string."""

    blocks: list[ArcBlock]
    title: str = "GNN_Flow"

    def to_dot(self) -> str:
        """Returns DOT string — pass directly to st.graphviz_chart()."""
        lines = [
            f"digraph {self.title} {{",
            "  rankdir=TB;",
            '  node [fontname="Helvetica"];',
            '  graph [bgcolor="transparent"];',
        ]
        for b in self.blocks:
            lines.append(b.to_dot_node())
        for i in range(len(self.blocks) - 1):
            src = self.blocks[i]._nid()
            dst = self.blocks[i + 1]._nid()
            lines.append(f"  {src} -> {dst};")
        lines.append("}")
        return "\n".join(lines)


# ── Shared result-rendering helpers ──────────────────────────────────────────

_STANCE_COLOR = {
    "supports":            "#d1fae5",
    "refutes":             "#fee2e2",
    "not_enough_evidence": "#f3f4f6",
}

_VERDICT_COLOR = {
    "supported":            "#d1fae5",
    "refuted":              "#fee2e2",
    "not_enough_evidence":  "#f3f4f6",
    "conflicting_evidence": "#fef3c7",
}


def _make_reason(ec_decision: str, final_layer: str, sup: float, ref: float, thr: float) -> str:
    if ec_decision == "supported":
        return f"Support signal decisive (sup={sup:.3f} > θ={thr:.2f}, ref={ref:.3f})."
    if ec_decision == "refuted":
        return f"Refute signal decisive (ref={ref:.3f} > θ={thr:.2f}, sup={sup:.3f})."
    if ec_decision == "conflicted":
        return f"Both sides decisive (sup={sup:.3f}, ref={ref:.3f} > θ={thr:.2f}) — VerdictHead resolves conflict."
    return f"EC below threshold on both sides (sup={sup:.3f}, ref={ref:.3f} ≤ θ={thr:.2f}) — VerdictHead decides."


def decision_path_info(result: dict) -> dict:
    """Extract EC vs VerdictHead decision layer info from a build_prediction_payload result."""
    has_ec = result.get("has_ec", False)
    if not has_ec:
        return {
            "path": "baseline",
            "has_ec": False,
            "support_score": 0.0,
            "refute_score": 0.0,
            "ec_threshold": None,
            "ec_decision": None,
            "final_layer": "verdicthead",
            "vh_pred": None,
            "override_reason": "Baseline model — no EC formula.",
        }

    sup = result.get("support_score", 0.0)
    ref = result.get("refute_score",  0.0)
    thr = result.get("ec_threshold",  0.35)

    # Prefer explicit fields passed through from predict(); fall back to score derivation
    # for older model outputs that predate this change.
    ec_decision = result.get("ec_decision")
    final_layer = result.get("final_layer")
    vh_pred     = result.get("vh_pred")

    if ec_decision is None:
        if sup > thr and ref > thr:
            ec_decision, final_layer = "conflicted", "verdicthead"
        elif sup > thr:
            ec_decision, final_layer = "supported",  "ec_symbolic"
        elif ref > thr:
            ec_decision, final_layer = "refuted",    "ec_symbolic"
        else:
            ec_decision, final_layer = "deferred",   "verdicthead"

    if final_layer == "ec_symbolic":
        path = "symbolic_override"
    elif ec_decision == "conflicted":
        path = "conflicting"
    else:
        path = "weak_ec"

    return {
        "path":           path,
        "has_ec":         True,
        "support_score":  sup,
        "refute_score":   ref,
        "ec_threshold":   thr,
        "ec_decision":    ec_decision,
        "final_layer":    final_layer,
        "vh_pred":        vh_pred,
        "override_reason": _make_reason(ec_decision, final_layer, sup, ref, thr),
    }


def evidence_table(result: dict) -> list[dict]:
    """Extract per-evidence table rows from a build_prediction_payload result."""
    rows = []
    for i, ev in enumerate(result.get("evidence_breakdown", [])):
        rows.append({
            "idx":               i,
            "text_short":        ev.get("text_short") or (ev.get("text", ""))[:150],
            "stance":            ev.get("stance", "not_enough_evidence"),
            "support_confidence": ev.get("support_confidence", 0.0),
            "refute_confidence":  ev.get("refute_confidence", 0.0),
            "is_score":          ev.get("is_score", 0.0),
            "ec_score":          ev.get("ec_score", 0.0),
            "evidence_weight":   ev.get("evidence_weight", 0.0),
            "source_trust":      ev.get("source_trust", 0.0),
            "modality":          ev.get("modality", ""),
            "source_type":       ev.get("source_type", ""),
            "source_id":         ev.get("source_id", ""),
            "nli_probs":         ev.get("nli_probs"),
        })
    return rows


def result_dot(result: dict) -> str:
    """Build a Graphviz DOT graph of an inference result.

    Reads claim_text, verdict, EC scores, and evidence_breakdown from result.
    claim_text must be added to the result dict by the caller (e.g. predictor).
    """
    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    claim_text = _esc((result.get("claim_text") or "")[:80])
    verdict    = result.get("verdict", "unknown")
    sup        = result.get("support_score", 0.0)
    ref        = result.get("refute_score", 0.0)
    thr        = result.get("ec_threshold")
    has_ec     = result.get("has_ec", False)
    breakdown  = result.get("evidence_breakdown", [])

    lines = [
        "digraph Claim_Graph {",
        "  rankdir=TB;",
        '  node [fontname="Helvetica", fontsize=10];',
        '  graph [bgcolor="transparent"];',
    ]

    lines.append(
        f'  claim [label="CLAIM\\n{claim_text}", shape=box, '
        f'style="filled,rounded", fillcolor="#dbeafe"];'
    )

    for i, ev in enumerate(breakdown):
        stance = ev.get("stance", "not_enough_evidence")
        ec     = ev.get("ec_score", 0.0)
        text   = _esc((ev.get("text_short") or ev.get("text", ""))[:60])
        color  = _STANCE_COLOR.get(stance, "#f3f4f6")
        ann    = f"{stance} | EC {ec:.3f}"
        lines.append(
            f'  ev{i} [label="Ev {i + 1}\\n{text}\\n{ann}", shape=box, '
            f'style="filled,rounded", fillcolor="{color}"];'
        )

    if has_ec:
        thr_str = f"θ={thr:.2f}" if thr is not None else ""
        lines.append(
            f'  ec_agg [label="EC Aggregate\\nsup={sup:.3f} | ref={ref:.3f}\\n{thr_str}", '
            f'shape=diamond, style="filled", fillcolor="#fef3c7"];'
        )

    v_color = _VERDICT_COLOR.get(verdict, "#f3f4f6")
    lines.append(
        f'  verdict_node [label="Verdict\\n{verdict}", shape=ellipse, '
        f'style="filled", fillcolor="{v_color}"];'
    )

    for i in range(len(breakdown)):
        lines.append(f"  claim -> ev{i};")

    if has_ec:
        for i, ev in enumerate(breakdown):
            if ev.get("stance") in ("supports", "refutes"):
                lines.append(f"  ev{i} -> ec_agg;")
        lines.append("  ec_agg -> verdict_node;")
    else:
        for i in range(len(breakdown)):
            lines.append(f"  ev{i} -> verdict_node;")

    lines.append("}")
    return "\n".join(lines)
