"""Reference tab — formulas, registry, assumptions, ADR & docs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig

_ADR_DIR  = Path("docs/adr")
_DOCS_DIR = Path("docs")


# ── Entry point ────────────────────────────────────────────────────────────────

def render(cfg: "AppConfig") -> None:
    tabs = st.tabs([
        "EC Formula",
        "Decision Logic",
        "Model Architecture",
        "Evidence Weights",
        "Source Registry",
        "Assumptions",
        "ADR & Docs",
    ])
    with tabs[0]:
        _render_ec_formula()
    with tabs[1]:
        _render_decision_logic(cfg)
    with tabs[2]:
        _render_model_architecture()
    with tabs[3]:
        _render_evidence_weights(cfg)
    with tabs[4]:
        _render_registry(cfg)
    with tabs[5]:
        _render_assumptions()
    with tabs[6]:
        _render_adr_docs()


# ── EC Formula ─────────────────────────────────────────────────────────────────

def _render_ec_formula() -> None:
    st.markdown("### Per-Evidence Confidence Formula")
    st.code("EC_i = 1 − (1 − ST_i)^(EW_i × IS_i)", language=None)
    st.markdown(
        "| Symbol | Meaning | Range |\n|--------|---------|-------|\n"
        "| ST | Source Trust — resolved from registry via `source_id` | [0, 1] |\n"
        "| EW | Evidence Weight — stance probability from H1 StanceHead (all models) | [0, 1] |\n"
        "| IS | Inference Strength — regression head output | [0, 1] |\n"
        "| EC | Epistemic Confidence — combined signal for this evidence item | [0, 1] |"
    )

    st.divider()
    st.markdown("### EC Aggregation (product-of-complements)")
    st.markdown(
        "For each direction d ∈ {support, refute}:\n\n"
        r"$$EC_d = 1 - \prod_{i\,:\,\text{stance}_i = d} (1 - EC_i)$$"
    )
    st.markdown(
        "Evidence with `not_enough_evidence` or `conflicting_evidence` stance is excluded  \n"
        "from both aggregations.  \n\n"
        "**Aggregation rationale** (ADR-010): probability that at least one contributing evidence  \n"
        "item is correct under source independence — diminishing returns when multiple items agree."
    )

    st.divider()
    st.markdown("### Inference Strength (IS) — rubric")
    st.markdown(
        "| IS | Description |\n|----|-----------|\n"
        "| 1.0 | Direct ground truth (simulator state, primary measurement) |\n"
        "| 0.8 | One-step extractive (direct quote, Boolean answer, official record) |\n"
        "| 0.6 | Abstractive / multi-source synthesis |\n"
        "| 0.4 | Partial or circumstantial (covers only part of claim) |\n"
        "| 0.2 | Speculative or highly hedged |\n"
        "| 0.0 | Unanswerable |"
    )
    st.caption(
        "IS captures logical binding strength, not stance direction. "
        "A high-trust source with an abstractive answer still gets IS = 0.6; ST captures source quality separately.  \n"
        "IS is detached from the verdict gradient to prevent shortcut learning (ADR-022)."
    )


# ── Decision Logic ─────────────────────────────────────────────────────────────

def _render_decision_logic(cfg: "AppConfig") -> None:
    default_thr = getattr(cfg, "default_ec_threshold", 0.35)
    st.markdown(
        f"### Decision Logic  `ec_threshold θ = {default_thr}`  "
        "(per-checkpoint hyperparameter, tuned via Optuna)"
    )
    st.markdown(
        "| Condition | Outcome |\n|-----------|---------|\n"
        "| sup > θ **and** ref > θ | Conflicting → fall through to VerdictHead |\n"
        "| sup > θ only | Symbolic override → **SUPPORTED** |\n"
        "| ref > θ only | Symbolic override → **REFUTED** |\n"
        "| neither > θ | EC weak → VerdictHead decides |\n"
        "| baseline model | VerdictHead always (no EC computed) |"
    )
    st.caption(
        "The `_EC_NEI_MAX` branch (forced NEI when both scores < 0.20) was removed in ADR-027. "
        "Full VerdictHead delegation is now used for all non-override cases."
    )

    st.divider()
    st.markdown("### Verdict Classes")
    st.markdown(
        "| Label | Symbol | Meaning |\n|-------|--------|----------|\n"
        "| supported | ✅ | Evidence confirms the claim |\n"
        "| refuted | ❌ | Evidence contradicts the claim |\n"
        "| not_enough_evidence | ~ | Insufficient or conflicting evidence |"
    )
    st.caption("`conflicting_evidence` exists in schema but is excluded from GNN training (ADR-007).")


# ── Model Architecture ─────────────────────────────────────────────────────────

_ARCH_DOTS = {
    "baseline": """\
digraph baseline {
  graph [bgcolor="#f8fafc", rankdir=LR, pad="0.4", nodesep=0.6, ranksep=0.5,
         label="baseline — GNN encoder + direct VerdictMLP on claim_emb (no EC)", labelloc=t,
         fontname="Arial", fontsize=10]
  node  [fontname="Arial", fontsize=9, style=filled, margin="0.10,0.05"]
  edge  [fontname="Arial", fontsize=8, color="#6b7280"]

  I   [label="Input\\nClaim 390d · Evidence 405d",    shape=rect, fillcolor="#dbeafe", fontcolor="#1e40af"]
  Enc [label="EpistemicEncoder\\nHeteroConv × 2 (GAT)\\nclaim_emb + ev_emb", shape=rect, fillcolor="#bfdbfe", fontcolor="#1e3a8a"]
  H1  [label="H1: StanceHead\\nev_ctx [512d] → stance [N_ev, 3]\\nTRAINING supervision only", shape=rect, fillcolor="#fef9c3", fontcolor="#92400e", style="filled,dashed"]
  H2  [label="H2: ISHead\\nev_ctx [512d] → IS [N_ev, 1]\\nTRAINING supervision only", shape=rect, fillcolor="#fef9c3", fontcolor="#92400e", style="filled,dashed"]
  VM  [label="VerdictMLP\\nclaim_emb 256d → 128 → ReLU → 3\\nINFERENCE path",  shape=rect, fillcolor="#fce7f3", fontcolor="#9d174d", penwidth=2]
  Out [label="Verdict",                               shape=ellipse, fillcolor="#dcfce7", fontcolor="#166534"]

  {rank=same; H1; H2}

  I   -> Enc
  Enc -> H1  [label="ev_ctx=cat(ev,cl)\\n(train)", style=dashed, color="#b45309"]
  Enc -> H2  [label="ev_ctx=cat(ev,cl)\\n(train)", style=dashed, color="#b45309"]
  Enc -> VM  [label="claim_emb", color="#1d4ed8", penwidth=2]
  VM  -> Out
}""",

    "v1-hgnn": """\
digraph v1_hgnn {
  graph [bgcolor="#f8fafc", rankdir=LR, pad="0.4", nodesep=0.6, ranksep=0.5,
         label="v1-hgnn — EC formula + symbolic decision at threshold θ", labelloc=t,
         fontname="Arial", fontsize=10]
  node  [fontname="Arial", fontsize=9, style=filled, margin="0.10,0.05"]
  edge  [fontname="Arial", fontsize=8, color="#6b7280"]

  I   [label="Input\\nClaim 390d · Evidence 405d", shape=rect, fillcolor="#dbeafe", fontcolor="#1e40af"]
  SR  [label="Source Registry\\nsource_id → ST lookup\\n(graph-build time)",   shape=cylinder, fillcolor="#f1f5f9", fontcolor="#374151"]
  Enc [label="EpistemicEncoder\\nHeteroConv × 2 (GAT)",     shape=rect, fillcolor="#bfdbfe", fontcolor="#1e3a8a"]
  H1  [label="H1: StanceHead\\nev_ctx [512d] → stance probs [N_ev, 3]",   shape=rect, fillcolor="#fef3c7", fontcolor="#92400e"]
  H2  [label="H2: ISHead (detach)\\nev_ctx [512d] → IS scalar [N_ev, 1]", shape=rect, fillcolor="#fef3c7", fontcolor="#92400e"]
  EC  [label="EC Formula\\n1−(1−ST)^(EW×IS)\\nper evidence", shape=rect, fillcolor="#d1fae5", fontcolor="#065f46"]
  Agg [label="Aggregation\\n1−∏(1−EC_i)\\n→ sup, ref",     shape=diamond, fillcolor="#eff6ff", fontcolor="#1d4ed8"]
  Dec [label="Decision (θ=0.35)\\nsup>θ → SUPPORTED\\nref>θ → REFUTED\\nelse → VerdictHead", shape=rect, fillcolor="#fef3c7", fontcolor="#92400e", penwidth=2]
  VH  [label="VerdictHead\\nEC scores [3d]\\n→ Linear → verdict", shape=rect, fillcolor="#fce7f3", fontcolor="#9d174d"]
  Out [label="Verdict",                                     shape=ellipse, fillcolor="#dcfce7", fontcolor="#166534"]

  {rank=same; H1; H2}

  I   -> Enc
  SR  -> EC  [label="ST", color="#374151", style=dashed]
  Enc -> H1  [label="ev_ctx=cat(ev,cl)"]
  Enc -> H2  [label="ev_ctx=cat(ev,cl)"]
  H1  -> EC  [label="EW (stance prob)"]
  H2  -> EC  [label="IS.detach()"]
  EC  -> Agg
  Agg -> Dec [label="sup, ref"]
  Dec -> Out [label="override", style=dashed, color="#16a34a"]
  Dec -> VH  [label="fallback", style=dashed, color="#6b7280"]
  VH  -> Out
}""",

    "v2-hgnn": """\
digraph v2_hgnn {
  graph [bgcolor="#f8fafc", rankdir=LR, pad="0.4", nodesep=0.6, ranksep=0.5,
         label="v2-hgnn — HybridVerdictHead: EC scores + claim_emb (ADR-023)", labelloc=t,
         fontname="Arial", fontsize=10]
  node  [fontname="Arial", fontsize=9, style=filled, margin="0.10,0.05"]
  edge  [fontname="Arial", fontsize=8, color="#6b7280"]

  I   [label="Input\\nClaim 390d · Evidence 405d", shape=rect, fillcolor="#dbeafe", fontcolor="#1e40af"]
  SR  [label="Source Registry\\nsource_id → ST lookup\\n(graph-build time)",   shape=cylinder, fillcolor="#f1f5f9", fontcolor="#374151"]
  Enc [label="EpistemicEncoder\\nHeteroConv × 2 (GAT)",     shape=rect, fillcolor="#bfdbfe", fontcolor="#1e3a8a"]
  H1  [label="H1: StanceHead\\nev_ctx [512d] → stance probs [N_ev, 3]",   shape=rect, fillcolor="#fef3c7", fontcolor="#92400e"]
  H2  [label="H2: ISHead (detach)\\nev_ctx [512d] → IS scalar [N_ev, 1]", shape=rect, fillcolor="#fef3c7", fontcolor="#92400e"]
  EC  [label="EC Formula\\n1−(1−ST)^(EW×IS)",              shape=rect, fillcolor="#d1fae5", fontcolor="#065f46"]
  Agg [label="Aggregation\\n→ sup, ref",                   shape=diamond, fillcolor="#eff6ff", fontcolor="#1d4ed8"]
  Dec [label="Decision (θ)\\nsymbolic override\\nor fallback", shape=rect, fillcolor="#fef3c7", fontcolor="#92400e", penwidth=2]
  HVH [label="HybridVerdictHead\\nclaim_emb [256d proj→16d]\\n+ EC scores [3d]\\n→ concat → MLP → 3", shape=rect, fillcolor="#fce7f3", fontcolor="#9d174d", penwidth=2]
  Out [label="Verdict",                                     shape=ellipse, fillcolor="#dcfce7", fontcolor="#166534"]

  {rank=same; H1; H2}

  I   -> Enc
  SR  -> EC  [label="ST", color="#374151", style=dashed]
  Enc -> H1  [label="ev_ctx=cat(ev,cl)"]
  Enc -> H2  [label="ev_ctx=cat(ev,cl)"]
  Enc -> HVH [label="claim_emb", color="#1d4ed8", penwidth=2]
  H1  -> EC  [label="EW (stance prob)"]
  H2  -> EC  [label="IS.detach()"]
  EC  -> Agg
  Agg -> Dec
  Dec -> Out  [label="override", style=dashed, color="#16a34a"]
  Dec -> HVH  [label="EC scores [3d]", color="#15803d"]
  HVH -> Out
}""",

    "v3-nli": """\
digraph v3_nli {
  graph [bgcolor="#f8fafc", rankdir=LR, pad="0.4", nodesep=0.6, ranksep=0.5,
         label="v3-nli — NLI probs as 408d features; claim-aware H1 on GNN output (ADR-029)", labelloc=t,
         fontname="Arial", fontsize=10]
  node  [fontname="Arial", fontsize=9, style=filled, margin="0.10,0.05"]
  edge  [fontname="Arial", fontsize=8, color="#6b7280"]

  I   [label="Input\\nClaim 390d\\nEvidence 408d (405d + 3d NLI probs\\noffline DeBERTa-v3-small)", shape=rect, fillcolor="#dbeafe", fontcolor="#1e40af"]
  SR  [label="Source Registry\\nsource_id → ST lookup\\n(graph-build time)",   shape=cylinder, fillcolor="#f1f5f9", fontcolor="#374151"]
  Enc [label="EpistemicEncoder\\nHeteroConv × 2 (GAT)\\n408d evidence features", shape=rect, fillcolor="#bfdbfe", fontcolor="#1e3a8a"]
  H1  [label="H1: StanceHead\\nev_ctx [512d] → stance probs [N_ev, 3]\\n(NLI features visible via GNN + claim ctx)", shape=rect, fillcolor="#fef3c7", fontcolor="#92400e"]
  H2  [label="H2: ISHead (detach)\\nev_ctx [512d] → IS scalar [N_ev, 1]", shape=rect, fillcolor="#fef3c7", fontcolor="#92400e"]
  EC  [label="EC Formula\\n1−(1−ST)^(EW×IS)",              shape=rect, fillcolor="#d1fae5", fontcolor="#065f46"]
  Agg [label="Aggregation\\n→ sup, ref",                   shape=diamond, fillcolor="#eff6ff", fontcolor="#1d4ed8"]
  Dec [label="Decision (θ)\\nsymbolic or fallback",        shape=rect, fillcolor="#fef3c7", fontcolor="#92400e", penwidth=2]
  HVH [label="HybridVerdictHead\\nclaim_emb [256d proj→16d]\\n+ EC scores [3d]\\n→ concat → MLP → verdict", shape=rect, fillcolor="#fce7f3", fontcolor="#9d174d", penwidth=2]
  Out [label="Verdict",                                     shape=ellipse, fillcolor="#dcfce7", fontcolor="#166534"]

  {rank=same; H1; H2}

  I   -> Enc
  SR  -> EC  [label="ST", color="#374151", style=dashed]
  Enc -> H1  [label="ev_ctx=cat(ev,cl)"]
  Enc -> H2  [label="ev_ctx=cat(ev,cl)"]
  Enc -> HVH [label="claim_emb", color="#1d4ed8", penwidth=2]
  H1  -> EC  [label="EW (stance prob)"]
  H2  -> EC  [label="IS.detach()"]
  EC  -> Agg
  Agg -> Dec
  Dec -> Out  [label="override", style=dashed, color="#16a34a"]
  Dec -> HVH  [label="EC scores [3d]", color="#15803d"]
  HVH -> Out
}""",
}

_ARCH_ABLATION = {
    "baseline": (
        "**Ablation baseline** — verdict is predicted directly from the claim node embedding "
        "with no EC formula or symbolic aggregation. "
        "H1 (StanceHead) and H2 (ISHead) are still present and provide multi-task training supervision "
        "for evidence node embeddings via stance CE + IS MSE losses. "
        "Their outputs are **not** used in the verdict inference path — only the encoder gradient signal matters. "
        "Serves as the control: if EC adds no value, baseline should match v1-hgnn."
    ),
    "v1-hgnn": (
        "**First epistemic model** — adds the EC formula and symbolic decision logic on top of baseline. "
        "VerdictHead only receives EC scores [3d] — no claim embedding. "
        "This creates an information bottleneck: all claim semantics must pass through EC aggregation. "
        "IS RMSE ≈ 0.234 before ADR-022 (gradient conflict); ≈ 0.12 after IS detach."
    ),
    "v2-hgnn": (
        "**Hybrid verdict head** (ADR-023) — fixes v1-hgnn's information bottleneck by concatenating "
        "claim_emb (projected to 16d) with EC scores [3d] before the verdict MLP. "
        "The claim embedding provides a direct gradient path to the encoder for verdict supervision. "
        "IS detach (ADR-022) retained — IS regression and verdict optimization remain decoupled."
    ),
    "v3-nli": (
        "**NLI-enriched GNN with graph-aware H1** (ADR-024) — evidence features are 408d "
        "(405d base + 3d NLI probability columns from offline DeBERTa-v3-small). "
        "The NLI probs act as strong semantic priors in the GAT: the encoder can use entailment confidence "
        "when computing attention weights. H1 StanceHead then predicts **graph-aware** stance from the "
        "GNN-enriched evidence embeddings — not directly from NLI output. "
        "This gives the GNN a meaningful role: refining NLI priors with multi-evidence graph context "
        "(corroboration, conflict, source relations)."
    ),
}


def _render_model_architecture() -> None:
    # Comparison table
    st.markdown("### Model Family Overview")
    st.markdown(
        "| Model | EC | NLI | H1 | VerdictHead input | Key ADR |\n"
        "|-------|-----|-----|-----|-------------------|----------|\n"
        "| baseline | ✗ | ✗ | ✓ train-only | claim_emb [256d] | ablation |\n"
        "| v1-hgnn  | ✓ | ✗ | ✓ → EC EW | EC scores [3d] | ADR-013/014 |\n"
        "| v2-hgnn  | ✓ | ✗ | ✓ → EC EW | EC scores [3d] + claim_emb proj [16d] | ADR-023 |\n"
        "| v3-nli   | ✓ | ✓ input feat | ✓ graph-aware | EC scores [3d] + claim_emb proj [16d] | ADR-024 |"
    )

    st.divider()
    st.markdown("### Architecture Diagram")
    model_key = st.selectbox(
        "Model", list(_ARCH_DOTS.keys()), key="ref_arch_model",
        format_func=lambda k: k,
    )

    st.markdown(f"_{_ARCH_ABLATION[model_key]}_")

    try:
        st.graphviz_chart(_ARCH_DOTS[model_key], width='stretch')
    except Exception as e:
        st.warning(f"Diagram unavailable: {e}")
        st.code(_ARCH_DOTS[model_key], language=None)

    st.divider()
    st.markdown("### Shared Components")
    st.markdown(
        "| Component | Description |\n|-----------|-------------|\n"
        "| EpistemicEncoder | 2-layer HeteroConv with GAT attention; "
        "processes CLAIM and EVIDENCE nodes over (EVIDENCE, has_evidence/connected_to/co_evidence, CLAIM) edges |\n"
        "| H1 StanceHead | MLP: ev_ctx=cat([ev_emb, claim_emb]) [512d] → stance logits [3] (supports/refutes/neutral) |\n"
        "| H2 ISHead | MLP: ev_ctx [512d] → IS scalar ∈ [0,1]; gradient detached before EC formula |\n"
        "| SymbolicAggregator | Product-of-complements: EC_d = 1 − ∏(1 − EC_i) per direction d |"
    )


# ── Evidence Weights ────────────────────────────────────────────────────────────

def _render_evidence_weights(cfg: "AppConfig") -> None:
    st.markdown("### Evidence Weight (EW) by Type")
    st.markdown(
        "EW_i = the stance probability from H1 (StanceHead or NLI). In EC formula, EW is the  \n"
        "weight by which source trust is exponentiated — higher EW = stronger contribution."
    )
    st.markdown(
        "| Evidence Type | EW | Description |\n|---|---|---|\n"
        "| perception | 0.95 | AI2THOR sensor-confirmed fact (simulator ground truth) |\n"
        "| testimony | 0.80 | Web text, PDFs, cited sources |\n"
        "| non_apprehension | 0.75 | Sensor-confirmed absence — closed-world only |\n"
        "| comparison_analogy | 0.65 | Numeric/statistical or analogical reasoning |\n"
        "| inference | 0.55 | Multi-step synthesis — compounds error |\n"
        "| postulation_derivation | 0.40 | Hypothetical derivation — least reliable |"
    )
    st.caption(
        "Multi-type combination uses product-of-complements (ADR-001):  \n"
        "`EW = 1 − Π(1 − wᵢ)` — same diminishing-returns formula as EC aggregation."
    )

    st.divider()
    st.markdown("### Source Trust Scale  (from registry)")
    st.markdown(_st_scale_markdown(cfg))


def _st_scale_markdown(cfg: "AppConfig") -> str:
    from app.core.loaders import load_registry
    records = load_registry(cfg.registry_path)
    if not records:
        return (
            "| Source Type | ST range | Entries |\n|-------------|----------|---------|\n"
            "| simulation | 1.00 | 1 |\n"
            "| sensor | 0.90 | 1 |\n"
            "| government | 0.85 – 0.92 | — |\n"
            "| academic | 0.80 – 0.90 | — |\n"
            "| news_media | 0.62 – 0.85 | — |\n"
            "| web_text | 0.55 – 0.60 | — |\n"
            "| social_media | 0.30 – 0.35 | — |\n"
            "| unknown | 0.40 | — |\n"
            "\n*Registry unavailable — showing static defaults.*"
        )
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_type[r.get("source_type") or "unknown"].append(r.get("source_trust", 0.0))
    lines = "| Source Type | ST range | Entries |\n|-------------|----------|---------|\n"
    for stype, sts in sorted(by_type.items(), key=lambda x: -max(x[1])):
        lo, hi = min(sts), max(sts)
        st_str = f"{lo:.2f}" if lo == hi else f"{lo:.2f} – {hi:.2f}"
        lines += f"| `{stype}` | {st_str} | {len(sts)} |\n"
    return lines.rstrip()


# ── Source Registry ────────────────────────────────────────────────────────────

def _render_registry(cfg: "AppConfig") -> None:
    from app.core.loaders import load_registry
    import pandas as pd

    records = load_registry(cfg.registry_path)
    if not records:
        st.info(f"Registry not found at `{cfg.registry_path}` — run `just registry` to generate it.")
        return

    df_all = pd.DataFrame(records)

    # Summary strip
    c_chart, c_stats = st.columns([3, 1])
    with c_chart:
        st.markdown("#### Source Trust Distribution")
        st.bar_chart(df_all["source_trust"].round(2).value_counts().sort_index())
    with c_stats:
        st.markdown("&nbsp;")
        st.metric("Sources",   str(len(records)))
        st.metric("Mean ST",   f"{df_all['source_trust'].mean():.3f}")
        st.metric("Median ST", f"{df_all['source_trust'].median():.3f}")

    st.divider()

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        search_q = st.text_input(
            "Search source_id / domain / name", key="ref_reg_search",
            placeholder="e.g. reuters, academic, .gov",
        )
    with c2:
        all_types = sorted({r.get("source_type", "?") for r in records})
        sel_type  = st.multiselect("Source Type", all_types, key="ref_reg_type",
                                   placeholder="All types")
    with c3:
        all_mods = sorted({r.get("modality", "?") for r in records})
        sel_mod  = st.multiselect("Modality", all_mods, key="ref_reg_modality",
                                  placeholder="All modalities")

    st_min, st_max = st.slider(
        "Source Trust range", 0.0, 1.0, (0.0, 1.0), 0.05, key="ref_reg_trust_range",
    )

    # Filter
    filtered = records
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = [
            r for r in filtered
            if q in r.get("source_id",  "").lower()
            or q in r.get("domain",      "").lower()
            or q in r.get("source_name", "").lower()
        ]
    if sel_type:
        filtered = [r for r in filtered if r.get("source_type") in sel_type]
    if sel_mod:
        filtered = [r for r in filtered if r.get("modality") in sel_mod]
    filtered = [r for r in filtered if st_min <= r.get("source_trust", 0.0) <= st_max]

    st.caption(f"Showing **{len(filtered)}** / {len(records)} entries")

    if not filtered:
        st.warning("No entries match the current filters.")
        return

    sorted_f = sorted(filtered, key=lambda x: x.get("source_trust", 0), reverse=True)
    rows = [
        {
            "source_id":   r.get("source_id",   ""),
            "name":        r.get("source_name", ""),
            "domain":      r.get("domain",       ""),
            "type":        r.get("source_type",  ""),
            "modality":    r.get("modality",     ""),
            "trust":       r.get("source_trust", 0.0),
            "prior_trust": r.get("prior_trust",  0.0),
            "default_IS":  r.get("default_inference_strength", 0.0),
        }
        for r in sorted_f
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(
            subset=["trust", "prior_trust", "default_IS"], cmap="RdYlGn", vmin=0, vmax=1
        ),
        width='stretch',
        height=min(500, 38 * len(rows) + 40),
    )

    # Detail expanders
    st.divider()
    st.markdown("#### Entry Detail (top 20)")
    for r in sorted_f[:20]:
        trust = r.get("source_trust", 0.0)
        color = "🟢" if trust >= 0.8 else "🟡" if trust >= 0.6 else "🔴"
        with st.expander(
            f"{color} `{r['source_id']}` — {r.get('source_name', '')}  ·  ST={trust:.3f}",
            expanded=False,
        ):
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown(f"**Domain:** `{r.get('domain', '—')}`")
                st.markdown(f"**Type:** `{r.get('source_type', '—')}`")
                st.markdown(f"**Modality:** `{r.get('modality', '—')}`")
                ev_types = ", ".join(r.get("default_evidence_types", []))
                st.markdown(f"**Evidence types:** `{ev_types or '—'}`")
            with c_right:
                st.metric("Source Trust", f"{r.get('source_trust', 0):.3f}")
                st.metric("Prior Trust",  f"{r.get('prior_trust',  0):.3f}")
                st.metric("Default IS",   f"{r.get('default_inference_strength', 0):.3f}")
            meta = r.get("trust_metadata", {})
            if meta:
                st.caption(
                    f"Methodology: `{meta.get('methodology_ref', '—')}`  ·  "
                    f"Version: `{meta.get('trust_version', '—')}`"
                )


# ── Design Assumptions ─────────────────────────────────────────────────────────

def _render_assumptions() -> None:
    st.markdown("### Epistemic Assumptions")
    st.markdown(
        "These modelling choices are fixed at training / inference time. "
        "Changing any of them requires re-training or re-evaluation.  \n"
        "*All assumptions are backed by an accepted ADR — see the ADR & Docs tab for details.*"
    )

    sections = [
        (
            "Source Trust (ST)  — ADR-009, ADR-020",
            [
                "ST is assigned statically from the registry; it does not adapt per-claim.",
                "Unknown sources receive ST = 0.40 (same as the hard fallback `DEFAULT_SOURCE_TRUST`). "
                "Unknown sources rank above social media (≤ 0.35) because identity is unverifiable but not inherently partisan.",
                "Web archives (web.archive.org) resolve to the original domain's ST (ADR-020).",
                "Social-media sources (twitter, reddit, …) are capped at ST ≤ 0.35.",
            ],
        ),
        (
            "Inference Strength (IS)  — ADR-010, ADR-022",
            [
                "IS is a learned regression head output ∈ [0, 1].",
                "IS is detached from the verdict gradient to prevent shortcut learning (ADR-022). "
                "Before detach, IS RMSE was 0.234 (competing gradients); after detach ~0.12.",
                "IS captures logical binding strength, not stance direction.",
                "Postulation/derivation evidence is excluded from IS training (ADR-005).",
            ],
        ),
        (
            "Evidence Confidence (EC)  — ADR-010, ADR-001",
            [
                "EC_i = 1 − (1 − ST_i)^(EW_i × IS_i)  — multiplicative-complementary form.",
                "EW (evidence weight) = stance probability from the stance head (H1 or NLI).",
                "The decisive threshold θ is a per-checkpoint hyperparameter (default 0.35, tuned via Optuna).",
                "EC aggregate uses product-of-complements: EC_d = 1 − ∏(1 − EC_i) for stance d.",
            ],
        ),
        (
            "Verdict Decision Logic  — ADR-027",
            [
                "If EC_support > θ AND EC_refute > θ → treat as conflicting, fall through to VerdictHead.",
                "If only EC_support > θ → hard SUPPORTED override.",
                "If only EC_refute  > θ → hard REFUTED override.",
                "Otherwise → VerdictHead softmax output decides (full delegation — ADR-027 removed _EC_NEI_MAX branch).",
                "Baseline model always uses VerdictHead (no EC computation).",
            ],
        ),
        (
            "Graph Construction  — ADR-003, ADR-024, ADR-026",
            [
                "Each claim becomes a single CLAIM node; each evidence item becomes an EVIDENCE node.",
                "HeteroData edges: (EVIDENCE, supports/refutes/neutral, CLAIM).",
                "v3-nli adds 3d NLI probability columns (offline DeBERTa-v3-small) to evidence features (408d = 405d base + 3d NLI). "
                "H1 StanceHead runs on GNN output, refining NLI priors with graph context (ADR-024).",
                "Boilerplate, empty, and near-duplicate evidence items are filtered before graph build.",
                "Floor-plan train/test split prevents floorplan leakage (ADR-003).",
            ],
        ),
        (
            "Synthetic Data  — ADR-013, ADR-017, ADR-018",
            [
                "Synthetic data covers absence-of-evidence (non_apprehension) patterns (ADR-013).",
                "IS jitter is applied to synthetic data to prevent IS=0 shortcut (ADR-017).",
                "NEI-heavy distribution used: ≥40% NEI in synthetic split (ADR-018).",
                "Synthetic claims are labelled with full EC pipeline (ADR-019); no LLM verdict.",
            ],
        ),
    ]

    for title, bullets in sections:
        with st.expander(f"**{title}**", expanded=False):
            for b in bullets:
                st.markdown(f"- {b}")


# ── ADR & Docs ─────────────────────────────────────────────────────────────────

def _render_adr_docs() -> None:
    t_adr, t_docs = st.tabs(["ADR Files", "Project Docs"])

    with t_adr:
        _render_adrs()

    with t_docs:
        _render_docs()


def _render_adrs() -> None:
    if not _ADR_DIR.exists():
        st.info("ADR directory not found at `docs/adr/`")
        return

    adr_files = sorted(_ADR_DIR.glob("*.md"))
    if not adr_files:
        st.info("No ADR files found.")
        return

    # Build (stem → (path, title)) map
    index: list[tuple[str, Path, str]] = []
    for f in adr_files:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        title = next((l.lstrip("#").strip() for l in lines if l.strip().startswith("#")), f.stem)
        index.append((f.stem, f, title))

    options = [f"{stem}  —  {title}" for stem, _, title in index]
    sel_idx = st.selectbox(
        f"Select ADR ({len(index)} files)",
        range(len(options)),
        format_func=lambda i: options[i],
        key="ref_adr_sel",
    )

    if sel_idx is not None:
        _, fpath, _ = index[sel_idx]
        content = fpath.read_text(encoding="utf-8", errors="replace")
        with st.container(border=True):
            st.markdown(content)

    st.divider()
    st.caption("ADR = Architecture Decision Record. Each documents a design choice, its context, and consequences.")


def _render_docs() -> None:
    if not _DOCS_DIR.exists():
        st.info("docs/ directory not found.")
        return

    doc_files = sorted(f for f in _DOCS_DIR.glob("*.md") if f.is_file())
    if not doc_files:
        st.info("No markdown files found in docs/.")
        return

    for f in doc_files:
        label = f.stem.replace("-", " ").replace("_", " ").title()
        with st.expander(f"**{label}**  (`{f.name}`)", expanded=False):
            content = f.read_text(encoding="utf-8", errors="replace")
            st.markdown(content)
