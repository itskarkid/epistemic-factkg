"""Graph Browser sub-tab — browse HeteroData graphs from the split-cache pickle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig


def render(cfg: "AppConfig") -> None:
    from app.core.loaders import load_graph_cache, build_graph_id_map, load_all_records_list
    from app.core.ui import build_pyvis_html

    st.markdown(
        "Browse the `HeteroData` graphs stored in the split-cache pickle. "
        "These are the graphs fed into the GNN during training / evaluation."
    )

    if not cfg.graph_cache_dir.exists():
        st.info(f"Graph cache directory `{cfg.graph_cache_dir}` not found. Run training to generate the cache.")
        return

    available_models = [
        m for m in cfg.model_keys
        if (cfg.graph_cache_dir / "graphs" / f"split_cache_{m}.pkl").exists()
    ]
    if not available_models:
        st.info("No `split_cache_*.pkl` files found.")
        return

    col_m, col_s, col_i = st.columns([2, 1, 3])
    with col_m:
        model_key = st.selectbox("Model", available_models, key="gb_model")
    with col_s:
        split = st.radio("Split", ["train", "val"], horizontal=True, key="gb_split")

    cache = load_graph_cache(model_key, cfg.graph_cache_dir)
    if cache is None:
        st.warning("Failed to load cache.")
        return

    graphs = cache.get(split, [])
    if not graphs:
        st.warning(f"No graphs for split '{split}'.")
        return

    with col_i:
        idx = st.slider("Graph index", 0, len(graphs) - 1, 0, key="gb_idx",
                        help=f"{len(graphs)} graphs in '{split}' split")

    g = graphs[idx]

    try:
        claim_y    = int(g["claim"].y[0].item())
        verdict_lbl = cfg.int_to_verdict.get(claim_y, str(claim_y))
    except Exception:
        verdict_lbl = "?"

    n_claim = g["claim"].x.shape[0]
    n_ev    = g["evidence"].x.shape[0]

    try:
        id_map      = build_graph_id_map(model_key, cfg.training_jsonl, cfg.splits_dir, cfg.graph_cache_dir)
        rev_map     = {v: k for k, v in id_map.items()}
        matched_cid = rev_map.get((split, idx))
    except Exception:
        matched_cid = None

    claim_text = ""
    ev_texts: list[str] = []
    if matched_cid:
        for rec in load_all_records_list(cfg.training_jsonl):
            if rec.get("id") == matched_cid:
                claim_text = rec.get("claim", "")
                ev_texts   = [e.get("text", "") for e in rec.get("evidence", [])]
                break

    st.markdown(
        f"**Split:** `{split}` · **Index:** {idx}/{len(graphs)-1} · "
        f"**Verdict (GT):** `{verdict_lbl}` · **Nodes:** {n_claim} claim + {n_ev} evidence"
        + (f"  · **Claim ID:** `{matched_cid}`" if matched_cid else "")
    )
    if claim_text:
        st.markdown(f"> {claim_text}")

    t_graph, t_arc = st.tabs(["Interactive Graph", "Computation Flow"])

    with t_graph:
        try:
            html = build_pyvis_html(g, claim_text=claim_text, ev_texts=ev_texts, height="520px")
            st.iframe(html, height=540)
        except Exception as e:
            st.warning(f"Graph visualization unavailable: {e}")

    with t_arc:
        try:
            from app.core.loaders import get_predictor
            predictor = get_predictor(model_key, cfg.graph_cache_dir, cfg.registry_path)
            if isinstance(predictor, str):
                st.info(f"Model not loaded ({predictor}) — showing generic flow.")
                from src.model.models import MODELS
                m = MODELS[model_key]()
                dot = m.arc_block_definition().to_dot()
            else:
                dot = predictor.arc_block_definition().to_dot()
            st.graphviz_chart(dot)
        except Exception as e:
            st.warning(f"Computation flow unavailable: {e}")

    with st.expander("Raw HeteroData repr", expanded=False):
        st.code(str(g), language=None)
