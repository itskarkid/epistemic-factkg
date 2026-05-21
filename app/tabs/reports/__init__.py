"""Reports tab — per-model training history, metrics, plots, and model comparisons."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig


# ── Per-model report ──────────────────────────────────────────────────────────

def _render_model_report(model_key: str, cfg: "AppConfig") -> None:
    from app.core.loaders import (
        load_verdict_metrics, load_stance_metrics,
        load_is_metrics, load_training_history,
    )

    vm = load_verdict_metrics(model_key,  cfg.reports_root)
    sm = load_stance_metrics(model_key,   cfg.reports_root)
    im = load_is_metrics(model_key,       cfg.reports_root)
    th = load_training_history(model_key, cfg.reports_root)

    if vm is None and th is None:
        st.info(f"No report files found for **{model_key}** — run `just run model` to generate them.")
        return

    # ── Top metric cards ──────────────────────────────────────────────────────
    if vm:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Verdict Accuracy", f"{vm['accuracy']:.1%}")
        c2.metric("Macro F1",         f"{vm['macro_f1']:.3f}")
        c3.metric("Weighted F1",      f"{vm['weighted_f1']:.3f}")
        skipped = vm.get("skipped", 0)
        c4.metric("Claims Evaluated", f"{vm['n_claims']:,}",
                  delta=f"-{skipped} skipped" if skipped else None,
                  delta_color="inverse" if skipped else "off")

    # ── Training history ──────────────────────────────────────────────────────
    if th is not None:
        history = th.get("history", th) if isinstance(th, dict) else th
        cov     = th.get("data_coverage") if isinstance(th, dict) else None

        if isinstance(history, list) and history:
            import pandas as pd
            col_loss, col_acc = st.columns(2)

            with col_loss:
                st.markdown("#### Training Loss")
                df_loss = pd.DataFrame([
                    {"Epoch": h["epoch"], "Train": h.get("train_loss"), "Val": h.get("val_loss")}
                    for h in history
                ]).set_index("Epoch")
                st.line_chart(df_loss)

            with col_acc:
                st.markdown("#### Verdict Accuracy")
                df_acc = pd.DataFrame([
                    {"Epoch": h["epoch"], "Train": h.get("train_v_acc"), "Val": h.get("val_v_acc")}
                    for h in history
                ]).set_index("Epoch")
                st.line_chart(df_acc)

        if cov:
            st.markdown("#### Data Coverage")
            col_t, col_v = st.columns(2)
            with col_t:
                st.markdown("**Train**")
                st.markdown(
                    f"Graphs: **{cov.get('train_graphs', '—')}** / "
                    f"{cov.get('train_total', '—')} total  "
                    f"(skipped {cov.get('train_skipped', 0)})"
                )
            with col_v:
                st.markdown("**Val**")
                st.markdown(
                    f"Graphs: **{cov.get('val_graphs', '—')}** / "
                    f"{cov.get('val_total', '—')} total  "
                    f"(skipped {cov.get('val_skipped', 0)})"
                )

    if not vm:
        st.info("Verdict metrics not found — run evaluation to generate them.")
        return

    # ── Detail tabs ───────────────────────────────────────────────────────────
    t_plots, t_verdict, t_stance, t_is, t_skipped = st.tabs([
        "Plots", "Verdict", "Stance", "Inf. Strength", "Skipped IDs",
    ])

    with t_plots:
        plots_dir = cfg.reports_root / model_key / "eval" / "plots"
        imgs = [
            (plots_dir / "confusion_matrix.png",   "Confusion Matrix"),
            (plots_dir / "class_f1.png",            "Per-Class F1"),
            (plots_dir / "per_source_accuracy.png", "Per-Source Accuracy"),
        ]
        cols = st.columns(len(imgs))
        for col, (img_path, caption) in zip(cols, imgs):
            if img_path.exists():
                col.image(str(img_path), caption=caption, width='stretch')
            else:
                col.caption(f"_{caption} — not generated yet_")

    with t_verdict:
        import pandas as pd
        st.markdown("##### Per-Class Breakdown")
        rows = [
            {"Class": cls, "Precision": f"{m['precision']:.3f}", "Recall": f"{m['recall']:.3f}",
             "F1": f"{m['f1']:.3f}", "N": m["support"]}
            for cls, m in vm.get("per_class", {}).items()
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Class"), width='stretch')

        conf = vm.get("confusion")
        if conf:
            st.markdown("##### Confusion Matrix")
            labels = list(vm.get("per_class", {}).keys()) or [str(i) for i in range(len(conf))]
            st.dataframe(
                pd.DataFrame(conf, index=labels, columns=labels),
                width='stretch',
            )

        per_src = vm.get("per_source")
        if per_src:
            st.markdown("##### Per-Source Accuracy")
            st.dataframe(
                pd.DataFrame([
                    {"Source": s, "Accuracy": f"{m['accuracy']:.1%}", "N": m["support"]}
                    for s, m in per_src.items()
                ]).set_index("Source"),
                width='stretch',
            )

    with t_stance:
        if sm:
            import pandas as pd
            ca, cb, cc, cd = st.columns(4)
            ca.metric("Accuracy", f"{sm['accuracy']:.1%}")
            cb.metric("Macro F1", f"{sm['macro_f1']:.3f}")
            cc.metric("ECE",      f"{sm['ece']:.4f}")
            cd.metric("Evidence", f"{sm.get('n_evidence', 0):,}")
            rows = [
                {"Class": cls, "Precision": f"{m['precision']:.3f}", "Recall": f"{m['recall']:.3f}",
                 "F1": f"{m['f1']:.3f}", "N": m["support"]}
                for cls, m in sm.get("per_class", {}).items()
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Class"), width='stretch')
        else:
            st.caption("stance_metrics.json not found.")

    with t_is:
        if im:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE",      f"{im['rmse']:.4f}")
            c2.metric("Pearson r", f"{im['pearson_r']:.4f}")
            c3.metric("Pred Mean", f"{im['pred_mean']:.4f}")
            c4.metric("True Mean", f"{im['true_mean']:.4f}")
            if im.get("n_evidence"):
                st.caption(f"{im['n_evidence']:,} evidence items evaluated")
        else:
            st.caption("is_metrics.json not found.")

    with t_skipped:
        skipped_ids   = vm.get("skipped_ids", [])
        skipped_count = vm.get("skipped", 0)

        if not skipped_count:
            st.success("No claims were skipped during evaluation.")
            return

        if not skipped_ids:
            st.warning(
                f"**{skipped_count} claims skipped** but IDs not saved. "
                "Re-run evaluation to capture them."
            )
            return

        st.caption(f"{len(skipped_ids)} claims skipped.")
        import pandas as pd
        from app.core.loaders import load_test_records
        test_records = load_test_records(cfg.training_jsonl, cfg.splits_dir)

        if isinstance(skipped_ids[0], int):
            idx_to_rec   = {i: r for i, r in enumerate(test_records)}
            skipped_rows = [
                {
                    "Row":      raw_id,
                    "Claim ID": (idx_to_rec.get(raw_id) or {}).get("id", "—"),
                    "Claim":    ((idx_to_rec.get(raw_id) or {}).get("claim") or "—")[:100],
                    "Verdict":  ((idx_to_rec.get(raw_id) or {}).get("verdict") or {}).get("label", "—"),
                }
                for raw_id in skipped_ids
            ]
        else:
            id_to_rec    = {r.get("id", ""): r for r in test_records}
            skipped_rows = [
                {
                    "Claim ID": str(cid),
                    "Claim":    (id_to_rec.get(str(cid)) or {}).get("claim", "—")[:100],
                    "Verdict":  ((id_to_rec.get(str(cid)) or {}).get("verdict") or {}).get("label", "—"),
                }
                for cid in skipped_ids
            ]

        st.dataframe(
            pd.DataFrame(skipped_rows),
            width='stretch',
            height=min(400, 36 * len(skipped_rows) + 38),
        )


# ── Comparison reports ────────────────────────────────────────────────────────

_LOWER_IS_BETTER = {"Stance ECE", "IS RMSE"}


def _parse_md_table(text: str, section: str) -> list[list[str]]:
    m = re.search(rf"## {re.escape(section)}\n", text)
    if not m:
        return []
    rows: list[list[str]] = []
    for line in text[m.end():].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _render_comparison_reports(cfg: "AppConfig") -> None:
    import pandas as pd

    comp_files = sorted(cfg.reports_root.glob("comparison_*.md"))
    if not comp_files:
        st.info("`comparison_*.md` files not found. Run the pipeline to generate them.")
        return

    labels = [f.stem.replace("comparison_", "").replace("_vs_", "  vs  ") for f in comp_files]
    sel    = st.selectbox("Select comparison", labels, key="cmp_select")
    idx    = labels.index(sel)
    text   = comp_files[idx].read_text(encoding="utf-8")

    parts   = comp_files[idx].stem.replace("comparison_", "").split("_vs_")
    model_a = parts[0]
    model_b = parts[1] if len(parts) > 1 else "?"
    st.markdown(f"### {model_a}  vs  {model_b}")

    gen_m = re.search(r"\*\*Generated:\*\*\s*(.*)", text)
    if gen_m:
        st.caption(f"Generated: {gen_m.group(1).strip()}")

    # Aggregate metrics with colored delta
    st.markdown("#### Aggregate Metrics")
    agg_rows = _parse_md_table(text, "Aggregate Metrics")
    if agg_rows and len(agg_rows) >= 2:
        df_agg = pd.DataFrame(agg_rows[1:], columns=agg_rows[0])

        def _delta_style(row: pd.Series) -> list[str]:
            metric = row.iloc[0]
            styles = [""] * (len(row) - 1)
            try:
                v = float(str(row.iloc[-1]).replace("+", ""))
            except ValueError:
                styles.append("")
                return styles
            lower_better = any(lb in metric for lb in _LOWER_IS_BETTER)
            good = (v < 0) if lower_better else (v > 0)
            styles.append(
                "color:#16a34a;font-weight:600" if good and v != 0
                else "color:#dc2626;font-weight:600" if not good and v != 0
                else ""
            )
            return styles

        st.dataframe(df_agg.style.apply(_delta_style, axis=1),
                     width='stretch', hide_index=True)

    # Per-source accuracy with colored delta
    st.markdown("#### Per-Source Verdict Accuracy")
    src_rows = _parse_md_table(text, "Per-Source Verdict Accuracy")
    if src_rows and len(src_rows) >= 2:
        df_src = pd.DataFrame(src_rows[1:], columns=src_rows[0])

        def _src_style(row: pd.Series) -> list[str]:
            styles = [""] * (len(row) - 1)
            try:
                v = float(str(row.iloc[-1]).replace("+", ""))
                styles.append(
                    "color:#16a34a;font-weight:600" if v > 0
                    else "color:#dc2626;font-weight:600" if v < 0
                    else ""
                )
            except ValueError:
                styles.append("")
            return styles

        st.dataframe(df_src.style.apply(_src_style, axis=1),
                     width='stretch', hide_index=True)

    with st.expander("Raw markdown", expanded=False):
        st.markdown(text)


# ── Dataset distributions ─────────────────────────────────────────────────────

def _dist_bar(counts: dict, total: int) -> None:
    """Render a small bar-chart table for a distribution dict."""
    import pandas as pd
    if not counts or not total:
        st.caption("—")
        return
    rows = [
        {"Label": k, "Count": v, "Pct": f"{v / total:.1%}"}
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Label"), width='stretch')


def _render_dataset_distributions(cfg: "AppConfig") -> None:
    from app.core.loaders import load_split_distributions
    from app.config import enum_label

    dists = load_split_distributions(cfg.training_jsonl, cfg.splits_dir)
    if not dists:
        st.info("No split index files found. Run the pipeline to generate them.")
        return

    splits = [s for s in ("train", "val", "test") if s in dists]
    cols   = st.columns(len(splits)) if splits else []

    for col, split in zip(cols, splits):
        d = dists[split]
        with col:
            st.markdown(f"### {split.capitalize()}")
            st.metric("Total records", f"{d['total']:,}")

            src = d.get("sources", {})
            if src:
                st.markdown("**Sources**")
                for ds, cnt in sorted(src.items(), key=lambda x: -x[1]):
                    st.markdown(f"- `{ds}`: **{cnt:,}** ({cnt / d['total']:.1%})")

            structural = d.get("structural", {})
            if structural:
                sc1, sc2 = st.columns(2)
                sc1.metric("Avg evidence", structural.get("avg_evidence", 0))
                sc2.metric("Avg triples",  structural.get("avg_triples",  0))

            st.markdown("**Verdict**")
            _dist_bar(
                {enum_label(k): v for k, v in d.get("verdict", {}).items()},
                d["total"],
            )

            with st.expander("Modality", expanded=False):
                _dist_bar(
                    {enum_label(k): v for k, v in d.get("modality", {}).items()},
                    sum(d.get("modality", {}).values()) or 1,
                )

            with st.expander("Stance", expanded=False):
                _dist_bar(
                    {enum_label(k): v for k, v in d.get("stance", {}).items()},
                    sum(d.get("stance", {}).values()) or 1,
                )

            with st.expander("Evidence types", expanded=False):
                _dist_bar(
                    {enum_label(k): v for k, v in d.get("evidence_types", {}).items()},
                    sum(d.get("evidence_types", {}).values()) or 1,
                )


# ── Main render ───────────────────────────────────────────────────────────────

def render(cfg: "AppConfig") -> None:
    t_models, t_dataset, t_cmp = st.tabs(["Per-Model Reports", "Dataset", "Comparisons"])

    with t_models:
        model_tabs = st.tabs([f"  {k}  " for k in cfg.model_keys])
        for tab, mk in zip(model_tabs, cfg.model_keys):
            with tab:
                _render_model_report(mk, cfg)

    with t_dataset:
        st.caption("Per-split distribution breakdown of the training dataset.")
        _render_dataset_distributions(cfg)

    with t_cmp:
        st.caption(
            "Head-to-head comparisons between model pairs. "
            "Green Δ = improvement for left model · Red Δ = regression."
        )
        _render_comparison_reports(cfg)
