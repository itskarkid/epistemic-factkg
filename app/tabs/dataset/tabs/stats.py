"""Stats sub-tab — compact dataset dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from app.config import AppConfig


_VERDICT_ICONS = {
    "supported": "✅",
    "refuted": "❌",
    "not_enough_evidence": "➖",
    "conflicting_evidence": "⚠️",
}

_STANCE_ICONS = {
    "supports": "✅",
    "refutes": "❌",
    "not_enough_evidence": "➖",
}


def _pct(n: int | float, total: int | float) -> float:
    return n / total if total else 0.0


def _fmt_pct(n: int | float, total: int | float) -> str:
    return f"{_pct(n, total):.1%}" if total else "—"


def _progress_row(label: str, count: int, total: int, icon: str = "") -> None:
    pct = _pct(count, total)
    text = f"{icon} {label}" if icon else label

    c1, c2 = st.columns([4, 1])
    c1.caption(text)
    c2.caption(f"{count:,} · {pct:.0%}")
    st.progress(pct)


def _dist_card(
    title: str,
    dist: dict[str, int],
    icons: dict[str, str] | None = None,
    limit: int = 6,
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")

        if not dist:
            st.caption("No data available")
            return

        total = sum(dist.values()) or 1
        rows = sorted(dist.items(), key=lambda x: -x[1])

        top_label, top_count = rows[0]
        st.caption(f"Top: `{top_label}` · {_fmt_pct(top_count, total)}")

        for label, count in rows[:limit]:
            icon = (icons or {}).get(label, "")
            _progress_row(str(label), count, total, icon)

        if len(rows) > limit:
            with st.expander(f"Show {len(rows) - limit} more"):
                for label, count in rows[limit:]:
                    st.caption(
                        f"`{label}` — **{count:,}** "
                        f"({_fmt_pct(count, total)})"
                    )


def _insights_panel(
    total_records: int,
    schema_invalid: int,
    logic_warnings: int,
    avg_evidence: float,
    verdict_dist: dict[str, int],
    source_dist: dict[str, int],
    stance_dist: dict[str, int],
) -> None:
    items: list[str] = []

    if total_records == 0:
        items.append("❌ Dataset is empty")
    else:
        if schema_invalid:
            items.append(f"❌ {schema_invalid:,} records failed schema validation")
        else:
            items.append("✅ All records passed schema validation")

        if logic_warnings:
            items.append(f"⚠️ {logic_warnings:,} records have logic warnings")
        else:
            items.append("✅ No logic warnings found")

        if avg_evidence < 1:
            items.append("⚠️ Average evidence per record is low")
        else:
            items.append(f"✅ Evidence coverage: {avg_evidence:.2f} per record")

    for name, dist in {
        "verdict labels": verdict_dist,
        "sources": source_dist,
        "stances": stance_dist,
    }.items():
        if not dist:
            continue

        total = sum(dist.values()) or 1
        label, count = max(dist.items(), key=lambda x: x[1])
        pct = _pct(count, total)

        if pct >= 0.75 and len(dist) > 1:
            items.append(f"⚠️ `{label}` dominates {name} at {pct:.1%}")

    with st.container(border=True):
        for item in items:
            st.markdown(item)


def _splits_card(splits: dict[str, int], total_records: int) -> None:
    with st.container(border=True):
        st.markdown("**Dataset Splits**")

        if not splits:
            st.caption("No split information available")
            return

        total_split = sum(splits.values()) or total_records or 1

        for label in ("train", "val", "test"):
            _progress_row(label.upper(), splits.get(label, 0), total_split)


def _readiness_card(
    training: dict[str, Any],
    gnn: dict[str, Any] | None,
    structure_dist: dict[str, int],
    total_records: int,
) -> None:
    with st.container(border=True):
        st.markdown("**Training Readiness**")

        train_pass = training.get("pass")

        if train_pass is True:
            st.success("ADR-006 targets met")
        elif train_pass is False:
            st.error("ADR-006 targets not met")
        else:
            st.info("Training status unavailable")

        if not gnn:
            st.caption("No GNN readiness details available")
            return

        avg_ev = gnn.get("avg_evidence_per_record", 0)
        label_ok = gnn.get("label_balance_ok")

        absence_n = structure_dist.get("absence", 0) if structure_dist else 0

        c1, c2 = st.columns(2)
        c1.metric("Avg Ev / Record", f"{avg_ev:.2f}")
        c2.metric("Absence", _fmt_pct(absence_n, total_records), f"{absence_n:,}")

        if label_ok:
            st.caption("✅ Label balance looks acceptable")
        else:
            st.caption("⚠️ Label balance needs review")


def render(cfg: "AppConfig") -> None:
    from app.core.loaders import load_dataset_stats

    stats = load_dataset_stats(cfg.data_report_dir, cfg.splits_dir)
    if stats is None:
        st.info("No validation data found. Run `just validate` first.")
        return

    counts = stats.get("counts", {})
    coverage = stats.get("coverage", {})
    splits = stats.get("splits", {})
    dists = stats.get("distributions", {})
    training = stats.get("training", {})

    gnn = training.get("gnn_readiness") if isinstance(training, dict) else None

    total_records = counts.get("total_records", 0) or 0
    schema_valid = counts.get("schema_valid", 0) or 0
    schema_invalid = counts.get("schema_invalid", 0) or 0
    logic_warnings = counts.get("logic_warnings_records", 0) or 0

    total_evidence = coverage.get("evidence_count_sum", 0) or 0
    avg_evidence = total_evidence / total_records if total_records else 0

    schema_pct = _pct(schema_valid, total_records)

    source_dist = dists.get("dataset", {}) or training.get("source_distribution", {})
    verdict_dist = dists.get("verdict_label", {})
    evidence_types = dists.get("evidence_types_all", {})
    modality_dist = dists.get("evidence_modality", {})
    stance_dist = dists.get("evidence_stance", {})
    structure_dist = dists.get("reasoning_structural", {})

    st.subheader("Dataset Dashboard")

    # ------------------------------------------------------------------
    # KPI STRIP
    # ------------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Records", f"{total_records:,}")
    k2.metric("Schema Valid", f"{schema_pct:.1%}" if total_records else "—")
    k3.metric("Invalid", f"{schema_invalid:,}")
    k4.metric("Warnings", f"{logic_warnings:,}")
    k5.metric("Avg Evidence", f"{avg_evidence:.2f}")

    if schema_invalid:
        st.error(f"{schema_invalid:,} records failed schema validation.")

    if logic_warnings:
        st.warning(f"{logic_warnings:,} records contain logic warnings.")

    # ------------------------------------------------------------------
    # SUMMARY ROW
    # ------------------------------------------------------------------
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Dataset Quality")
        _insights_panel(
            total_records=total_records,
            schema_invalid=schema_invalid,
            logic_warnings=logic_warnings,
            avg_evidence=avg_evidence,
            verdict_dist=verdict_dist,
            source_dist=source_dist,
            stance_dist=stance_dist,
        )

    with right:
        st.markdown("### Training Status")
        _readiness_card(
            training=training,
            gnn=gnn,
            structure_dist=structure_dist,
            total_records=total_records,
        )

    st.divider()

    # ------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------
    tab_overview, tab_labels, tab_evidence, tab_source = st.tabs(
        ["Overview", "Labels", "Evidence", "Source / Structure"]
    )

    with tab_overview:
        _splits_card(splits, total_records)

    with tab_labels:
        c1, c2 = st.columns(2)
        with c1:
            _dist_card("Verdict Labels", verdict_dist, _VERDICT_ICONS, limit=5)
        with c2:
            _dist_card("Evidence Stance", stance_dist, _STANCE_ICONS, limit=5)

    with tab_evidence:
        c1, c2 = st.columns(2)
        with c1:
            _dist_card("Evidence Types", evidence_types, limit=6)
        with c2:
            _dist_card("Evidence Modality", modality_dist, limit=6)

    with tab_source:
        c1, c2 = st.columns(2)
        with c1:
            _dist_card("Source Dataset", source_dist, limit=6)
        with c2:
            _dist_card("Claim Structure", structure_dist, limit=6)

    with st.expander("Raw stats JSON", expanded=False):
        st.json(stats, expanded=1)