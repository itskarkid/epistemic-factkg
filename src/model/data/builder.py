"""Build per-claim HeteroData graphs for EpistemicHGNN (V1).

Graph structure per claim:
  Nodes:  claim (1), evidence (N_ev), triple (N_tr, AI2THOR only)
  Edges:  has_evidence, connected_to, co_evidence, has_triple, from_triple

No stance-typed edges — stance is learned by H1 from supervised labels.
EW and ST are stored as separate tensors (not encoder input features).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from src.epistemic.formula import combine_evidence_weights
from src.epistemic.registry import get_source_trust, load_source_trust_registry
from src.model.data.featurizer import Featurizer
from src.model.data.types import (
    STANCE_TO_INT,
    VERDICT_TO_INT,
    ClaimGraph,
    EdgeType,
    NodeType,
    get_source_category,
)


_BOILERPLATE = frozenset(["no sensor evidence found for this object type."])
_VAGUE_PLACEHOLDER = re.compile(
    r"\bthe described (element|substance|compound|entity|object|material|item)\b",
    re.IGNORECASE,
)


class ClaimGraphBuilder:
    """Converts a v3.0 unified record dict into a PyG HeteroData subgraph."""

    def __init__(
        self,
        registry: dict[str, dict],
        featurizer: Featurizer,
        use_nli: bool = False,
    ) -> None:
        self.registry = registry
        self.featurizer = featurizer
        self.use_nli = use_nli

    @classmethod
    def from_paths(
        cls,
        registry_path: str | Path,
        embed_cache_path: str | Path | None = None,
    ) -> ClaimGraphBuilder:
        registry = load_source_trust_registry(registry_path)
        featurizer = Featurizer(cache_path=embed_cache_path)
        return cls(registry, featurizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_items(
        self,
        claim: str,
        resolved_items: list[dict],
    ) -> ClaimGraph | None:
        """Build a ClaimGraph from pre-resolved evidence items (app inference path).

        resolved_items elements must have: text, modality, source_id, resolved_trust.
        - evidence_types derived from _MODALITY_TO_EVIDENCE_TYPES (src/)
        - verdict and reasoning omitted — builder already defaults both
        """
        from src.model.data.types import _MODALITY_TO_EVIDENCE_TYPES

        evidence = []
        for ev in resolved_items:
            modality = ev.get("modality", "web_text")
            evidence.append({
                "text":           ev["text"],
                "stance":         None,
                "modality":       modality,
                "evidence_types": _MODALITY_TO_EVIDENCE_TYPES.get(modality, ["testimony"]),
                "source_id":      ev.get("source_id") or f"{ev.get('source_type', 'unknown')}_{modality}",
                "triples":        ev.get("triples") or [],
            })
        return self.build({"claim": claim, "evidence": evidence})

    def build(self, record: dict) -> ClaimGraph | None:
        """Convert a single v3.0 record to a ClaimGraph.

        Returns None if no evidence remains after filtering, or if verdict is
        unknown. Callers should filter to training evidence types before calling build().
        """
        data = HeteroData()
        evidence_items = record.get("evidence", [])

        # Drop pre-fix AI2THOR boilerplate absence evidence (replaced by Fix A)
        evidence_items = [
            ev for ev in evidence_items
            if (ev.get("text") or "").strip().lower() not in _BOILERPLATE
        ]

        # Fix I: drop evidence with unfilled synthetic template placeholders
        evidence_items = [
            ev for ev in evidence_items
            if not _VAGUE_PLACEHOLDER.search(ev.get("text") or "")
        ]

        # Drop evidence items with empty or whitespace-only text
        evidence_items = [ev for ev in evidence_items if (ev.get("text") or "").strip()]

        # Fix J: drop near-duplicate evidence items (full-text fingerprint)
        _seen_ev: set[str] = set()
        _deduped = []
        for ev in evidence_items:
            key = (ev.get("text") or "").strip().lower()
            if key not in _seen_ev:
                _seen_ev.add(key)
                _deduped.append(ev)
        evidence_items = _deduped

        # ── Claim node ────────────────────────────────────────────────
        strategy = self._extract_strategy(record)
        claim_text_emb = self.featurizer.encode_texts([record["claim"]])  # [1, 384]
        claim_strategy = self.featurizer.encode_reasoning_strategy(strategy).unsqueeze(
            0
        )  # [1, 6]
        data[NodeType.CLAIM].x = torch.cat(
            [claim_text_emb, claim_strategy], dim=1
        )  # [1, 390]

        # ── Evidence nodes ────────────────────────────────────────────
        n_ev = len(evidence_items)
        if n_ev == 0:
            return None  # no evidence remains after filtering — skip record

        nli_probs = None
        if self.use_nli:
            ev_texts = [ev.get("text", "") for ev in evidence_items]
            nli_probs = self.featurizer.encode_nli_stance(
                record["claim"], ev_texts
            )  # [N_ev, 3]
        ev_features, stance_y, is_y, ew_vals, st_vals = self._build_evidence(
            evidence_items, nli_probs=nli_probs
        )
        data[NodeType.EVIDENCE].x = ev_features  # [N_ev, 400] or [N_ev, 403]
        data[NodeType.EVIDENCE].stance_y = stance_y  # [N_ev]
        data[NodeType.EVIDENCE].is_y = is_y  # [N_ev]
        data[NodeType.EVIDENCE].ew = ew_vals  # [N_ev]
        data[NodeType.EVIDENCE].st = st_vals  # [N_ev]

        # ── Triple nodes (AI2THOR only) ───────────────────────────────
        triple_texts, has_triple_idx, from_triple_idx = self._collect_triples(
            record, evidence_items
        )
        n_tr = len(triple_texts)
        if n_tr > 0:
            data[NodeType.TRIPLE].x = self.featurizer.encode_texts(
                triple_texts
            )  # [N_tr, 384]

        # ── Edges ─────────────────────────────────────────────────────
        ev_range = torch.arange(n_ev, dtype=torch.long)
        claim_idx = torch.zeros(n_ev, dtype=torch.long)

        # claim → evidence
        data[
            NodeType.CLAIM, EdgeType.HAS_EVIDENCE, NodeType.EVIDENCE
        ].edge_index = torch.stack([claim_idx, ev_range])  # [2, N_ev]

        # evidence → claim (neutral reverse)
        data[
            NodeType.EVIDENCE, EdgeType.CONNECTED_TO, NodeType.CLAIM
        ].edge_index = torch.stack([ev_range, claim_idx])  # [2, N_ev]

        # evidence → evidence (windowed co-evidence: max 5 nearest neighbours per node)
        # Fully-connected O(N²) caused over-smoothing for dense AVeriTeC claims (max 16 ev).
        _MAX_CO_EV = 5
        if n_ev > 1:
            pairs = []
            for i in range(n_ev):
                near = sorted(
                    [j for j in range(n_ev) if j != i], key=lambda j: abs(j - i)
                )
                for j in near[:_MAX_CO_EV]:
                    pairs.append((i, j))
            src, dst = zip(*pairs)
            data[
                NodeType.EVIDENCE, EdgeType.CO_EVIDENCE, NodeType.EVIDENCE
            ].edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
        else:
            data[
                NodeType.EVIDENCE, EdgeType.CO_EVIDENCE, NodeType.EVIDENCE
            ].edge_index = torch.zeros(2, 0, dtype=torch.long)

        # AI2THOR triple edges
        if n_tr > 0 and has_triple_idx:
            c_src, t_dst = zip(*has_triple_idx)
            data[
                NodeType.CLAIM, EdgeType.HAS_TRIPLE, NodeType.TRIPLE
            ].edge_index = torch.tensor([list(c_src), list(t_dst)], dtype=torch.long)
        if n_tr > 0 and from_triple_idx:
            e_src, t_dst = zip(*from_triple_idx)
            data[
                NodeType.EVIDENCE, EdgeType.FROM_TRIPLE, NodeType.TRIPLE
            ].edge_index = torch.tensor([list(e_src), list(t_dst)], dtype=torch.long)

        label = VERDICT_TO_INT.get(
            record.get("verdict", {}).get("label", "not_enough_evidence"), 2
        )
        data[NodeType.CLAIM].y = torch.tensor([label], dtype=torch.long)
        dataset = record.get("provenance", {}).get("dataset", "unknown")
        return ClaimGraph(data=data, label=label, dataset=dataset)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_strategy(self, record: dict) -> str:
        return (record.get("reasoning") or {}).get("strategy") or "testimonial_lookup"

    def _build_evidence(
        self, evidence_items: list[dict], nli_probs: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        texts = [ev.get("text", "") for ev in evidence_items]
        text_embs = self.featurizer.encode_texts(texts)  # [N_ev, 384]

        modality_vecs, et_vecs, src_vecs = [], [], []
        stance_ints, is_vals, ew_vals, st_vals = [], [], [], []

        for ev in evidence_items:
            modality_vecs.append(self.featurizer.encode_modality(ev.get("modality")))
            et_vecs.append(
                self.featurizer.encode_evidence_types(ev.get("evidence_types", []))
            )
            cat = get_source_category(ev.get("source_id", "unknown_web"), self.registry)
            src_vecs.append(self.featurizer.encode_source_type(cat))

            stance_ints.append(
                STANCE_TO_INT.get(ev.get("stance", "not_enough_evidence"), 2)
            )
            is_vals.append(float(ev.get("inference_strength", 0.6)))
            ew_vals.append(combine_evidence_weights(ev.get("evidence_types", [])))
            st_vals.append(
                get_source_trust(ev.get("source_id", "unknown_web"), self.registry)
            )

        ev_features = torch.cat(
            [
                text_embs,
                torch.stack(modality_vecs),
                torch.stack(et_vecs),
                torch.stack(src_vecs),
            ],
            dim=1,
        )  # [N_ev, 400]

        if nli_probs is not None and nli_probs.shape[0] == ev_features.shape[0]:
            ev_features = torch.cat([ev_features, nli_probs], dim=1)  # [N_ev, 403]

        return (
            ev_features,
            torch.tensor(stance_ints, dtype=torch.long),
            torch.tensor(is_vals, dtype=torch.float32),
            torch.tensor(ew_vals, dtype=torch.float32),
            torch.tensor(st_vals, dtype=torch.float32),
        )

    def _collect_triples(
        self,
        record: dict,
        evidence_items: list[dict],
    ) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int]]]:
        """Collect unique triple texts and build edge index lists.

        Returns:
            triple_texts:    List of "s p o" strings (deduplicated).
            has_triple_idx:  [(claim_idx=0, triple_idx), ...] for claim-level triples.
            from_triple_idx: [(ev_idx, triple_idx), ...] for evidence-level triples.
        """
        triple_to_idx: dict[str, int] = {}
        has_triple_idx: list[tuple[int, int]] = []
        from_triple_idx: list[tuple[int, int]] = []

        def _register(triple: list) -> int:
            key = f"{triple[0]} {triple[1]} {triple[2]}"
            if key not in triple_to_idx:
                triple_to_idx[key] = len(triple_to_idx)
            return triple_to_idx[key]

        # Claim-level triples → has_triple edges
        for triple in record.get("claim_triples") or []:
            if len(triple) == 3:
                t_idx = _register(triple)
                has_triple_idx.append((0, t_idx))

        # Evidence-level triples → from_triple edges
        for ev_idx, ev in enumerate(evidence_items):
            for triple in ev.get("triples") or []:
                if len(triple) == 3:
                    t_idx = _register(triple)
                    from_triple_idx.append((ev_idx, t_idx))

        return list(triple_to_idx.keys()), has_triple_idx, from_triple_idx
