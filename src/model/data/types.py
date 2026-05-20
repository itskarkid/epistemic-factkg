"""Node/edge type constants and graph container for EpistemicHGNN (V1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

import torch
from torch_geometric.data import HeteroData

from src.epistemic.formula import CONFIDENCE_WEIGHTS, combine_evidence_weights

_EMBED_DIM = 384


class NodeType(StrEnum):
    CLAIM = auto()
    EVIDENCE = auto()
    TRIPLE = auto()


class EdgeType(StrEnum):
    HAS_EVIDENCE = auto()  # claim → evidence
    CONNECTED_TO = auto()  # evidence → claim (neutral reverse)
    CO_EVIDENCE = auto()  # evidence → evidence (fully connected within claim)
    HAS_TRIPLE = auto()  # claim → triple (AI2THOR only)
    FROM_TRIPLE = auto()  # evidence → triple (AI2THOR only)


# ── Verdict ──────────────────────────────────────────────────────────────────
class VERDICT(StrEnum):
    SUPPORTED = auto()
    REFUTED = auto()
    NOT_ENOUGH_EVIDENCE = auto()

VERDICT_TO_INT: dict[str, int] = {
    VERDICT.SUPPORTED.value: 0,
    VERDICT.REFUTED.value: 1,
    VERDICT.NOT_ENOUGH_EVIDENCE.value: 2,
}
NUM_VERDICT = len(VERDICT_TO_INT)  # 3


# ── Stance (H1 supervision labels) ───────────────────────────────────────────
# not_enough_evidence and conflicting_evidence map to neutral.
class STANCE(StrEnum):
    SUPPORTS = auto()
    REFUTES = auto()
    NOT_ENOUGH_EVIDENCE = auto()
    CONFLICTING_EVIDENCE = auto()

STANCE_TO_INT: dict[str, int] = {
    STANCE.SUPPORTS.value: 0,
    STANCE.REFUTES.value: 1,
    STANCE.NOT_ENOUGH_EVIDENCE.value: 2,
    STANCE.CONFLICTING_EVIDENCE.value: 2,
}
NUM_STANCE = 3  # supports / refutes / neutral


# ── Evidence type (multi-hot, 5 training types) ───────────────────────────────
class EVIDENCE_TYPE(StrEnum):
    PERCEPTION = auto()
    TESTIMONY = auto()
    NON_APPREHENSION = auto()
    COMPARISON_ANALOGY = auto()
    INFERENCE = auto()

EVIDENCE_TYPE_TO_INT: dict[str, int] = {
    EVIDENCE_TYPE.PERCEPTION.value: 0,
    EVIDENCE_TYPE.TESTIMONY.value: 1,
    EVIDENCE_TYPE.NON_APPREHENSION.value: 2,
    EVIDENCE_TYPE.COMPARISON_ANALOGY.value: 3,
    EVIDENCE_TYPE.INFERENCE.value: 4,
}
NUM_EVIDENCE_TYPE = len(EVIDENCE_TYPE_TO_INT)  # 5


# ── Reasoning strategy (one-hot, claim node) ─────────────────────────────────
class REASONING_STRATEGY(StrEnum):
    DIRECT_OBSERVATION = auto()
    ABSENCE_DETECTION = auto()
    SPATIAL_COMPARISON = auto()
    TESTIMONIAL_LOOKUP = auto()
    MULTI_HOP_INFERENCE = auto()
    CONFLICTING_EVIDENCE = auto()

REASONING_STRATEGY_TO_INT: dict[str, int] = {
    REASONING_STRATEGY.DIRECT_OBSERVATION.value: 0,
    REASONING_STRATEGY.ABSENCE_DETECTION.value: 1,
    REASONING_STRATEGY.SPATIAL_COMPARISON.value: 2,
    REASONING_STRATEGY.TESTIMONIAL_LOOKUP.value: 3,
    REASONING_STRATEGY.MULTI_HOP_INFERENCE.value: 4,
    REASONING_STRATEGY.CONFLICTING_EVIDENCE.value: 5,
}
NUM_REASONING_STRATEGY = len(REASONING_STRATEGY_TO_INT)  # 6


# ── Modality (one-hot, evidence node) ────────────────────────────────────────
class MODALITY(StrEnum):
    SENSOR = auto()               # 0 — physical sensor / IoT / AI2THOR
    WEB_TEXT = auto()             # 1 — HTML web page
    VIDEO = auto()                # 2
    AUDIO = auto()                # 3
    IMAGE = auto()                # 4
    PDF = auto()                  # 5 — academic/government PDF document
    WEB_TABLE = auto()            # 6 — structured HTML table
    ANNOTATOR_KNOWLEDGE = auto()  # 7 — AVeriTeC annotator background knowledge
    OTHER = auto()                # 8 — LLM-generated synthetic / misc
    UNKNOWN = auto()              # 9 — unmapped / unanswerable

MODALITY_TO_INT: dict[str, int] = {
    MODALITY.SENSOR.value:              0,
    MODALITY.WEB_TEXT.value:            1,
    MODALITY.VIDEO.value:               2,
    MODALITY.AUDIO.value:               3,
    MODALITY.IMAGE.value:               4,
    MODALITY.PDF.value:                 5,
    MODALITY.WEB_TABLE.value:           6,
    MODALITY.ANNOTATOR_KNOWLEDGE.value: 7,
    MODALITY.OTHER.value:               8,
    MODALITY.UNKNOWN.value:             9,
}
NUM_MODALITY = len(MODALITY_TO_INT)  # 10


# ── Source type (one-hot, evidence node) ─────────────────────────────────────

class SOURCE_TYPE(StrEnum):
    NEWS_MEDIA = auto()
    ACADEMIC = auto()
    GOVERNMENT = auto()
    SOCIAL_MEDIA = auto()
    SENSOR = auto()
    UNKNOWN = auto()  # includes unanswerable + other + missing registry entries

SOURCE_TYPE_TO_INT: dict[str, int] = {
    SOURCE_TYPE.NEWS_MEDIA.value: 0,
    SOURCE_TYPE.ACADEMIC.value: 1,
    SOURCE_TYPE.GOVERNMENT.value: 2,
    SOURCE_TYPE.SOCIAL_MEDIA.value: 3,
    SOURCE_TYPE.SENSOR.value: 4,      # real-world direct measurement (slot previously held by simulation)
    SOURCE_TYPE.UNKNOWN.value: 5,
}
NUM_SOURCE_TYPE = len(SOURCE_TYPE_TO_INT)  # 6

# Maps registry source_type values → the 6 encoder categories.
class _REGISTRY_TYPE(StrEnum):
    NEWS_MEDIA = auto()
    FACT_CHECKER = auto()
    TESTIMONY = auto()
    GOVERNMENT = auto()
    SCIENTIFIC_PAPER = auto()
    KNOWLEDGE_GRAPH = auto()
    SOCIAL_MEDIA = auto()
    SENSOR = auto()
    SIMULATION = auto()
    WEB_ARCHIVE = auto()
    LLM_GENERATED = auto()
    NGO_OR_ORG = auto()
    WEB_TEXT = auto()
    UNKNOWN = auto()

_REGISTRY_TYPE_TO_CATEGORY: dict[str, str] = {
    _REGISTRY_TYPE.NEWS_MEDIA.value: SOURCE_TYPE.NEWS_MEDIA.value,
    _REGISTRY_TYPE.FACT_CHECKER.value: SOURCE_TYPE.NEWS_MEDIA.value,
    _REGISTRY_TYPE.TESTIMONY.value: SOURCE_TYPE.UNKNOWN.value,    # annotator_knowledge → unknown, not news
    _REGISTRY_TYPE.GOVERNMENT.value: SOURCE_TYPE.GOVERNMENT.value,
    _REGISTRY_TYPE.SCIENTIFIC_PAPER.value: SOURCE_TYPE.ACADEMIC.value,
    _REGISTRY_TYPE.KNOWLEDGE_GRAPH.value: SOURCE_TYPE.ACADEMIC.value,
    _REGISTRY_TYPE.SOCIAL_MEDIA.value: SOURCE_TYPE.SOCIAL_MEDIA.value,
    _REGISTRY_TYPE.SENSOR.value: SOURCE_TYPE.SENSOR.value,           # physical sensor / IoT → slot 4
    _REGISTRY_TYPE.SIMULATION.value: SOURCE_TYPE.SENSOR.value,       # AI2THOR simulation maps to sensor category
    _REGISTRY_TYPE.WEB_ARCHIVE.value: SOURCE_TYPE.UNKNOWN.value,
    _REGISTRY_TYPE.LLM_GENERATED.value: SOURCE_TYPE.UNKNOWN.value,
    _REGISTRY_TYPE.NGO_OR_ORG.value: SOURCE_TYPE.UNKNOWN.value,
    _REGISTRY_TYPE.WEB_TEXT.value: SOURCE_TYPE.UNKNOWN.value,
    _REGISTRY_TYPE.UNKNOWN.value: SOURCE_TYPE.UNKNOWN.value,
}


def get_source_category(source_id: str, registry: dict[str, dict]) -> str:
    """Resolve a source_id to one of the 6 encoder source-type categories."""
    entry = registry.get(source_id, {})
    source_type = entry.get("source_type", "unknown")
    return _REGISTRY_TYPE_TO_CATEGORY.get(source_type, "unknown")


# ── Epistemic defaults (display hints for app UI) ─────────────────────────────
# Semantic mapping: how each modality is delivered → what reasoning type it supports.
# EW is computed from these via combine_evidence_weights() (formula.py),
# so any change to CONFIDENCE_WEIGHTS is automatically reflected here.
_MODALITY_TO_EVIDENCE_TYPES: dict[str, list[str]] = {
    MODALITY.SENSOR.value:               [EVIDENCE_TYPE.PERCEPTION.value],
    MODALITY.WEB_TEXT.value:             [EVIDENCE_TYPE.TESTIMONY.value],
    MODALITY.VIDEO.value:                [EVIDENCE_TYPE.PERCEPTION.value],
    MODALITY.AUDIO.value:                [EVIDENCE_TYPE.PERCEPTION.value],
    MODALITY.IMAGE.value:                [EVIDENCE_TYPE.PERCEPTION.value],
    MODALITY.PDF.value:                  [EVIDENCE_TYPE.TESTIMONY.value],
    MODALITY.WEB_TABLE.value:            [EVIDENCE_TYPE.COMPARISON_ANALOGY.value,
                                          EVIDENCE_TYPE.TESTIMONY.value],
    MODALITY.ANNOTATOR_KNOWLEDGE.value:  [EVIDENCE_TYPE.TESTIMONY.value],
    MODALITY.OTHER.value:                [EVIDENCE_TYPE.TESTIMONY.value],
    MODALITY.UNKNOWN.value:              [EVIDENCE_TYPE.TESTIMONY.value],
}


def get_epistemic_defaults(modality: str, source_trust: float) -> tuple[float, float]:
    """Return (evidence_weight, is_seed) display hints for the app UI.

    EW is computed from modality → evidence types → combine_evidence_weights()
    using CONFIDENCE_WEIGHTS from epistemic/formula.py (single source of truth).

    IS seed = min(0.8, max(0.1, source_trust)) — the training label formula;
    the model predicts IS at inference; this is only a prior estimate.
    Pass the fallback source trust for the selected source_type category.

    Args:
        modality:      String value from MODALITY enum (e.g. MODALITY.WEB_TEXT.value).
        source_trust:  Source trust float from the registry for the selected source.

    Returns:
        (ew, is_seed) both rounded to 3 decimal places.
    """
    ev_types = _MODALITY_TO_EVIDENCE_TYPES.get(
        modality, [EVIDENCE_TYPE.TESTIMONY.value]
    )
    ew = combine_evidence_weights(ev_types, CONFIDENCE_WEIGHTS)
    is_seed = min(0.8, max(0.1, source_trust))
    return round(ew, 3), round(is_seed, 3)


# ── Node feature dimensions ───────────────────────────────────────────────────

CLAIM_DIM = _EMBED_DIM + NUM_REASONING_STRATEGY  # 390
EVIDENCE_DIM = _EMBED_DIM + NUM_MODALITY + NUM_EVIDENCE_TYPE + NUM_SOURCE_TYPE  # 405
EVIDENCE_DIM_NLI = EVIDENCE_DIM + 3  # 408 — with NLI stance probs appended
TRIPLE_DIM = _EMBED_DIM


# ── Graph container ───────────────────────────────────────────────────────────


@dataclass
class ClaimGraph:
    """A single claim's heterogeneous subgraph.

    data["evidence"].stance_y  — int tensor [N_ev]:   H1 supervision labels
    data["evidence"].is_y      — float tensor [N_ev]: H2 supervision targets
    data["evidence"].ew        — float tensor [N_ev]: epistemic weight (not encoder input)
    data["evidence"].st        — float tensor [N_ev]: source trust (not encoder input)
    """

    data: HeteroData
    label: int  # verdict integer for final eval
    dataset: str  # provenance.dataset for per-source breakdown

    def to(self, device: torch.device) -> ClaimGraph:
        return ClaimGraph(
            data=self.data.to(device),
            label=self.label,
            dataset=self.dataset,
        )
