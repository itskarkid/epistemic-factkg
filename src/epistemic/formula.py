"""EC formula — epistemic confidence computation (the core math).

All verdicts are derived from:
  EC_i = 1 - (1 - ST_i)^(EW_i * IS_i)
  SupportScore = 1 - ∏(1 - EC_i) over supporters
  RefuteScore = 1 - ∏(1 - EC_i) over refuters
  Verdict = thresholds applied to (SupportScore, RefuteScore)

This module is implementation-agnostic. It's used by:
- Adapters to verify synthetic data
- Model symbolic aggregator at inference
- Evaluators to recompute verdicts
"""

from src.epistemic.enums import EvidenceStance, EvidenceType, Verdict


# EW_i weights used in EC_i = 1 - (1 - ST_i)^(EW_i * IS_i).
# These are the epistemic-type weights for the diminishing-returns formula.
CONFIDENCE_WEIGHTS: dict[EvidenceType, float] = {
    EvidenceType.PERCEPTION: 0.95,
    EvidenceType.TESTIMONY: 0.80,
    EvidenceType.NON_APPREHENSION: 0.75,
    EvidenceType.COMPARISON_ANALOGY: 0.65,
    EvidenceType.INFERENCE: 0.55,
    EvidenceType.POSTULATION_DERIVATION: 0.40,
}

TRAINING_EVIDENCE_TYPES: frozenset[str] = frozenset(
    {
        EvidenceType.PERCEPTION,
        EvidenceType.TESTIMONY,
        EvidenceType.NON_APPREHENSION,
        EvidenceType.COMPARISON_ANALOGY,
        EvidenceType.INFERENCE,
    }
)

# EC_i below this floor is overridden to stance = not_enough_evidence
MIN_EVIDENCE_CONFIDENCE: float = 0.10

# Verdict derivation thresholds (see ADR-014)
SUPPORT_THRESHOLD: float = 0.75
REFUTE_THRESHOLD: float = 0.75
CONFLICT_FLOOR: float = 0.40


def combine_evidence_weights(
    evidence_types: list[str], weights: dict | None = None
) -> float:
    """Diminishing-returns combination: 1 - Π(1 - wᵢ).

    Computes EW_i for an evidence item with multiple epistemic types.
    A single type returns its own weight; multiple types always yield a higher
    combined weight than any individual, with the strongest dominating.
    """
    w = weights or CONFIDENCE_WEIGHTS
    complement = 1.0
    for et in evidence_types:
        complement *= 1.0 - w.get(et, 0.0)
    return round(1.0 - complement, 4)




def compute_evidence_confidence(st: float, ew: float, inference_strength: float) -> float:
    """EC_i = 1 - (1 - ST_i)^(EW_i * IS_i).

    Args:
        st:                 Source trustworthiness ST_i from registry (0–1).
        ew:                 Epistemic-type weight EW_i = combine_evidence_weights(evidence_types) (0–1).
        inference_strength: IS_i from rubric (0–1).

    Returns:
        EC_i in [0, 1] rounded to 4 decimal places.
    """
    exponent = ew * inference_strength
    if exponent == 0.0:
        return 0.0
    return round(1.0 - (1.0 - st) ** exponent, 4)


def aggregate_scores(
    evidence_items: list[dict], registry: dict[str, dict] | None = None
) -> tuple[float, float]:
    """Compute (support_score, refute_score) from a list of evidence item dicts.

    Evidence items with stance 'not_enough_evidence' or 'conflicting_evidence'
    are excluded from both aggregations. Absence (non_apprehension) claims carry
    'supports' stance directly, so they contribute to support score naturally.

    Args:
        evidence_items: List of evidence dicts (schema v3.0).
        registry:       Source trust registry dict {source_id: record}.
                        If None, DEFAULT_SOURCE_TRUST is used for all items.

    Returns:
        (support_score, refute_score) — both in [0, 1].
    """
    from src.epistemic.registry import get_source_trust

    reg = registry or {}
    support_complements: list[float] = []
    refute_complements: list[float] = []

    for ev in evidence_items:
        stance = ev.get("stance", "")
        if stance in (
            EvidenceStance.NOT_ENOUGH_EVIDENCE,
            EvidenceStance.CONFLICTING_EVIDENCE,
        ):
            continue

        source_id = ev.get("source_id", "unknown_web")
        st = get_source_trust(source_id, reg)
        ew = combine_evidence_weights(ev.get("evidence_types", []))
        inference_strength = ev.get("inference_strength", 0.6)
        ec = compute_evidence_confidence(st, ew, inference_strength)

        if stance == EvidenceStance.SUPPORTS:
            support_complements.append(1.0 - ec)
        elif stance == EvidenceStance.REFUTES:
            refute_complements.append(1.0 - ec)

    support_score = _product_complement(support_complements)
    refute_score = _product_complement(refute_complements)
    return round(support_score, 4), round(refute_score, 4)


def derive_verdict(support_score: float, refute_score: float) -> str:
    """Derive verdict label from aggregated support and refute scores.

    Thresholds (ADR-014):
      supported            : support >= 0.75 AND refute < 0.40
      refuted              : refute  >= 0.75 AND support < 0.40
      conflicting_evidence : support >= 0.40 AND refute >= 0.40
      not_enough_evidence  : everything else
    """
    if support_score >= SUPPORT_THRESHOLD and refute_score < CONFLICT_FLOOR:
        return Verdict.SUPPORTED
    if refute_score >= REFUTE_THRESHOLD and support_score < CONFLICT_FLOOR:
        return Verdict.REFUTED
    if support_score >= CONFLICT_FLOOR and refute_score >= CONFLICT_FLOOR:
        return Verdict.CONFLICTING_EVIDENCE
    return Verdict.NOT_ENOUGH_EVIDENCE


def is_training_record(record: dict) -> bool:
    """True if the record contains at least one training EvidenceType (ADR-011).

    In schema v3.0, checks evidence_types_all in the epistemic block.
    Records whose every evidence type is postulation_derivation are excluded.
    """
    evidence_types_all = record.get("epistemic", {}).get("evidence_types_all", [])
    return bool(set(evidence_types_all) & TRAINING_EVIDENCE_TYPES)


def _product_complement(complements: list[float]) -> float:
    """Helper: compute 1 - Π(complements)."""
    if not complements:
        return 0.0
    result = 1.0
    for c in complements:
        result *= c
    return 1.0 - result
