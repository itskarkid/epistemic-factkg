"""SymbolicAggregator — stateless EC formula, no trainable parameters.

EC_i  = 1 - (1 - ST_i)^(EW_i * IS_i)
SupportScore = 1 - prod(1 - EC_i)  over supporters
RefuteScore  = 1 - prod(1 - EC_i)  over refuters

Called only at inference — gradients never flow through here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.model.architecture.arc_block import ArcBlock


_SUPPORT_STANCE = 0  # STANCE_TO_INT["supports"]
_REFUTE_STANCE = 1


class SymbolicAggregator:
    """Stateless: no __init__ parameters, no nn.Module inheritance."""

    def compute_soft_scores(
        self,
        stance_probs: torch.Tensor,  # [N_ev, 3] — softmax from H1 (differentiable)
        is_pred: torch.Tensor,  # [N_ev] or [N_ev,1]
        ew: torch.Tensor,  # [N_ev]
        st: torch.Tensor,  # [N_ev]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable soft symbolic scores for end-to-end training.

        Uses softmax stance probabilities instead of argmax so gradients
        flow back through H1 and H2 into the encoder.

        soft_support = 1 - ∏(1 - EC_i * p_support_i)
        soft_refute  = 1 - ∏(1 - EC_i * p_refute_i)
        soft_nei     = 1 - max(soft_support, soft_refute)
        """
        is_flat = is_pred.view(-1).float()
        ec = 1.0 - (1.0 - st.float()) ** (ew.float() * is_flat)

        p_support = stance_probs[:, 0]
        p_refute = stance_probs[:, 1]

        support_score = 1.0 - torch.prod(1.0 - ec * p_support)
        refute_score = 1.0 - torch.prod(1.0 - ec * p_refute)
        nei_score = 1.0 - torch.clamp(torch.max(support_score, refute_score), 0.0, 1.0)

        return support_score.unsqueeze(0), refute_score.unsqueeze(0), nei_score.unsqueeze(0)

    def compute_scores(
        self,
        stance_pred: torch.Tensor,  # [N_ev] int — argmax of H1 logits
        is_pred: torch.Tensor,  # [N_ev] or [N_ev,1] float from H2
        ew: torch.Tensor,  # [N_ev] float — pre-computed epistemic weight
        st: torch.Tensor,  # [N_ev] float — source trust from registry
    ) -> tuple[float, float, float]:
        """Return (support_score, refute_score, nei_score) ∈ [0.0, 1.0]."""
        is_flat = is_pred.view(-1).float()
        ew_flat = ew.view(-1).float()
        st_flat = st.view(-1).float()

        ec = 1.0 - (1.0 - st_flat) ** (ew_flat * is_flat)

        support_mask = stance_pred == _SUPPORT_STANCE
        refute_mask = stance_pred == _REFUTE_STANCE

        support_score = _aggregate(ec, support_mask)
        refute_score = _aggregate(ec, refute_mask)
        nei_score = 1.0 - max(float(support_score), float(refute_score))

        return float(support_score), float(refute_score), nei_score

    def arc_block_definition(self, inference_out=None) -> "ArcBlock":
        from src.model.architecture.arc_block import ArcBlock
        ann: dict[str, str] = {}
        if inference_out is not None:
            sup = inference_out.get("support_score")
            ref = inference_out.get("refute_score")
            thr = inference_out.get("ec_threshold")
            if sup is not None:
                ann["sup"] = f"{sup:.3f}"
            if ref is not None:
                ann["ref"] = f"{ref:.3f}"
            if thr is not None:
                ann["θ"] = f"{thr:.2f}"
        return ArcBlock(
            name="EC Aggregation",
            detail="SymbolicAggregator · sup/ref/nei scores via EC formula",
            node_id="aggregation",
            color="#d1fae5",
            live_annotations=ann,
        )

    def get_verdict(self, support_score: float, refute_score: float) -> str:
        if support_score >= 0.75 and refute_score < 0.40:
            return "supported"
        if refute_score >= 0.75 and support_score < 0.40:
            return "refuted"
        if support_score >= 0.40 and refute_score >= 0.40:
            return "conflicting_evidence"
        return "not_enough_evidence"


def _aggregate(ec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = ec[mask]
    if selected.numel() == 0:
        return torch.tensor(0.0)
    return 1.0 - torch.prod(1.0 - selected)
