"""EpistemicHGNN — neuro-symbolic fact-verification model (ADR-013, ADR-014).

Architecture:
  EpistemicEncoder  (shared HeteroConv, config-driven)
      ↓ cat([ev_emb, claim_emb]) [N_ev, 2*hidden_dim]  — claim-aware context
  StanceHead   H1  → stance logits  [N_ev, 3]
  ISHead       H2  → IS scalars     [N_ev, 1]
      ↓ differentiable soft EC aggregation (per claim)
  VerdictHead      → verdict logits [N_claims, 3]

Training loss:
  stance_CE + λ₁ * IS_MSE + λ₂ * verdict_CE
  Gradients flow through H1 (soft stance probs) and H2 (IS) into the encoder.

At inference: hard argmax stance → symbolic EC scores → VerdictHead → verdict string.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.model.architecture.aggregator import SymbolicAggregator
from src.model.config import GraphConfig
from src.model.architecture.encoder import EpistemicEncoder
from src.model.architecture.heads import ISHead, StanceHead, VerdictHead
from src.model.data.types import VERDICT_TO_INT, NodeType

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}


class EpistemicHGNN(nn.Module):
    """
    Args:
        graph_config:  Node dims + edge types (use GraphConfig.v1()).
        hidden_dim:    Encoder output dim; heads read this dim.
        heads:         GAT attention heads in encoder.
        dropout:       Encoder inter-layer dropout.
    """


    def __init__(
        self,
        graph_config: GraphConfig | None = None,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        ec_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        cfg = graph_config or GraphConfig.v1()
        self.encoder = EpistemicEncoder(cfg, hidden_dim, heads, dropout)
        self.stance_head = StanceHead(hidden_dim * 2)
        self.is_head = ISHead(hidden_dim * 2)
        self.verdict_head = VerdictHead()
        self.aggregator = SymbolicAggregator()
        self.ec_threshold = ec_threshold

    def arc_block_definition(self, inference_out=None):
        from src.model.architecture.arc_block import ArcBlock, CompositeArcBlock
        return CompositeArcBlock(
            blocks=[
                ArcBlock("Input Features", "Claim 390d · Evidence 405d", node_id="input_features", color="#dbeafe"),
                self.encoder.arc_block_definition(inference_out),
                self.stance_head.arc_block_definition(inference_out),
                self.is_head.arc_block_definition(inference_out),
                ArcBlock("EC Formula", "EC = 1−(1−ST)^(EW×IS) · per-evidence score", node_id="ec_formula", color="#d1fae5"),
                self.aggregator.arc_block_definition(inference_out),
                self.verdict_head.arc_block_definition(inference_out),
            ],
            title="EpistemicHGNN",
        )

    def result_dot(self, result: dict) -> str:
        from src.model.architecture.arc_block import result_dot
        return result_dot(result)

    def decision_path_info(self, result: dict) -> dict:
        from src.model.architecture.arc_block import decision_path_info
        return decision_path_info(result)

    def evidence_table(self, result: dict) -> list[dict]:
        from src.model.architecture.arc_block import evidence_table
        return evidence_table(result)

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns:
            stance_logits  : [N_ev, 3]
            is_pred        : [N_ev, 1]
            ec_scores      : [N_claims, 3]  — (sup, ref, nei) soft EC scores
            verdict_logits : [N_claims, 3]  — from soft symbolic scores
        """
        x_dict = self.encoder(data)
        ev_emb    = x_dict[NodeType.EVIDENCE]  # [N_ev, hidden_dim]
        claim_emb = x_dict[NodeType.CLAIM]     # [N_claims, hidden_dim]

        batch_ptr = getattr(data[NodeType.EVIDENCE], "batch", None)
        if batch_ptr is None:
            batch_ptr = torch.zeros(ev_emb.shape[0], dtype=torch.long, device=ev_emb.device)
        ev_ctx = torch.cat([ev_emb, claim_emb[batch_ptr]], dim=-1)  # [N_ev, 2*hidden_dim]

        stance_logits = self.stance_head(ev_ctx)  # [N_ev, 3]
        is_pred       = self.is_head(ev_ctx)       # [N_ev, 1]

        # Soft symbolic scores — differentiable (uses softmax probs, not argmax)
        ec_scores, verdict_logits = self._soft_verdict_logits(data, stance_logits, is_pred.detach())

        return {
            "stance_logits": stance_logits,
            "is_pred": is_pred,
            "ec_scores": ec_scores,
            "verdict_logits": verdict_logits,
        }

    @torch.no_grad()
    def predict(self, data: HeteroData) -> dict:
        """Full neuro-symbolic inference for a single graph.

        Uses soft EC scores from forward (same path as training) for thresholds,
        eliminating the train-inference gap from hard-argmax EC computation.
        """
        _EC_DECISIVE = self.ec_threshold

        out = self.forward(data)
        stance_pred = out["stance_logits"].argmax(dim=-1)

        ec = out["ec_scores"][0]  # [3] — (sup, ref, nei) for single claim
        sup = float(ec[0])
        ref = float(ec[1])

        vh_pred = _INT_TO_VERDICT.get(int(out["verdict_logits"].argmax(dim=-1).item()), "not_enough_evidence")

        if sup > _EC_DECISIVE and ref > _EC_DECISIVE:
            verdict = vh_pred
            decision_path = "vh_conflict"
            ec_decision   = "conflicted"
            final_layer   = "verdicthead"
        elif sup > _EC_DECISIVE:
            verdict = "supported"
            decision_path = "symbolic_supported"
            ec_decision   = "supported"
            final_layer   = "ec_symbolic"
        elif ref > _EC_DECISIVE:
            verdict = "refuted"
            decision_path = "symbolic_refuted"
            ec_decision   = "refuted"
            final_layer   = "ec_symbolic"
        else:
            verdict = vh_pred
            decision_path = "vh_fallback"
            ec_decision   = "deferred"
            final_layer   = "verdicthead"

        return {
            **out,
            "stance_pred":   stance_pred,
            "support_score": sup,
            "refute_score":  ref,
            "verdict":       verdict,
            "decision_path": decision_path,
            "ec_decision":   ec_decision,
            "final_layer":   final_layer,
            "vh_pred":       vh_pred,
        }

    def build_prediction_payload(
        self,
        out: dict,
        graph_data: HeteroData,
        resolved_items: list[dict],
    ) -> dict:
        """Convert raw predict() output to app-level payload dict."""
        from src.epistemic.formula import compute_evidence_confidence
        from src.model.data.types import STANCE_TO_INT

        _int_to_stance = {}
        for k, v in STANCE_TO_INT.items():
            if v not in _int_to_stance:
                _int_to_stance[v] = k

        stance_probs  = torch.softmax(out["stance_logits"], dim=-1)
        verdict_probs = torch.softmax(out["verdict_logits"], dim=-1)[0].tolist()

        ev_data = graph_data[NodeType.EVIDENCE]
        ew_vals = ev_data.ew.tolist()
        st_vals = ev_data.st.tolist()
        is_vals = out["is_pred"].view(-1).tolist()

        breakdown = []
        for i in range(min(len(resolved_items), len(out["stance_pred"]))):
            s_idx = int(out["stance_pred"][i].item())
            ec    = compute_evidence_confidence(st_vals[i], ew_vals[i], is_vals[i])
            ev    = resolved_items[i]
            text  = ev.get("text", "")
            breakdown.append({
                "text":              text,
                "text_short":        (text[:150] + "…") if len(text) > 150 else text,
                "modality":          ev.get("modality", "web_text"),
                "source_type":       ev.get("source_type", "unknown"),
                "stance":            _int_to_stance.get(s_idx, "not_enough_evidence"),
                "support_confidence": round(float(stance_probs[i, 0].item()), 3),
                "refute_confidence":  round(float(stance_probs[i, 1].item()), 3),
                "is_score":          round(is_vals[i], 3),
                "source_trust":      round(st_vals[i], 3),
                "evidence_weight":   round(ew_vals[i], 3),
                "ec_score":          round(ec, 3),
                "source_id":         ev.get("source_id", ""),
                "nli_probs":         None,
            })

        return {
            "verdict":            out["verdict"],
            "verdict_probs":      verdict_probs,
            "support_score":      float(out.get("support_score", 0.0)),
            "refute_score":       float(out.get("refute_score",  0.0)),
            "has_ec":             True,
            "ec_threshold":       self.ec_threshold,
            "decision_path":      out.get("decision_path", "vh_fallback"),
            "ec_decision":        out.get("ec_decision",   "deferred"),
            "final_layer":        out.get("final_layer",   "verdicthead"),
            "vh_pred":            out.get("vh_pred"),
            "evidence_breakdown": breakdown,
            "hetero_data":        graph_data,
        }

    # ------------------------------------------------------------------

    def _soft_verdict_logits(
        self,
        data: HeteroData,
        stance_logits: torch.Tensor,
        is_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-claim soft symbolic scores and map through VerdictHead.

        Returns (ec_scores [N_claims, 3], verdict_logits [N_claims, 3]).
        Uses the batch pointer to group evidence by claim; falls back to a
        single claim when batch pointer is absent (single-graph inference).
        """
        ev = data[NodeType.EVIDENCE]
        batch_ptr = getattr(ev, "batch", None)
        n_claims = data[NodeType.CLAIM].x.shape[0]

        if batch_ptr is None:
            batch_ptr = torch.zeros(
                stance_logits.shape[0], dtype=torch.long, device=stance_logits.device
            )

        stance_probs = torch.softmax(stance_logits, dim=-1)
        scores = torch.zeros(n_claims, 3, device=stance_logits.device)

        for c in range(n_claims):
            mask = batch_ptr == c
            sup, ref, nei = self.aggregator.compute_soft_scores(
                stance_probs[mask],
                is_pred[mask],
                ew=ev.ew[mask],
                st=ev.st[mask],
            )
            scores[c, 0] = sup
            scores[c, 1] = ref
            scores[c, 2] = nei

        return scores, self.verdict_head(scores)  # [n_claims, 3], [n_claims, 3]
