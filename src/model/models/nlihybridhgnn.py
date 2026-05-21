"""NLIHybridHGNN — v3-nli: NLI-enriched graph model with graph-aware stance head.

Evidence node features are 408d — a frozen NLI cross-encoder (DeBERTa-v3-small, MNLI)
contributes 3 semantic columns stored at ev.x[:, -3:]:
  [p_contradiction, p_entailment, p_neutral]

These NLI columns are used as INITIAL FEATURES for the GNN, not as the final stance.
The GNN performs graph-aware contextual reasoning over all evidence items, then H1
StanceHead predicts stance from graph-enriched claim-aware embeddings.

Pipeline:
  NLI probs (offline DeBERTa) → ev.x features (408d = 405d base + 3d NLI)
  → GNN Encoder (graph-aware reasoning over multi-evidence context)
  → cat([ev_emb, claim_emb]) → H1 StanceHead (claim-aware graph stance)
  → cat([ev_emb, claim_emb]) → H2 ISHead (claim-aware inference strength)
  → EC formula → SymbolicAggregator → HybridVerdictHead

Training loss: stance CE (H1) + IS regression (H2) + verdict CE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.model.architecture.aggregator import SymbolicAggregator
from src.model.architecture.encoder import EpistemicEncoder
from src.model.architecture.heads import HybridVerdictHead, ISHead, StanceHead
from src.model.config import GraphConfig
from src.model.data.types import VERDICT_TO_INT, NodeType

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}


class NLIHybridHGNN(nn.Module):
    """v3-nli: NLI-enriched GNN with graph-aware claim-aware H1 StanceHead.

    408d evidence features (405d base + 3d offline DeBERTa NLI probs) feed the GNN.
    H1 StanceHead predicts stance from claim-aware graph-enriched context — not directly from NLI output.

    Args:
        graph_config: Defaults to GraphConfig.v2() (408d evidence nodes).
        hidden_dim:   Encoder output dim (256).
        heads:        GAT attention heads (4).
        dropout:      Dropout (0.1).
        ec_threshold: Symbolic decision threshold θ (0.35).
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
        cfg = graph_config or GraphConfig.v2()
        self.encoder = EpistemicEncoder(cfg, hidden_dim, heads, dropout)
        self.stance_head = StanceHead(hidden_dim * 2)
        self.is_head = ISHead(hidden_dim * 2)
        self.verdict_head = HybridVerdictHead(hidden_dim)
        self.aggregator = SymbolicAggregator()
        self.ec_threshold = ec_threshold

    # ── App / visualisation helpers ───────────────────────────────────────────

    def arc_block_definition(self, inference_out=None):
        from src.model.architecture.arc_block import ArcBlock, CompositeArcBlock
        return CompositeArcBlock(
            blocks=[
                ArcBlock(
                    "Input Features",
                    "Claim 390d · Evidence 408d (405d base + 3d NLI probs from offline DeBERTa-v3-small)",
                    node_id="input_features", color="#dbeafe",
                ),
                self.encoder.arc_block_definition(inference_out),
                self.stance_head.arc_block_definition(inference_out),
                self.is_head.arc_block_definition(inference_out),
                ArcBlock(
                    "EC Formula",
                    "Graph-aware stance probs [sup, ref, nei] → EC",
                    node_id="ec_formula", color="#d1fae5",
                ),
                self.aggregator.arc_block_definition(inference_out),
                self.verdict_head.arc_block_definition(inference_out),
            ],
            title="NLIHybridHGNN",
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

    # ── Forward / inference ───────────────────────────────────────────────────

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns:
            stance_logits  : [N_ev, 3]   — graph-aware stance (H1 on GNN output)
            is_pred        : [N_ev, 1]
            ec_scores      : [N_claims, 3]  — (sup, ref, nei)
            verdict_logits : [N_claims, 3]
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
        ec_scores, verdict_logits = self._soft_verdict_logits(
            stance_logits, is_pred.detach(), claim_emb, data
        )

        return {
            "stance_logits": stance_logits,
            "is_pred": is_pred,
            "ec_scores": ec_scores,
            "verdict_logits": verdict_logits,
        }

    def _soft_verdict_logits(
        self,
        stance_logits: torch.Tensor,
        is_pred: torch.Tensor,
        claim_emb: torch.Tensor,
        data: HeteroData,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EC formula using graph-aware stance probs from H1 StanceHead.

        Returns (ec_scores [N_claims, 3], verdict_logits [N_claims, 3]).
        """
        ev = data[NodeType.EVIDENCE]
        n_claims = data[NodeType.CLAIM].x.shape[0]
        batch_ptr = getattr(ev, "batch", None)

        if batch_ptr is None:
            batch_ptr = torch.zeros(
                is_pred.shape[0], dtype=torch.long, device=is_pred.device
            )

        stance_probs = torch.softmax(stance_logits, dim=-1)  # [N_ev, 3]: [sup, ref, nei]

        scores = torch.zeros(n_claims, 3, device=is_pred.device)
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

        return scores, self.verdict_head(scores, claim_emb)

    @torch.no_grad()
    def predict(self, data: HeteroData) -> dict:
        """Inference: symbolic EC threshold on graph-aware scores; VerdictHead as fallback.

        decision_path is one of:
          "symbolic_supported" — sup > θ, ref ≤ θ  (EC override → supported)
          "symbolic_refuted"   — ref > θ, sup ≤ θ  (EC override → refuted)
          "vh_conflict"        — both > θ           (VerdictHead resolves conflict)
          "vh_fallback"        — neither > θ        (VerdictHead decides)
        """
        out = self.forward(data)

        ec = out["ec_scores"][0]
        sup = float(ec[0])
        ref = float(ec[1])

        if sup > self.ec_threshold and ref > self.ec_threshold:
            verdict = _INT_TO_VERDICT[int(out["verdict_logits"].argmax(dim=-1).item())]
            decision_path = "vh_conflict"
        elif sup > self.ec_threshold:
            verdict = "supported"
            decision_path = "symbolic_supported"
        elif ref > self.ec_threshold:
            verdict = "refuted"
            decision_path = "symbolic_refuted"
        else:
            verdict = _INT_TO_VERDICT[int(out["verdict_logits"].argmax(dim=-1).item())]
            decision_path = "vh_fallback"

        stance_pred = out["stance_logits"].argmax(dim=-1)

        return {
            **out,
            "stance_pred": stance_pred,
            "support_score": sup,
            "refute_score": ref,
            "verdict": verdict,
            "decision_path": decision_path,
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

        verdict_probs = torch.softmax(out["verdict_logits"], dim=-1)[0].tolist()

        ev_data = graph_data[NodeType.EVIDENCE]
        ew_vals = ev_data.ew.tolist()
        st_vals = ev_data.st.tolist()
        is_vals = out["is_pred"].view(-1).tolist()
        nli_raw = ev_data.x[:, -3:].tolist()  # [p_contradiction, p_entailment, p_neutral] — display only

        stance_probs = torch.softmax(out["stance_logits"], dim=-1)  # [N_ev, 3] from H1

        breakdown = []
        for i in range(min(len(resolved_items), len(out["stance_pred"]))):
            s_idx = int(out["stance_pred"][i].item())
            ec    = compute_evidence_confidence(st_vals[i], ew_vals[i], is_vals[i])
            ev    = resolved_items[i]
            text  = ev.get("text", "")
            raw   = nli_raw[i] if i < len(nli_raw) else [0.0, 0.0, 0.0]

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
                "nli_probs": {
                    "contradiction": round(raw[0], 3),
                    "entailment":    round(raw[1], 3),
                    "neutral":       round(raw[2], 3),
                },
            })

        return {
            "verdict":            out["verdict"],
            "verdict_probs":      verdict_probs,
            "support_score":      float(out.get("support_score", 0.0)),
            "refute_score":       float(out.get("refute_score",  0.0)),
            "has_ec":             True,
            "ec_threshold":       self.ec_threshold,
            "evidence_breakdown": breakdown,
            "hetero_data":        graph_data,
        }
