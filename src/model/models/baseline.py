"""BaselineHGNN — ablation model for publication comparison.

Same graph encoder and multi-task supervision as EpistemicHGNN, but the
verdict is predicted directly from claim node embeddings after message
passing — bypassing the Pramana EC formula and symbolic aggregation.

Ablation story (for paper):
  EpistemicHGNN:  evidence → IS → EC formula → SymbolicAggregator → VerdictHead
  BaselineHGNN:   evidence → HeteroConv → claim node → MLP → verdict

Everything upstream of verdict is identical: same graph structure, same
HeteroConv encoder, same StanceHead (H1), same ISHead (H2).  The only
difference is the verdict pathway — no EC weighting, no symbolic scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.model.architecture.encoder import EpistemicEncoder
from src.model.architecture.heads import ISHead, StanceHead
from src.model.config import GraphConfig
from src.model.data.types import NUM_VERDICT, VERDICT_TO_INT, NodeType

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}


class BaselineHGNN(nn.Module):
    """Ablation baseline: HeteroConv encoder + direct claim-node verdict.

    Args:
        graph_config:  Node dims + edge types (use GraphConfig.v1()).
        hidden_dim:    Encoder output dim; shared with stance and IS heads.
        heads:         GAT attention heads in encoder.
        dropout:       Encoder inter-layer and verdict MLP dropout.
    """


    def __init__(
        self,
        graph_config: GraphConfig | None = None,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        ec_threshold: float = 0.35,  # unused — baseline has no EC path
    ) -> None:
        super().__init__()
        cfg = graph_config or GraphConfig.v1()
        self.encoder = EpistemicEncoder(cfg, hidden_dim, heads, dropout)
        self.stance_head = StanceHead(hidden_dim)
        self.is_head = ISHead(hidden_dim)
        # Verdict from claim node only — no EC formula, no symbolic aggregation
        self.verdict_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_VERDICT),
        )

    def arc_block_definition(self, inference_out=None):
        from src.model.architecture.arc_block import ArcBlock, CompositeArcBlock
        ann: dict[str, str] = {}
        if inference_out is not None and inference_out.get("verdict"):
            ann["verdict"] = inference_out["verdict"]
            vp = inference_out.get("verdict_probs")
            if vp:
                ann["probs"] = "[" + ", ".join(f"{p:.2f}" for p in vp) + "]"
        return CompositeArcBlock(
            blocks=[
                ArcBlock("Input Features", "Claim 390d · Evidence 400d", node_id="input_features", color="#dbeafe"),
                self.encoder.arc_block_definition(inference_out),
                self.stance_head.arc_block_definition(inference_out),
                self.is_head.arc_block_definition(inference_out),
                ArcBlock("Verdict MLP", "claim_emb 256d → Linear(256→128) → ReLU → Linear(128→3) · no EC", node_id="verdict_mlp", color="#fce7f3", live_annotations=ann),
            ],
            title="BaselineHGNN",
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
            verdict_logits : [N_claims, 3]  — from claim node MLP, no EC
        """
        x_dict = self.encoder(data)
        ev_emb = x_dict[NodeType.EVIDENCE]  # [N_ev, hidden_dim]
        claim_emb = x_dict[NodeType.CLAIM]  # [N_claims, hidden_dim]

        return {
            "stance_logits": self.stance_head(ev_emb),
            "is_pred": self.is_head(ev_emb),
            "verdict_logits": self.verdict_mlp(claim_emb),
        }

    @torch.no_grad()
    def predict(self, data: HeteroData) -> dict:
        """Inference — no symbolic scores (baseline has no EC formula)."""
        out = self.forward(data)
        stance_pred = out["stance_logits"].argmax(dim=-1)
        verdict_idx = out["verdict_logits"].argmax(dim=-1).item()
        verdict = _INT_TO_VERDICT.get(int(verdict_idx), "not_enough_evidence")
        return {
            **out,
            "stance_pred": stance_pred,
            "verdict": verdict,
        }

    def build_prediction_payload(
        self,
        out: dict,
        graph_data: HeteroData,
        resolved_items: list[dict],
    ) -> dict:
        """Convert raw predict() output to app-level payload dict (no EC)."""
        from src.epistemic.formula import compute_evidence_confidence
        from src.model.data.types import STANCE_TO_INT, NodeType

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
                "stance_confidence": round(float(stance_probs[i, s_idx].item()), 3),
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
            "support_score":      0.0,
            "refute_score":       0.0,
            "has_ec":             False,
            "ec_threshold":       None,
            "evidence_breakdown": breakdown,
            "hetero_data":        graph_data,
        }
