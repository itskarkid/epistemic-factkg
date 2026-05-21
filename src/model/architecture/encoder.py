"""EpistemicEncoder — shared HeteroConv encoder built from GraphConfig.

Reads node dims and edge types from GraphConfig at construction time.
Never hardcodes node or edge names — adding a new node/edge type to the
config is sufficient; no code change needed here.

Output: context-enriched evidence embeddings [N_ev, hidden_dim].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

from src.model.config import GraphConfig

if TYPE_CHECKING:
    from src.model.architecture.arc_block import ArcBlock


class EpistemicEncoder(nn.Module):
    """Two-layer HeteroConv encoder.

    Args:
        graph_config: V1 (or later) graph schema.
        hidden_dim:   Output embedding dim for all node types.
        heads:        Number of GAT attention heads.
        dropout:      Dropout applied to inter-layer activations.
    """

    def __init__(
        self,
        graph_config: GraphConfig,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_config = graph_config
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        # Project each node type from its input dim to hidden_dim
        self.input_proj = nn.ModuleDict(
            {
                ntype: nn.Linear(dim, hidden_dim)
                for ntype, dim in graph_config.node_dims.items()
            }
        )

        # Two-layer HeteroConv — built dynamically from config
        self.conv1 = self._make_conv(hidden_dim, heads)
        self.conv2 = self._make_conv(hidden_dim, heads)
        self.act = nn.ELU()

    def _make_conv(self, hidden_dim: int, heads: int) -> HeteroConv:
        convs: dict = {}
        for src, rel, dst in self.graph_config.edge_types:
            convs[(src, rel, dst)] = GATConv(
                (-1, -1),
                hidden_dim // heads,
                heads=heads,
                concat=True,
                add_self_loops=False,
            )
        return HeteroConv(convs, aggr="sum")

    def arc_block_definition(self, inference_out=None) -> "ArcBlock":  # noqa: ARG002
        from src.model.architecture.arc_block import ArcBlock
        n_edge_types = len(self.graph_config.edge_types)
        return ArcBlock(
            name="GNN Encoder",
            detail=f"HeteroConv · {self.heads} GAT heads · 2 layers · {n_edge_types} edge types → {self.hidden_dim}d",
            node_id="gnn_encoder",
            color="#e0f2fe",
        )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Run two-layer message passing.

        Returns:
            Dict mapping node type → context-enriched embedding tensor.
            Callers typically only use data[NodeType.EVIDENCE].
        """
        # Project all node types to hidden_dim
        x_dict: dict[str, torch.Tensor] = {}
        for ntype, proj in self.input_proj.items():
            if ntype in data.node_types and data[ntype].x is not None:
                x_dict[ntype] = self.act(proj(data[ntype].x))

        edge_index_dict = {
            (src, rel, dst): data[src, rel, dst].edge_index
            for (src, rel, dst) in self.graph_config.edge_types
            if (src, rel, dst) in data.edge_types
        }

        # Layer 1 with residual skip (preserves projected node features, incl. claim text)
        x0 = x_dict
        x1 = self.conv1(x0, edge_index_dict)
        x1 = {k: self.act(self.dropout(v)) for k, v in x1.items()}
        x1 = {k: x1[k] + x0[k] if k in x0 else x1[k] for k in x1}

        # Layer 2 with residual skip
        x2 = self.conv2(x1, edge_index_dict)
        x2 = {k: self.act(v) for k, v in x2.items()}
        x2 = {k: x2[k] + x1[k] if k in x1 else x2[k] for k in x2}

        return x2
