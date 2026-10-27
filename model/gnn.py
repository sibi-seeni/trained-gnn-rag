"""Relevance-only GATv2 ranker.

    node features -> Linear -> [GATv2Conv -> LayerNorm -> ELU -> Dropout] x L -> head([h_L, h_0]) -> score

- `layers: 0` turns it into a plain MLP on the same node features (no message passing).
  That's the control that tells you whether gains come from the graph or just from
  training a reranker on HotpotQA.
- The head sees h_0 (the node before message passing) as well as h_L, so the model can
  always fall back to "this passage alone looks relevant".
- `encode()` returns node embeddings and (optionally) attention weights, so the Week 3
  trust stage and output heads can sit on top without rewriting this file.
"""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class RelevanceGNN(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=2, heads=4, edge_dim=None, dropout=0.2, attn_dropout=0.1):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ELU(), nn.Dropout(dropout))
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=edge_dim,
                      dropout=attn_dropout, add_self_loops=True, fill_value="mean", residual=True)
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.act, self.drop = nn.ELU(), nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def encode(self, data, return_attention=False):
        h0 = self.inp(data.x)
        h, atts = h0, []
        ea = getattr(data, "edge_attr", None)
        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                out, (ei, a) = conv(h, data.edge_index, ea, return_attention_weights=True)
                atts.append((ei, a))
            else:
                out = conv(h, data.edge_index, ea)
            h = self.drop(self.act(norm(out)))
        return torch.cat([h, h0], dim=-1), atts

    def forward(self, data):
        z, _ = self.encode(data)
        return self.head(z).squeeze(-1)


def build_model(cfg, in_dim, edge_dim):
    M = cfg["model"]
    return RelevanceGNN(in_dim, hidden=M.get("hidden", 256), layers=M.get("layers", 2), heads=M.get("heads", 4),
                        edge_dim=edge_dim if M.get("layers", 2) > 0 else None,
                        dropout=M.get("dropout", 0.2), attn_dropout=M.get("attn_dropout", 0.1))
