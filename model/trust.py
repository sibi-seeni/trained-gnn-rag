"""Week 3 models: the trust stage (for version B) and trust inside message passing (version C).

Version A is model/gnn.RelevanceGNN (Week 2). Everything here uses the same backbone size
(hidden 256, 2 GATv2 layers, 4 heads) and the same node features; the trust stage and C also see the
agree / contradict edge columns.

B  trust stage:  a RelevanceGNN-shaped network trained with BCE to say "real" (1) vs "planted" (0).
                 Its output enters only at final ranking:  final = relevance_A - lambda * (1 - trust).
C  TrustMPGNN:   before every GATv2 layer a small gate reads the current node states and outputs a trust
                 t_j in (0, 1); every message sent by node j in that layer (self-loop included) is multiplied
                 by t_j, so an untrusted passage can't push its content into its neighbours. Two output heads on
                 [h_L, h_0]: relevance (the ranking score) and trust. The gates and the trust head are all
                 trained with the same BCE target as the trust stage.
"""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from model.gnn import RelevanceGNN, build_model


class TrustGATv2Conv(GATv2Conv):
    """GATv2Conv whose messages are scaled by the sender's trust (set per call via `trust=`)."""

    def forward(self, x, edge_index, edge_attr=None, trust=None):
        self._trust = trust
        try:
            return super().forward(x, edge_index, edge_attr)
        finally:
            self._trust = None

    def message(self, x_j, alpha, edge_index_j):
        m = x_j * alpha.unsqueeze(-1)
        if getattr(self, "_trust", None) is not None:
            m = m * self._trust[edge_index_j].view(-1, 1, 1)
        return m


class TrustMPGNN(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=2, heads=4, edge_dim=None, dropout=0.2, attn_dropout=0.1):
        super().__init__()
        assert hidden % heads == 0
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ELU(), nn.Dropout(dropout))
        self.convs = nn.ModuleList([
            TrustGATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=edge_dim,
                           dropout=attn_dropout, add_self_loops=True, fill_value="mean", residual=True)
            for _ in range(layers)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, 1))
                                    for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.act, self.drop = nn.ELU(), nn.Dropout(dropout)

        def head():
            return nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.head, self.trust_head = head(), head()

    def forward_all(self, data):
        """Returns (relevance scores, trust logit of the trust head, [gate logits per layer])."""
        h0 = self.inp(data.x)
        h, gate_logits = h0, []
        ea = getattr(data, "edge_attr", None)
        for conv, norm, gate in zip(self.convs, self.norms, self.gates):
            g = gate(h).squeeze(-1)
            gate_logits.append(g)
            h = self.drop(self.act(norm(conv(h, data.edge_index, ea, trust=torch.sigmoid(g)))))
        z = torch.cat([h, h0], dim=-1)
        return self.head(z).squeeze(-1), self.trust_head(z).squeeze(-1), gate_logits

    def forward(self, data):
        return self.forward_all(data)[0]

    def trust_logit(self, data):
        return self.forward_all(data)[1]


def build(cfg, in_dim, edge_dim):
    M = cfg["model"]
    kind = M.get("kind", "relevance")
    if kind == "trust_mp":
        return TrustMPGNN(in_dim, hidden=M.get("hidden", 256), layers=M.get("layers", 2), heads=M.get("heads", 4),
                          edge_dim=edge_dim, dropout=M.get("dropout", 0.2), attn_dropout=M.get("attn_dropout", 0.1))
    return build_model(cfg, in_dim, edge_dim)  # relevance, relevance_adv, trust_stage: same RelevanceGNN shape


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["cfg"], ck["in_dim"], ck["edge_dim"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["cfg"]
