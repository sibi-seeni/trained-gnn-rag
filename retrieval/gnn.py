"""Use a trained checkpoint as a retriever: same interface idea as retrieval/dense.rank."""
import torch

from model.gnn import build_model


class GNNRanker:
    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = build_model(ck["cfg"], ck["in_dim"], ck["edge_dim"]).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def scores(self, data):
        return self.model(data.to(self.device)).cpu()

    def rank(self, data):
        """Passage indices, best first (question node, if any, is excluded)."""
        s = self.scores(data)
        s[~data.score_mask.cpu()] = float("-inf")
        return torch.argsort(s, descending=True)[: int(data.score_mask.sum())].tolist()
