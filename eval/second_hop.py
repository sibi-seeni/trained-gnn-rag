"""Retrieval metrics focused on the second fact.

A "record" is one question's ranking:
    {"qid", "qtype", "order": [node idx, best first], "flags": [gold?, in rank order],
     "hop2_rank": 1-based rank of the second-fact passage (None if undefined)}

Metrics:
    hop2@k        second-fact passage is in the top k                  <- main Week 2 number
    hop2_mrr      mean 1/rank of the second-fact passage
    hop2_rank     mean rank of the second-fact passage (lower is better)
    recall@k, both_gold@k, mrr   same definitions as Week 1 (eval/retrieval_metrics.py)
Every metric is also reported per question type with suffixes _bridge / _comparison.
"""
import numpy as np
import torch

from eval.retrieval_metrics import retrieval_metrics

TYPES = {0: "bridge", 1: "comparison"}


def make_record(d, scores):
    """d: one prepared Data object; scores: 1-D tensor over its nodes."""
    s = scores.detach().cpu().float().clone()
    s[~d.score_mask.cpu()] = float("-inf")
    order = torch.argsort(s, descending=True)[: int(d.score_mask.sum())].tolist()
    y, hop2 = d.y.cpu().bool().tolist(), d.hop2.cpu().tolist()
    h2 = next((r + 1 for r, j in enumerate(order) if hop2[j]), None)
    return {"qid": d.qid, "qtype": TYPES.get(int(d.qtype), "other"), "order": order,
            "flags": [y[j] for j in order], "hop2_rank": h2}


def dense_records(data):
    """Plain similarity search on the same embeddings (cos column), as records."""
    return [make_record(d, d.cos) for d in data]


def _core(recs, ks):
    out = retrieval_metrics([r["flags"] for r in recs], ks)
    h = [r["hop2_rank"] for r in recs if r["hop2_rank"] is not None]
    for k in ks:
        out[f"hop2@{k}"] = float(np.mean([x <= k for x in h])) if h else float("nan")
    out["hop2_mrr"] = float(np.mean([1 / x for x in h])) if h else float("nan")
    out["hop2_rank"] = float(np.mean(h)) if h else float("nan")
    out["n"] = len(recs)
    return out


def metrics(recs, ks=(2, 4, 10)):
    out = _core(recs, ks)
    for t in ("bridge", "comparison"):
        sub = [r for r in recs if r["qtype"] == t]
        if sub:
            out.update({f"{k}_{t}": v for k, v in _core(sub, ks).items()})
    return out


def hits(recs, k=4):
    """Per-question 0/1 vector for hop2@k, keyed by qid (for paired comparisons)."""
    return {r["qid"]: float(r["hop2_rank"] is not None and r["hop2_rank"] <= k)
            for r in recs if r["hop2_rank"] is not None}
