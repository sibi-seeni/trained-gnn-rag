"""Turn the saved Week 1 passage graphs into question-conditioned GNN inputs.

The Week 1 graphs only hold passage vectors (x) and edge-type flags. Relevance depends on
the question, so every passage node gets
    [p, q, p*q, cos(p, q)]      (384*3 + 1 = 1153 dims with MiniLM)
On top of that this module
  - makes the graph undirected (NOTES open issue: title edges were one-directional),
  - marks which gold passage is the "second fact" (hop2), see `hop2_index`,
  - optionally adds a question node linked to every passage (helps comparison questions),
  - keeps only the edge-attribute columns a config asks for.

Edge-attribute columns after preparation (index -> meaning):
    0 shared_entity   1 title_mention   2 sim   3 agree (Week 3)   4 contradict (Week 3)   5 question_link
"""
import json

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from retrieval.embed import encode_cached

EDGE_COLS = ["shared_entity", "title", "sim", "agree", "contradict", "qlink"]
QTYPE = {"bridge": 0, "comparison": 1}


def hop2_index(ex, cos):
    """Pick the gold passage that is the 'second fact' for this question.

    Rule 1 (title): in a bridge question one gold usually names the other's title,
    e.g. passage A says "...directed by Christopher Nolan" and passage B is titled
    "Christopher Nolan". The named one (B) is the second hop.
    Rule 2 (cos): otherwise, the gold that is less similar to the question under the
    dense retriever (the one plain search struggles with).
    Returns (index or -1, rule).
    """
    gold = [j for j, p in enumerate(ex["passages"]) if p["gold"]]
    if len(gold) != 2:
        return -1, "none"
    a, b = gold
    pa, pb = ex["passages"][a], ex["passages"][b]
    a_names_b = pb["title"].lower() in pa["text"].lower()
    b_names_a = pa["title"].lower() in pb["text"].lower()
    if a_names_b and not b_names_a:
        return b, "title"
    if b_names_a and not a_names_b:
        return a, "title"
    return (a if cos[a] < cos[b] else b), "cos"


def prepare_one(g, ex, q_vec, question_node=False, edge_cols=(0, 1)):
    n = g.num_nodes
    p = g.x.float()
    q = torch.as_tensor(q_vec, dtype=torch.float)
    cos = p @ q
    x = torch.cat([p, q.expand_as(p), p * q, cos[:, None]], dim=1)

    ea = g.edge_attr.float() if g.edge_attr is not None else torch.zeros(g.edge_index.size(1), 5)
    ei, ea = to_undirected(g.edge_index, ea, num_nodes=n, reduce="max")
    ea = torch.cat([ea, torch.zeros(ea.size(0), 1)], dim=1)  # col 5: question link

    y = g.y.float().clone()
    score_mask = torch.ones(n, dtype=torch.bool)
    if question_node:
        qx = torch.cat([q, q, q * q, torch.ones(1)])[None]
        x = torch.cat([x, qx])
        idx, qi = torch.arange(n), torch.full((n,), n)
        q_ei = torch.cat([torch.stack([idx, qi]), torch.stack([qi, idx])], dim=1)
        q_ea = torch.zeros(2 * n, ea.size(1)); q_ea[:, 5] = 1.0
        ei, ea = torch.cat([ei, q_ei], 1), torch.cat([ea, q_ea])
        y = torch.cat([y, torch.zeros(1)])
        cos = torch.cat([cos, torch.zeros(1)])
        score_mask = torch.cat([score_mask, torch.zeros(1, dtype=torch.bool)])

    h2, rule = hop2_index(ex, cos[:n].numpy())
    hop2 = torch.zeros(len(y), dtype=torch.bool)
    if h2 >= 0:
        hop2[h2] = True

    ea = ea[:, list(edge_cols)] if len(edge_cols) else None
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y, cos=cos, score_mask=score_mask, hop2=hop2,
                qtype=torch.tensor([QTYPE.get(ex["type"], -1)]), hop2_rule=rule, qid=ex["qid"])


def load_split(cfg, split):
    """Load examples + saved graphs for a split and return a list of prepared Data objects."""
    out_dir, E = cfg["data"]["out_dir"], cfg["embed"]
    G = cfg.get("graph", {})
    exs = [json.loads(l) for l in open(f"{out_dir}/{split}.jsonl")]
    graphs = torch.load(f"{out_dir}/" + G.get("file", "graphs_{split}.pt").format(split=split), weights_only=False)
    assert len(graphs) == len(exs), f"{split}: {len(graphs)} graphs vs {len(exs)} examples"
    qv = encode_cached([e["question"] for e in exs], E["model"], E["cache_dir"], f"q_{split}_all")
    data = []
    for g, e, q in zip(graphs, exs, qv):
        assert g.qid == e["qid"], "graph/example order mismatch"
        data.append(prepare_one(g, e, q, G.get("question_node", False), G.get("edge_cols", [0, 1])))
    return data


def summarize(data):
    rules = [d.hop2_rule for d in data]
    return {r: rules.count(r) for r in ("title", "cos", "none")}


if __name__ == "__main__":  # quick check: python -m graph.dataset configs/gnn_relevance.yaml
    import sys
    from common.config import load_cfg
    cfg = load_cfg(sys.argv[1])
    for split in ("train", "val"):
        d = load_split(cfg, split)
        print(split, len(d), "x:", tuple(d[0].x.shape), "edge_attr:",
              None if d[0].edge_attr is None else tuple(d[0].edge_attr.shape),
              "hop2 rule counts:", summarize(d),
              "avg undirected edges:", float(np.mean([x.edge_index.size(1) for x in d])))
