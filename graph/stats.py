from torch_geometric.utils import degree
def stats(graphs):
    deg = iso = gold_linked = 0
    for g in graphs:
        d = degree(g.edge_index[0], g.num_nodes)
        deg += d.mean().item(); iso += (d == 0).sum().item()
        gi = g.y.nonzero().flatten().tolist()
        if len(gi) == 2:
            e = set(map(tuple, g.edge_index.t().tolist()))
            gold_linked += (gi[0], gi[1]) in e or (gi[1], gi[0]) in e
    n = len(graphs)
    return {"avg_degree": deg/n, "isolated_per_graph": iso/n, "gold_pair_linked": gold_linked/n}