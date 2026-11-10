"""Train the Week 3 models on clean + attacked train graphs.

Usage (from repo root, after attacks/plant.py has written train_none and train_edit):
    python -m model.train_trust configs/trust_stage.yaml   --seed 0   # B's trust stage
    python -m model.train_trust configs/gnn_trust_mp.yaml  --seed 0   # C
    python -m model.train_trust configs/gnn_relevance_adv.yaml --seed 0   # exploratory control (see NOTES)

Training data is the same for all three: the 1800 fit questions twice, once clean (train_none) and once with the
`edit` fakes planted (train_edit, fake seed 8; val fakes use seed 7). Dev = the same 200 questions as Week 2, also
clean + attacked. Val is never touched here.

Losses (fakes are never gold, so they are negatives for relevance):
    trust_stage     BCE(trust, real)                                   select on dev fake-detection AUC
    trust_mp (C)    listwise(relevance) + w * BCE on trust head + gates  select on mean(dev clean, dev attacked) hop2@4
    relevance_adv   listwise(relevance)                                select on mean(dev clean, dev attacked) hop2@4

Outputs: checkpoints/{name}_s{seed}.pt, results/week3/logs/{name}_s{seed}.csv, one row in results/week3.jsonl.
"""
import argparse, copy, csv, json, os, time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from common.config import load_cfg
from eval.second_hop import make_record, metrics
from graph.dataset import load_split
from model.train import listwise_loss
from model.trust import build


def auc(scores, labels):
    """ROC AUC with ties counted half (labels: 1 = positive)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def trust_bce(logit, data):
    m = data.score_mask
    return F.binary_cross_entropy_with_logits(logit[m], (~data.fake[m]).float())


@torch.no_grad()
def node_scores(model, data, kind, device, bs=256):
    """Per-graph (relevance, trust_logit) tensors; either can be None."""
    model.eval()
    out = []
    for batch in DataLoader(data, batch_size=bs, shuffle=False):
        batch = batch.to(device)
        if kind == "trust_mp":
            rel, tr, _ = model.forward_all(batch)
        elif kind == "trust_stage":
            rel, tr = None, model(batch)
        else:
            rel, tr = model(batch), None
        for i in range(batch.num_graphs):
            sl = slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))
            out.append((None if rel is None else rel[sl].cpu(), None if tr is None else tr[sl].cpu()))
    return out


def dev_metric(model, kind, dev_clean, dev_att, device):
    if kind == "trust_stage":
        sc = node_scores(model, dev_clean + dev_att, kind, device)
        s = torch.cat([t for _, t in sc]).numpy()
        y = torch.cat([(~d.fake).float() for d in dev_clean + dev_att]).numpy()
        return {"sel": auc(s, y), "auc": auc(s, y)}
    res = {}
    for tag, ds in (("clean", dev_clean), ("att", dev_att)):
        sc = node_scores(model, ds, kind, device)
        res[tag] = metrics([make_record(d, r) for d, (r, _) in zip(ds, sc)])
    return {"sel": (res["clean"]["hop2@4"] + res["att"]["hop2@4"]) / 2, "clean_hop2@4": res["clean"]["hop2@4"],
            "att_hop2@4": res["att"]["hop2@4"], "att_poison@4": res["att"]["poison@4"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    T, name, kind = cfg["train"], cfg["name"], cfg["model"]["kind"]
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if a.device == "auto" else a.device))

    clean = load_split(cfg, "train", "none")
    att = load_split(cfg, "train", cfg["attack"]["train"])
    assert [d.qid for d in clean] == [d.qid for d in att]
    perm = np.random.RandomState(cfg["seed"]).permutation(len(clean))  # same dev split as Week 2
    n_dev = int(T.get("dev_frac", 0.1) * len(clean))
    dev_idx, fit_idx = perm[:n_dev], perm[n_dev:]
    fit_set = [clean[i] for i in fit_idx] + [att[i] for i in fit_idx]
    dev_clean, dev_att = [clean[i] for i in dev_idx], [att[i] for i in dev_idx]

    seed_everything(a.seed)
    in_dim, edge_dim = fit_set[0].x.size(1), fit_set[0].edge_attr.size(1)
    model = build(cfg, in_dim, edge_dim).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=T.get("lr", 1e-3), weight_decay=T.get("weight_decay", 1e-4))
    loader = DataLoader(fit_set, batch_size=T.get("batch_size", 64), shuffle=True)
    w_trust, patience = T.get("trust_weight", 1.0), T.get("patience", 12)
    print(f"[{name} s{a.seed}] kind={kind} device={dev} fit={len(fit_set)} dev={len(dev_clean)}+{len(dev_att)} "
          f"in_dim={in_dim} edge_dim={edge_dim} params={n_params:,}")

    os.makedirs("results/week3/logs", exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
    log = open(f"results/week3/logs/{name}_s{a.seed}.csv", "w", newline="")
    w = None
    best, best_ep, best_state, best_m, t0 = -1.0, 0, None, None, time.time()
    for ep in range(1, T.get("epochs", 80) + 1):
        model.train(); tot = 0.0
        for batch in loader:
            batch = batch.to(dev)
            if kind == "trust_stage":
                loss = trust_bce(model(batch), batch)
            elif kind == "trust_mp":
                rel, tr, gates = model.forward_all(batch)
                tl = trust_bce(tr, batch) + sum(trust_bce(g, batch) for g in gates) / len(gates)
                loss = listwise_loss(rel, batch) + w_trust * tl
            else:
                loss = listwise_loss(model(batch), batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * batch.num_graphs
        m = dev_metric(model, kind, dev_clean, dev_att, dev)
        if w is None:
            w = csv.writer(log); w.writerow(["epoch", "loss"] + [f"dev_{k}" for k in m])
        w.writerow([ep, tot / len(fit_set)] + list(m.values())); log.flush()
        if m["sel"] > best + 1e-6:
            best, best_ep, best_state, best_m = m["sel"], ep, copy.deepcopy(model.state_dict()), m
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d} loss {tot/len(fit_set):.4f} dev " +
                  " ".join(f"{k} {v:.3f}" for k, v in m.items()) + f" (best {best:.3f} @ {best_ep})")
        if ep - best_ep >= patience:
            print(f"  early stop at ep {ep}")
            break
    log.close()
    torch.save({"state_dict": best_state, "cfg": cfg, "in_dim": in_dim, "edge_dim": edge_dim, "seed": a.seed},
               f"checkpoints/{name}_s{a.seed}.pt")
    row = {"method": name, "kind": kind, "seed": a.seed, "best_epoch": best_ep, "params": n_params,
           "train_time_s": round(time.time() - t0, 1), **{f"dev_{k}": v for k, v in best_m.items()}}
    with open("results/week3.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
