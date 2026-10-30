"""Train the relevance-only ranker (GATv2, or MLP with model.layers: 0) and score it on val.

Usage (from repo root):
    python -m model.train configs/gnn_relevance.yaml --seed 0
    python -m model.train configs/mlp_relevance.yaml --seed 1

Splits:
    train.jsonl (2000 q)  -> 90% fit / 10% dev (dev is fixed by cfg.seed, the same for every run seed)
    val.jsonl   (500 q)   -> reported numbers only; never used for model selection.

Outputs:
    checkpoints/{name}_s{seed}.pt               best-on-dev weights + what's needed to rebuild the model
    results/week2/logs/{name}_s{seed}.csv       per-epoch loss and dev metrics
    results/week2/preds/{name}_s{seed}.jsonl    per-question val rankings (used by eval/compare_week2.py)
    results/week2.jsonl                         one summary row per run
"""
import argparse, copy, csv, json, os, time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from common.config import load_cfg
from graph.dataset import load_split
from model.gnn import build_model
from eval.second_hop import make_record, metrics


def listwise_loss(scores, data):
    """Softmax over the passages of each question; maximise log-prob of the gold ones."""
    s = scores.masked_fill(~data.score_mask, -1e4)
    s, pad = to_dense_batch(s, data.batch, fill_value=-1e4)
    y, _ = to_dense_batch(data.y, data.batch, fill_value=0.0)
    logp = F.log_softmax(s, dim=1)
    n_gold = y.sum(1)
    keep = n_gold > 0
    return -((logp * y).sum(1)[keep] / n_gold[keep]).mean()


def bce_loss(scores, data, pos_weight=4.0):
    m = data.score_mask
    return F.binary_cross_entropy_with_logits(scores[m], data.y[m], pos_weight=torch.tensor(pos_weight, device=scores.device))


@torch.no_grad()
def predict(model, data, device, bs=256):
    model.eval()
    recs = []
    for batch in DataLoader(data, batch_size=bs, shuffle=False):
        batch = batch.to(device)
        s = model(batch)
        for i, d in enumerate(batch.to_data_list()):
            recs.append(make_record(d, s[batch.ptr[i]:batch.ptr[i + 1]]))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_train", type=int, default=None, help="debug: cap number of train questions")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    T, name = cfg["train"], cfg["name"]
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if a.device == "auto" else a.device))

    train_all = load_split(cfg, "train")
    val = load_split(cfg, "val")
    perm = np.random.RandomState(cfg["seed"]).permutation(len(train_all))  # fixed dev split
    n_dev = int(T.get("dev_frac", 0.1) * len(train_all))
    dev_set = [train_all[i] for i in perm[:n_dev]]
    fit_set = [train_all[i] for i in perm[n_dev:]][: a.max_train]

    seed_everything(a.seed)
    in_dim = fit_set[0].x.size(1)
    edge_dim = None if fit_set[0].edge_attr is None else fit_set[0].edge_attr.size(1)
    model = build_model(cfg, in_dim, edge_dim).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=T.get("lr", 1e-3), weight_decay=T.get("weight_decay", 1e-4))
    loader = DataLoader(fit_set, batch_size=T.get("batch_size", 64), shuffle=True)
    sel, patience, bce_w = T.get("select", "hop2@4"), T.get("patience", 12), T.get("bce_weight", 0.0)
    print(f"[{name} s{a.seed}] device={dev} fit={len(fit_set)} dev={len(dev_set)} val={len(val)} "
          f"in_dim={in_dim} edge_dim={edge_dim} params={n_params:,}")

    os.makedirs("results/week2/logs", exist_ok=True); os.makedirs("results/week2/preds", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log = open(f"results/week2/logs/{name}_s{a.seed}.csv", "w", newline="")
    w = csv.writer(log); w.writerow(["epoch", "loss", f"dev_{sel}", "dev_both_gold@4", "dev_mrr"])

    best, best_ep, best_state, t0 = -1.0, 0, None, time.time()
    for ep in range(1, T.get("epochs", 80) + 1):
        model.train(); tot = 0.0
        for batch in loader:
            batch = batch.to(dev)
            s = model(batch)
            loss = listwise_loss(s, batch) if T.get("loss", "listwise") == "listwise" else bce_loss(s, batch)
            if bce_w:
                loss = loss + bce_w * bce_loss(s, batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * batch.num_graphs
        m = metrics(predict(model, dev_set, dev))
        w.writerow([ep, tot / len(fit_set), m[sel], m["both_gold@4"], m["mrr"]]); log.flush()
        if m[sel] > best + 1e-6:
            best, best_ep, best_state = m[sel], ep, copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d} loss {tot/len(fit_set):.4f} dev {sel} {m[sel]:.3f} (best {best:.3f} @ {best_ep})")
        if ep - best_ep >= patience:
            print(f"  early stop at ep {ep}")
            break
    train_time = time.time() - t0
    log.close()

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "cfg": cfg, "in_dim": in_dim, "edge_dim": edge_dim, "seed": a.seed},
               f"checkpoints/{name}_s{a.seed}.pt")

    recs = predict(model, val, dev)
    with open(f"results/week2/preds/{name}_s{a.seed}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    row = {"method": name, "seed": a.seed, "split": "val", "best_epoch": best_ep, f"dev_{sel}": best,
           "params": n_params, "train_time_s": round(train_time, 1), **metrics(recs)}
    with open("results/week2.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                      if "_bridge" not in k and "_comparison" not in k}, indent=2))


if __name__ == "__main__":
    main()
