"""Build passage graphs for a split, save them, and print/record the graph stats.

Usage (from repo root):
    python -m graph.run_graphs configs/baseline_plain.yaml               # entity + title edges
    python -m graph.run_graphs configs/baseline_plain.yaml --sim 0.5     # + similarity edges
"""
import argparse, json, os
import numpy as np, torch, yaml
from tqdm import tqdm
from retrieval.embed import encode_cached
from graph.build_graphs import build
from graph.stats import stats

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--split", default="val")
ap.add_argument("--sim", type=float, default=None, help="cosine threshold for similarity edges")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
E, out_dir = cfg["embed"], cfg["data"]["out_dir"]
exs = [json.loads(l) for l in open(f"{out_dir}/{a.split}.jsonl")]

# some HotpotQA examples have fewer than 10 paragraphs, so slice by offsets, not i*10
offs = np.cumsum([0] + [len(e["passages"]) for e in exs])
pv = encode_cached([p["title"] + ". " + p["text"] for e in exs for p in e["passages"]],
                   E["model"], E["cache_dir"], f"p_{a.split}_all")

graphs = [build(e, pv[offs[i]:offs[i + 1]], a.sim) for i, e in enumerate(tqdm(exs, desc="graphs"))]

tag = f"{a.split}" + (f"_sim{a.sim}" if a.sim is not None else "")
torch.save(graphs, f"{out_dir}/graphs_{tag}.pt")

row = {"graphs": tag, "n": len(graphs), **stats(graphs)}
# gold-pair connectivity by question type (bridge questions are the ones that need the "second hop")
for qtype in ("bridge", "comparison"):
    sub = [g for g, e in zip(graphs, exs) if e["type"] == qtype]
    if sub:
        row[f"gold_pair_linked_{qtype}"] = stats(sub)["gold_pair_linked"]
row["avg_edges_per_graph"] = float(np.mean([g.edge_index.size(1) for g in graphs]))

print(json.dumps(row, indent=2))
os.makedirs("results", exist_ok=True)
with open("results/graph_stats.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")