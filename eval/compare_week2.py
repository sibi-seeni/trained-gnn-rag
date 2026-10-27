"""Week 2 comparison: plain search vs trained rankers on finding the second fact.

Usage (from repo root, after the training runs):
    python -m eval.compare_week2 configs/gnn_relevance.yaml

Reads every results/week2/preds/{name}_s{seed}.jsonl, recomputes plain dense search on
the same 500 val questions, and prints:
  1. a markdown table (mean ± std over seeds) to paste into NOTES.md
  2. paired bootstrap 95% CIs on hop2@4 for gnn vs dense and gnn vs mlp
  3. how many second facts each model rescues / loses relative to plain search
  4. the seed-noise number (sigma_seed) the pre-registered threshold uses
Also writes results/week2/summary.json.
"""
import glob, json, os, re, sys
from collections import defaultdict

import numpy as np

from common.config import load_cfg
from graph.dataset import load_split
from eval.second_hop import dense_records, metrics, hits

COLS = ["hop2@2", "hop2@4", "hop2_mrr", "both_gold@2", "both_gold@4", "recall@4", "mrr",
        "hop2@4_bridge", "hop2@4_comparison"]
K = 4


def paired_bootstrap(a, b, n=10000, seed=0):
    """a, b: dict qid -> per-question score. Returns mean diff (b - a) and 95% CI, in points."""
    q = sorted(set(a) & set(b))
    d = np.array([b[x] - a[x] for x in q])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


def main():
    cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else "configs/gnn_relevance.yaml")
    val = load_split(cfg, "val")
    dense = dense_records(val)

    runs = defaultdict(dict)  # name -> seed -> records
    for path in sorted(glob.glob("results/week2/preds/*_s*.jsonl")):
        name, seed = re.match(r".*/(.+)_s(\d+)\.jsonl", path).groups()
        runs[name][int(seed)] = [json.loads(l) for l in open(path)]
    if not runs:
        sys.exit("no runs in results/week2/preds/ yet; train first")

    # 1. table
    table = {"plain_search": {c: (100 * metrics(dense)[c], 0.0, 1) for c in COLS}}
    for name, by_seed in runs.items():
        ms = [metrics(r) for r in by_seed.values()]
        table[name] = {c: (100 * np.mean([m[c] for m in ms]), 100 * np.std([m[c] for m in ms], ddof=1)
                           if len(ms) > 1 else 0.0, len(ms)) for c in COLS}
    lines = ["| method | seeds | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for name, row in table.items():
        n = next(iter(row.values()))[2]
        cells = [f"{m:.1f}" if name == "plain_search" or n == 1 else f"{m:.1f} ± {s:.1f}" for m, s, _ in row.values()]
        lines.append(f"| {name} | {n if name != 'plain_search' else '-'} | " + " | ".join(cells) + " |")
    print(f"\n## Week 2 results (val, {len(val)} questions, numbers in points, mean ± std over seeds)\n")
    print("\n".join(lines))

    # 2. paired bootstrap on per-question hop2@4, averaged over seeds
    def seed_avg(by_seed):
        hs = [hits(r, K) for r in by_seed.values()]
        return {q: np.mean([h[q] for h in hs]) for q in hs[0]}
    dense_h = hits(dense, K)
    avg = {name: seed_avg(bs) for name, bs in runs.items()}
    print(f"\n## Paired bootstrap on hop2@{K} (10k resamples over questions; points)\n")
    pairs = [("plain_search", n) for n in avg] + \
            [(m, g) for m in avg for g in avg if m.startswith("mlp") and g.startswith("gnn")]
    boot = {}
    for a_name, b_name in pairs:
        a = dense_h if a_name == "plain_search" else avg[a_name]
        mean, lo, hi = paired_bootstrap(a, avg[b_name])
        boot[f"{b_name}_vs_{a_name}"] = [mean, lo, hi]
        print(f"- {b_name} − {a_name}: {mean:+.1f}  [95% CI {lo:+.1f}, {hi:+.1f}]")

    # 3. rescued / lost second facts (majority vote over seeds)
    print(f"\n## Second facts rescued / lost vs plain search (hop2@{K}, majority over seeds)\n")
    flow = {}
    for name, h in avg.items():
        resc = sum(1 for q in h if h[q] > 0.5 and dense_h.get(q, 0) == 0)
        lost = sum(1 for q in h if h[q] <= 0.5 and dense_h.get(q, 0) == 1)
        flow[name] = {"rescued": resc, "lost": lost}
        print(f"- {name}: rescued {resc}, lost {lost}, net {resc - lost:+d}")

    # 4. seed noise for the pre-registered threshold
    sig = {n: table[n][f"hop2@{K}"][1] for n in runs if table[n][f"hop2@{K}"][2] > 1}
    sigma_seed = max(sig.values()) if sig else float("nan")
    print(f"\n## Seed noise\n\nsigma_seed (largest seed std of hop2@{K} among trained models) = {sigma_seed:.2f} points")
    print("Copy this into PREREGISTRATION.md section 4 and commit it before any Week 3 trust numbers exist.")

    os.makedirs("results/week2", exist_ok=True)
    json.dump({"table": {n: {c: v[:2] for c, v in r.items()} for n, r in table.items()},
               "bootstrap_hop2@4": boot, "rescued_lost": flow, "sigma_seed_hop2@4": sigma_seed,
               "hop2_rule_counts_val": {r: sum(d.hop2_rule == r for d in val) for r in ("title", "cos", "none")}},
              open("results/week2/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
