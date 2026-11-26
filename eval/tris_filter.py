"""Side comparison: TRIS-style filtering (Layers 1-2) on our planted fakes. Exploratory, not part of the decision.

TRIS (Munir et al., arXiv 2609.00470) filters the retrieved passages before the reader. Layers 1 and 2 need no LLM:
  Layer 1  embed candidates with all-MiniLM-L6-v2, K-Means (C clusters), keep the majority cluster, but only if it
           holds more than half the candidates (else keep everything).
  Layer 2  on what Layer 1 kept, drop a passage whose first 20 tokens overlap the question too much: token Jaccard,
           bigram or trigram overlap >= 0.8, or the question's first 12 tokens appearing twice in the prefix.
  If everything is removed, the original list is returned.
Settings, tokeniser and order of operations follow the released code (github.com/akibjawad14/tris, commit 3a79518,
src/defense.py `TriLayerSieve`), with --cluster_count 2 as in every command of their experiments.md (the paper text
says 3, so C = 3 and 5 are a sweep). The filter sees passage text only, no title, as in their code. Layer 3 (LLM
consistency) is not run.

Each question's full candidate set (10 real passages + 3 fakes on attacked sets) goes through the filter. Then the
committed Week 3/5 rankings (results/week3/preds/) are re-scored with the removed passages dropped; plain search is
recomputed from the graphs. A removed second-fact passage counts as a miss (not dropped from the denominator).

Usage (from repo root, after attacks/plant.py has rebuilt data/processed/val_{none,edit,connected,connected_q}.*):
    python -m eval.tris_filter
Writes results/tris/compare.md, results/tris/summary.json, results/tris/removed.jsonl.
"""
import argparse, json, os, re
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans

from common.config import load_cfg
from eval.compare_week2 import paired_bootstrap
from eval.second_hop import dense_records, metrics
from graph.dataset import load_split
from retrieval.embed import encode_cached

SETS = ["none", "edit", "connected", "connected_q"]
METHODS = ["plain_search", "A", "B", "C", "A_adv", "B_adv", "C_rank"]
SEEDS = [0, 1, 2]
JUDGE = "sentence-transformers/all-MiniLM-L6-v2"
PREFIX, JAC, NGRAM, QPREFIX = 20, 0.8, 0.8, 12     # TriLayerSieve defaults, as used in their experiments.md
# filter configs: name -> (cluster count for Layer 1 or None, run Layer 2?)
CONFIGS = {"L1 (C=2)": (2, False), "L1 (C=3)": (3, False), "L1 (C=5)": (5, False), "L2": (None, True),
           "L1+L2 (C=2)": (2, True), "L1+L2 (C=3)": (3, True), "L1+L2 (C=5)": (5, True)}
MAIN, FULL = "L1+L2 (C=2)", ["L1+L2 (C=2)", "L2"]


def tok(t):
    return re.findall(r"\b\w+\b", t.lower())


def ngrams(ts, n):
    return {tuple(ts[i:i + n]) for i in range(len(ts) - n + 1)} if len(ts) >= n else set()


def layer1(emb, C):
    n = len(emb)
    if n < 3 or C is None or C < 2:
        return set()
    labels = KMeans(n_clusters=min(C, n), random_state=42, n_init=10).fit_predict(emb)
    lab, size = Counter(labels).most_common(1)[0]
    if size <= n / 2:                                   # no clear majority: keep everything
        return set()
    return {i for i in range(n) if labels[i] != lab}


def layer2(question, texts, idx):
    q = tok(question)
    qs, q2, q3, qp = set(q), ngrams(q, 2), ngrams(q, 3), " ".join(q[:min(len(q), QPREFIX)])
    out = set()
    for i in idx:
        p = tok(texts[i])[:PREFIX]
        ps = set(p)
        jac = len(qs & ps) / len(qs | ps) if qs | ps else 0.0
        b = len(q2 & ngrams(p, 2)) / max(1, len(q2))
        t = len(q3 & ngrams(p, 3)) / max(1, len(q3))
        rep = len(qp) > 0 and " ".join(p).count(qp) >= 2
        if jac >= JAC or b >= NGRAM or t >= NGRAM or rep:
            out.add(i)
    return out


def sieve(question, texts, emb, C, use_l2):
    n = len(texts)
    r1 = layer1(emb, C)
    r2 = layer2(question, texts, [i for i in range(n) if i not in r1]) if use_l2 else set()
    removed = r1 | r2
    if len(removed) == n:                               # TRIS fallback: return the original list
        return r1, r2, set()
    return r1, r2, removed


def refilter(rec, d, removed):
    """Drop removed nodes from a ranking record; a removed hop2 passage is a miss (rank past the end)."""
    y, fk, h2 = d.y.bool().tolist(), d.fake.bool().tolist(), d.hop2.bool().tolist()
    order = [j for j in rec["order"] if j not in removed]
    hop2 = [j for j in range(len(h2)) if h2[j]]
    rank = next((r + 1 for r, j in enumerate(order) if h2[j]), None)
    if rank is None and hop2:
        rank = len(rec["order"]) + 1
    out = {"qid": rec["qid"], "qtype": rec["qtype"], "order": order, "flags": [y[j] for j in order], "hop2_rank": rank}
    if any(fk):
        out["fakes"] = [fk[j] for j in order]
    return out


def per_q(recs, col):
    if col == "hop2@4":
        return {r["qid"]: float(r["hop2_rank"] is not None and r["hop2_rank"] <= 4) for r in recs}
    return {r["qid"]: float(any(r.get("fakes", [])[:4])) for r in recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tris")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_cfg("configs/gnn_relevance.yaml")
    out_dir, E = cfg["data"]["out_dir"], cfg["embed"]

    filt = {}                      # (set, config) -> stats
    downstream = defaultdict(dict) # (method, set, config or "-") -> seed -> records
    qlen = {}
    rows = []
    for st in SETS:
        exs = [json.loads(l) for l in open(f"{out_dir}/val_{st}.jsonl")]
        data = load_split(cfg, "val", st)
        texts = [p["text"] for e in exs for p in e["passages"]]
        emb = encode_cached(texts, JUDGE, E["cache_dir"], f"tris_text_val_{st}")
        offs = np.cumsum([0] + [len(e["passages"]) for e in exs])
        for e, d in zip(exs, data):    # node order in the graphs = passage order in the jsonl
            assert d.qid == e["qid"] and len(d.y) == len(e["passages"])
            assert d.fake.tolist() == [p["fake"] for p in e["passages"]]

        removed = {c: {} for c in CONFIGS}
        for c, (C, l2) in CONFIGS.items():
            S = Counter()
            for i, (e, d) in enumerate(zip(exs, data)):
                r1, r2, rm = sieve(e["question"], [p["text"] for p in e["passages"]], emb[offs[i]:offs[i + 1]], C, l2)
                removed[c][e["qid"]] = rm
                rows.append({"set": st, "config": c, "qid": e["qid"], "l1": sorted(r1), "l2": sorted(r2),
                             "removed": sorted(rm)})
                fk, y, h2 = d.fake.tolist(), d.y.bool().tolist(), d.hop2.bool().tolist()
                for j in range(len(fk)):
                    gone = j in rm
                    S["fake" if fk[j] else "real"] += 1; S[("fake_rm" if fk[j] else "real_rm")] += gone
                    if y[j]: S["gold"] += 1; S["gold_rm"] += gone
                    if h2[j]: S["hop2"] += 1; S["hop2_rm"] += gone
                    if fk[j]: S["fake_l1"] += j in r1; S["fake_l2"] += j in r2
                S["q"] += 1
                S["q_fake_left"] += any(fk[j] and j not in rm for j in range(len(fk)))
                S["q_none_removed"] += not rm
                S["kept"] += len(fk) - len(rm)
                if st == "connected_q" and c == "L2":
                    n_q = len(tok(e["question"]))
                    for j in range(len(fk)):
                        if fk[j]:
                            qlen.setdefault("caught" if j in rm else "missed", []).append(n_q)
            pct = lambda a, b: 100 * S[a] / S[b] if S[b] else float("nan")
            filt[(st, c)] = {"fakes_removed": pct("fake_rm", "fake"), "real_removed": pct("real_rm", "real"),
                             "gold_removed": pct("gold_rm", "gold"), "hop2_removed": pct("hop2_rm", "hop2"),
                             "fakes_removed_by_L1": pct("fake_l1", "fake"), "fakes_removed_by_L2": pct("fake_l2", "fake"),
                             "q_with_fake_left": pct("q_fake_left", "q"), "q_untouched": pct("q_none_removed", "q"),
                             "passages_kept": S["kept"] / S["q"]}

        # downstream: committed rankings, re-scored after filtering
        by_qid = {d.qid: d for d in data}
        base = {"plain_search": {0: dense_records(data)}}
        for m in METHODS[1:]:
            base[m] = {s: [json.loads(l) for l in open(f"results/week3/preds/{m}_{st}_s{s}.jsonl")] for s in SEEDS}
        for m, per_seed in base.items():
            for s, recs in per_seed.items():
                for r in recs:     # committed rankings must line up with the rebuilt graphs
                    assert r["flags"] == [bool(by_qid[r["qid"]].y[j]) for j in r["order"]], (m, st, s)
                downstream[(m, st, "-")][s] = recs
                for c in CONFIGS:
                    downstream[(m, st, c)][s] = [refilter(r, by_qid[r["qid"]], removed[c][r["qid"]]) for r in recs]
        print(f"{st}: done ({len(exs)} questions)")

    with open(f"{a.out}/removed.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---------------------------------------------------------------- aggregation
    def agg(m, st, c, col):
        v = [100 * metrics(r)[col] for r in downstream[(m, st, c)].values()]
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    def seed_avg(m, st, c, col):
        ds = [per_q(r, col) for r in downstream[(m, st, c)].values()]
        return {q: float(np.mean([x[q] for x in ds])) for q in ds[0]}

    def effect(m, st, c, col):
        return paired_bootstrap(seed_avg(m, st, "-", col), seed_avg(m, st, c, col))

    table = {f"{m}/{st}/{c}": {col: agg(m, st, c, col) for col in ("hop2@4", "poison@4", "both_gold@4")
                               if not (st == "none" and col == "poison@4")}
             for m in METHODS for st in SETS for c in ["-"] + list(CONFIGS)}
    effects = {f"{m}/{st}/{c}": {col: effect(m, st, c, col) for col in ("hop2@4", "poison@4")
                                 if not (st == "none" and col == "poison@4")}
               for m in ("A", "B", "C") for st in SETS for c in FULL}

    # ---------------------------------------------------------------- prediction checks (NOTES.md, committed first)
    F = lambda st, c, k: filt[(st, c)][k]
    T = lambda m, st, c, col: table[f"{m}/{st}/{c}"][col][0]
    p1 = (F("connected_q", "L2", "fakes_removed") >= 80 and F("edit", "L2", "fakes_removed") < 5
          and F("connected", "L2", "fakes_removed") < 5 and all(F(st, "L2", "real_removed") < 1 for st in SETS))
    p2 = all(abs(F(st, "L1 (C=2)", "fakes_removed") - F(st, "L1 (C=2)", "hop2_removed")) <= 15 for st in SETS[1:])
    p3 = (all(T(m, "edit", MAIN, "poison@4") > 80 for m in METHODS) and
          all(T(m, "connected_q", "-", "poison@4") - T(m, "connected_q", MAIN, "poison@4") >= 30 for m in METHODS))
    p4 = all(T(m, "none", "-", "hop2@4") - T(m, "none", MAIN, "hop2@4") > 2 for m in ("A", "B", "C"))
    p5 = all(T("C", st, "-", "hop2@4") > T("B", st, MAIN, "hop2@4") for st in ("edit", "connected"))
    preds = {"1_L2_catches_question_prefix_only": p1, "2_L1_cannot_separate_fakes_from_gold": p2,
             "3_poison_edit_stays_high_connected_q_drops_30": p3, "4_clean_cost_over_2_points": p4,
             "5_C_beats_B_plus_filter": p5}

    # ---------------------------------------------------------------- markdown
    L = ["# Side comparison: TRIS-style filtering (Layers 1-2) on our fakes\n",
         "Exploratory, not part of the pre-registered decision. Val, 500 questions; 10 real passages + 3 fakes per "
         "question on attacked sets. Settings from the released TRIS code (commit 3a79518). Points.\n",
         "\n## 1. What the filter removes\n",
         "| set | filter | fakes removed | by L1 | by L2 | real removed | gold removed | hop2 removed | "
         "questions with a fake left | questions untouched | passages kept / q |", "|" + "---|" * 11]
    for st in SETS:
        for c in CONFIGS:
            r = filt[(st, c)]
            fk = lambda k: "-" if st == "none" else f"{r[k]:.1f}"
            L.append(f"| {st} | {c} | {fk('fakes_removed')} | {fk('fakes_removed_by_L1')} | {fk('fakes_removed_by_L2')} | "
                     f"{r['real_removed']:.1f} | {r['gold_removed']:.1f} | {r['hop2_removed']:.1f} | "
                     f"{fk('q_with_fake_left')} | {r['q_untouched']:.1f} | {r['passages_kept']:.2f} |")
    if qlen:
        L.append(f"\n`connected_q`, Layer 2 alone: question length (tokens) for fakes it caught, median "
                 f"{np.median(qlen.get('caught', [np.nan])):.0f} (n = {len(qlen.get('caught', []))}); for fakes it missed, "
                 f"median {np.median(qlen.get('missed', [np.nan])):.0f} (n = {len(qlen.get('missed', []))}).")

    for c in FULL:
        for col in ("hop2@4", "poison@4"):
            sets = SETS if col == "hop2@4" else SETS[1:]
            L.append(f"\n## 2{'a' if c == MAIN else 'b'}. {col} without → with `{c}` (mean over seeds 0-2)\n")
            L.append("| method | " + " | ".join(sets) + " |")
            L.append("|" + "---|" * (len(sets) + 1))
            for m in METHODS:
                if m == "A_adv":
                    L.append("| *exploratory* |" + " |" * len(sets))
                cells = [f"{table[f'{m}/{st}/-'][col][0]:.1f} → {table[f'{m}/{st}/{c}'][col][0]:.1f}" for st in sets]
                L.append(f"| {m} | " + " | ".join(cells) + " |")

    L.append(f"\n## 3. Filter effect, paired bootstrap over questions (with − without, 95% CI, seed-averaged)\n")
    L.append("| method | filter | set | hop2@4 | poison@4 |")
    L.append("|---|---|---|---|---|")
    for c in FULL:
        for m in ("A", "B", "C"):
            for st in SETS:
                e = effects[f"{m}/{st}/{c}"]
                fmt = lambda k: "-" if k not in e else f"{e[k][0]:+.1f} [{e[k][1]:+.1f}, {e[k][2]:+.1f}]"
                L.append(f"| {m} | {c} | {st} | {fmt('hop2@4')} | {fmt('poison@4')} |")

    L.append("\n## 4. Cluster-count sweep (hop2@4 / poison@4, mean over seeds)\n")
    L.append("| method | filter | " + " | ".join(SETS) + " |")
    L.append("|" + "---|" * (len(SETS) + 2))
    for m in ("plain_search", "A", "B", "C"):
        for c in ["-"] + [k for k in CONFIGS if k.startswith("L1+L2")]:
            cells = [f"{table[f'{m}/{st}/{c}']['hop2@4'][0]:.1f}" +
                     ("" if st == "none" else f" / {table[f'{m}/{st}/{c}']['poison@4'][0]:.1f}") for st in SETS]
            L.append(f"| {m} | {'no filter' if c == '-' else c} | " + " | ".join(cells) + " |")

    L.append("\n## 5. Predictions (NOTES.md, committed before running)\n")
    for k, v in preds.items():
        L.append(f"- {k}: **{'held' if v else 'did not hold'}**")

    open(f"{a.out}/compare.md", "w").write("\n".join(L) + "\n")
    json.dump({"filter": {f"{st}/{c}": v for (st, c), v in filt.items()}, "table": table, "effects": effects,
               "predictions": preds, "qlen_connected_q_L2": {k: [int(np.median(v)), len(v)] for k, v in qlen.items()},
               "settings": {"judge": JUDGE, "prefix_tokens": PREFIX, "jaccard": JAC, "ngram": NGRAM,
                            "query_prefix_tokens": QPREFIX, "kmeans": "random_state=42, n_init=10",
                            "tris_commit": "3a79518"}},
              open(f"{a.out}/summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
