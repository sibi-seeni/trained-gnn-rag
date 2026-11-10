"""Week 3 / Week 5 make-or-break comparison (PREREGISTRATION.md): A vs B vs C on clean and attacked val graphs.

Usage (from repo root, after scripts/week3.sh has planted the fakes and trained everything):
    python -m eval.week3                     # attacks: edit (pre-registered), connected, connected_q (Week 5)
    python -m eval.week3 --attacks edit      # Week 3 first pass only

Per seed s in 0, 1, 2:
    A       checkpoints/gnn_relevance_s{s}.pt                       (Week 2 model, retrained, edge cols 0-1)
    B       A's score - lambda * (1 - trust_stage_s{s})             lambda picked on dev (never val)
    C       checkpoints/gnn_trust_mp_s{s}.pt, relevance head
  exploratory, not part of the decision:
    A_adv   checkpoints/gnn_relevance_adv_s{s}.pt                   relevance-only, trained on C's data
    B_adv   A_adv - lambda * (1 - trust_stage)                      lambda picked on dev
    C_rank  C - lambda * (1 - C's own trust head)                   lambda picked on dev
lambda is chosen per seed to maximise mean(dev clean hop2@4, dev attacked hop2@4), the same criterion C's
checkpoint is chosen on; ties go to the smaller lambda.

Writes results/week3/preds/{method}_{set}_s{seed}.jsonl, results/week3/compare.md, results/week3/summary.json.
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import torch

from common.config import load_cfg
from eval.compare_week2 import paired_bootstrap
from eval.second_hop import dense_records, hits, make_record, metrics
from graph.dataset import load_split
from model.train_trust import auc, node_scores
from model.trust import load

SEEDS = [0, 1, 2]
LAMBDAS = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
COLS = ["hop2@4", "poison@4", "hop2@2", "both_gold@4", "recall@4", "hop2@4_bridge", "hop2@4_comparison"]
SIGMA_SEED, FLOOR = 0.61, 3.0       # PREREGISTRATION.md section 4 (frozen in commit 52d3dc8)
DELTA = max(FLOOR, 2 * SIGMA_SEED)
DECISION = ["A", "B", "C"]
EXPLORE = ["A_adv", "B_adv", "C_rank"]


def recs_from(data, scores):
    return [make_record(d, s) for d, s in zip(data, scores)]


def combine(rel, trust_logit, lam):
    return [r - lam * (1 - torch.sigmoid(t)) for r, t in zip(rel, trust_logit)]


def pick_lambda(rel_c, tr_c, rel_a, tr_a, dev_c, dev_a):
    best = None
    for lam in LAMBDAS:
        m = (metrics(recs_from(dev_c, combine(rel_c, tr_c, lam)))["hop2@4"] +
             metrics(recs_from(dev_a, combine(rel_a, tr_a, lam)))["hop2@4"]) / 2
        if best is None or m > best[1] + 1e-9:
            best = (lam, m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="+", default=["edit", "connected", "connected_q"])
    ap.add_argument("--train_attack", default="edit")
    a = ap.parse_args()

    cfg_a, cfg_t = load_cfg("configs/gnn_relevance.yaml"), load_cfg("configs/trust_stage.yaml")
    sets = ["none"] + a.attacks
    val = {s: {"A": load_split(cfg_a, "val", s), "T": load_split(cfg_t, "val", s)} for s in sets}
    tr_c = {"A": load_split(cfg_a, "train", "none"), "T": load_split(cfg_t, "train", "none")}
    tr_a = {"A": load_split(cfg_a, "train", a.train_attack), "T": load_split(cfg_t, "train", a.train_attack)}
    perm = np.random.RandomState(cfg_a["seed"]).permutation(len(tr_c["A"]))
    dev_idx = perm[: int(cfg_a["train"].get("dev_frac", 0.1) * len(tr_c["A"]))]
    dev = {"clean": {k: [v[i] for i in dev_idx] for k, v in tr_c.items()},
           "att": {k: [v[i] for i in dev_idx] for k, v in tr_a.items()}}

    recs = defaultdict(dict)       # (method, set) -> seed -> records
    lambdas = defaultdict(dict)    # method -> seed -> lambda
    trust_auc = defaultdict(list)  # (model, set) -> per-seed AUC
    os.makedirs("results/week3/preds", exist_ok=True)
    dev_c, dev_a = dev["clean"], dev["att"]

    for s in SEEDS:
        mA, _ = load(f"checkpoints/gnn_relevance_s{s}.pt")
        mT, _ = load(f"checkpoints/trust_stage_s{s}.pt")
        mC, _ = load(f"checkpoints/gnn_trust_mp_s{s}.pt")
        adv_path = f"checkpoints/gnn_relevance_adv_s{s}.pt"
        mAdv = load(adv_path)[0] if os.path.exists(adv_path) else None

        def rel(model, kind, data):
            return [r for r, _ in node_scores(model, data, kind, "cpu")]

        def tru(model, kind, data):
            return [t for _, t in node_scores(model, data, kind, "cpu")]

        # lambda on dev
        dA = {k: rel(mA, "relevance", dev[k]["A"]) for k in dev}
        dT = {k: tru(mT, "trust_stage", dev[k]["T"]) for k in dev}
        dC = {k: node_scores(mC, dev[k]["T"], "trust_mp", "cpu") for k in dev}
        lambdas["B"][s] = pick_lambda(dA["clean"], dT["clean"], dA["att"], dT["att"], dev_c["A"], dev_a["A"])
        lambdas["C_rank"][s] = pick_lambda([r for r, _ in dC["clean"]], [t for _, t in dC["clean"]],
                                           [r for r, _ in dC["att"]], [t for _, t in dC["att"]], dev_c["T"], dev_a["T"])
        if mAdv is not None:
            dAdv = {k: rel(mAdv, "relevance_adv", dev[k]["T"]) for k in dev}
            lambdas["B_adv"][s] = pick_lambda(dAdv["clean"], dT["clean"], dAdv["att"], dT["att"], dev_c["T"], dev_a["T"])

        for st in sets:
            vA, vT = val[st]["A"], val[st]["T"]
            sA, sT = rel(mA, "relevance", vA), tru(mT, "trust_stage", vT)
            sC = node_scores(mC, vT, "trust_mp", "cpu")
            out = {"A": recs_from(vA, sA),
                   "B": recs_from(vA, combine(sA, sT, lambdas["B"][s][0])),
                   "C": recs_from(vT, [r for r, _ in sC]),
                   "C_rank": recs_from(vT, combine([r for r, _ in sC], [t for _, t in sC], lambdas["C_rank"][s][0]))}
            if mAdv is not None:
                sAdv = rel(mAdv, "relevance_adv", vT)
                out["A_adv"] = recs_from(vT, sAdv)
                out["B_adv"] = recs_from(vT, combine(sAdv, sT, lambdas["B_adv"][s][0]))
            for meth, r in out.items():
                recs[(meth, st)][s] = r
                with open(f"results/week3/preds/{meth}_{st}_s{s}.jsonl", "w") as f:
                    for x in r:
                        f.write(json.dumps(x) + "\n")
            if st != "none":
                y = torch.cat([(~d.fake).float() for d in vT]).numpy()
                trust_auc[("trust_stage", st)].append(auc(torch.cat(sT).numpy(), y))
                trust_auc[("C_trust_head", st)].append(auc(torch.cat([t for _, t in sC]).numpy(), y))
        print(f"seed {s}: lambda B={lambdas['B'][s]}, C_rank={lambdas['C_rank'][s]}, B_adv={lambdas['B_adv'].get(s)}")

    for st in sets:
        recs[("plain_search", st)][0] = dense_records(val[st]["A"])

    # ---------------------------------------------------------------- tables
    def agg(meth, st):
        ms = [metrics(r) for r in recs[(meth, st)].values()]
        return {c: (100 * np.mean([m.get(c, np.nan) for m in ms]),
                    100 * np.std([m.get(c, np.nan) for m in ms], ddof=1) if len(ms) > 1 else 0.0) for c in COLS}

    methods = ["plain_search"] + DECISION + [m for m in EXPLORE if (m, "none") in recs]
    lines, summary = [], {"delta": DELTA, "sigma_seed": SIGMA_SEED, "lambdas": {m: {s: v[0] for s, v in d.items()}
                                                                              for m, d in lambdas.items()}}
    lines.append(f"# Week 3 / 5: trust in message passing (C) vs trust at final ranking (B)\n")
    lines.append(f"Val, {len(val['none']['A'])} questions, 3 fakes per question on attacked sets. Points, mean ± std over "
                 f"seeds 0-2. λ per seed (B): {[lambdas['B'][s][0] for s in SEEDS]}, (C_rank): "
                 f"{[lambdas['C_rank'][s][0] for s in SEEDS]}, (B_adv): {[lambdas['B_adv'][s][0] for s in SEEDS] if lambdas['B_adv'] else '-'}. "
                 f"Rows below the line are exploratory and not part of the decision.\n")
    table = {}
    for st in sets:
        lines.append(f"\n## {'clean (no fakes)' if st == 'none' else 'attack: ' + st}\n")
        cols = [c for c in COLS if st != "none" or c != "poison@4"]
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for meth in methods:
            if meth == "A_adv":
                lines.append("| *exploratory* |" + " |" * len(cols))
            row = agg(meth, st); table[f"{meth}/{st}"] = row
            cells = [f"{m:.1f}" if meth == "plain_search" else f"{m:.1f} ± {sd:.1f}" for c in cols for m, sd in [row[c]]]
            lines.append(f"| {meth} | " + " | ".join(cells) + " |")
        if st != "none":
            lines.append("\nTrust AUC (real vs planted, val nodes): " + ", ".join(
                f"{k} {100 * np.mean(trust_auc[(k, st)]):.1f} ± {100 * np.std(trust_auc[(k, st)], ddof=1):.1f}"
                for k in ("trust_stage", "C_trust_head")))
    summary["table"] = table
    summary["trust_auc"] = {f"{k}/{st}": v for (k, st), v in trust_auc.items()}

    # ---------------------------------------------------------------- the pre-registered rule
    def per_seed(meth, st, col):
        return {s: metrics(r)[col] * 100 for s, r in recs[(meth, st)].items()}

    def seed_avg_hits(meth, st):
        hs = [hits(r, 4) for r in recs[(meth, st)].values()]
        return {q: np.mean([h[q] for h in hs]) for q in hs[0]}

    def rule(st, b="B", c="C", a_ref="A"):
        B, C = per_seed(b, st, "hop2@4"), per_seed(c, st, "hop2@4")
        diff = np.mean(list(C.values())) - np.mean(list(B.values()))
        mean, lo, hi = paired_bootstrap(seed_avg_hits(b, st), seed_avg_hits(c, st))
        pB, pC = np.mean(list(per_seed(b, st, "poison@4").values())), np.mean(list(per_seed(c, st, "poison@4").values()))
        cA, cC = np.mean(list(per_seed(a_ref, "none", "hop2@4").values())), np.mean(list(per_seed(c, "none", "hop2@4").values()))
        conds = {"1_size": diff >= DELTA, "2_every_seed": all(C[s] > B[s] for s in SEEDS), "3_bootstrap": lo > 0,
                 "4_guardrails": (pC - pB <= 2.0) and abs(cC - cA) <= 2.0}
        outcome = "PASS" if all(conds.values()) else ("WEAK" if diff >= 0 else "FAIL")
        return {"C_minus_B": diff, "per_seed_B": B, "per_seed_C": C, "bootstrap": [mean, lo, hi],
                "poison_B": pB, "poison_C": pC, "clean_A": cA, "clean_C": cC, "conditions": conds, "outcome": outcome}

    lines.append(f"\n## Pre-registered rule (Δ = max(3.0, 2 × {SIGMA_SEED}) = {DELTA:.1f} points)\n")
    lines.append("| attack | C − B (mean) | per seed B / C | bootstrap 95% CI | poison@4 B / C | clean hop2@4 A / C | "
                 "1 size | 2 every seed | 3 CI > 0 | 4 guardrails | outcome |")
    lines.append("|" + "---|" * 11)
    summary["rule"] = {}
    for st in a.attacks:
        r = rule(st); summary["rule"][st] = r
        ps = ", ".join(f"{r['per_seed_B'][s]:.1f}/{r['per_seed_C'][s]:.1f}" for s in SEEDS)
        ok = lambda k: "yes" if r["conditions"][k] else "no"
        lines.append(f"| {st} | {r['C_minus_B']:+.1f} | {ps} | [{r['bootstrap'][1]:+.1f}, {r['bootstrap'][2]:+.1f}] | "
                     f"{r['poison_B']:.1f} / {r['poison_C']:.1f} | {r['clean_A']:.1f} / {r['clean_C']:.1f} | "
                     f"{ok('1_size')} | {ok('2_every_seed')} | {ok('3_bootstrap')} | {ok('4_guardrails')} | **{r['outcome']}** |")
    lines.append("\nThe pre-registered attack set is `edit`; `connected` and `connected_q` are the Week 5 harder attacks, "
                 "judged with the same rule (PREREGISTRATION.md section 6 and change log).")

    # exploratory: same arithmetic for other pairs, clearly not the decision
    if lambdas["B_adv"]:
        lines.append("\n### Exploratory: the same rule with B_adv in B's place (B's relevance part trained on C's data)\n")
        lines.append("| attack | C − B_adv (mean) | per seed B_adv / C | bootstrap 95% CI | poison@4 B_adv / C | "
                     "1 size | 2 every seed | 3 CI > 0 | 4 guardrails | outcome |")
        lines.append("|" + "---|" * 10)
        summary["rule_vs_B_adv"] = {}
        for st in a.attacks:
            r = rule(st, b="B_adv"); summary["rule_vs_B_adv"][st] = r
            ps = ", ".join(f"{r['per_seed_B'][s]:.1f}/{r['per_seed_C'][s]:.1f}" for s in SEEDS)
            ok = lambda k: "yes" if r["conditions"][k] else "no"
            lines.append(f"| {st} | {r['C_minus_B']:+.1f} | {ps} | [{r['bootstrap'][1]:+.1f}, {r['bootstrap'][2]:+.1f}] | "
                         f"{r['poison_B']:.1f} / {r['poison_C']:.1f} | {ok('1_size')} | {ok('2_every_seed')} | "
                         f"{ok('3_bootstrap')} | {ok('4_guardrails')} | {r['outcome']} |")
        lines.append("\n### Exploratory pairs (same arithmetic, not part of the decision)\n")
        lines.append("| attack | pair | diff (mean) | bootstrap 95% CI | every seed |")
        lines.append("|---|---|---|---|---|")
        summary["exploratory"] = {}
        for st in a.attacks:
            for b, c in (("A_adv", "C"), ("B_adv", "C"), ("B", "B_adv"), ("A", "A_adv"), ("C", "C_rank"), ("A", "B")):
                B, C = per_seed(b, st, "hop2@4"), per_seed(c, st, "hop2@4")
                d = np.mean(list(C.values())) - np.mean(list(B.values()))
                _, lo, hi = paired_bootstrap(seed_avg_hits(b, st), seed_avg_hits(c, st))
                summary["exploratory"][f"{c}_vs_{b}/{st}"] = [d, lo, hi]
                lines.append(f"| {st} | {c} − {b} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
                             f"{'yes' if all(C[s] > B[s] for s in SEEDS) else 'no'} |")

    text = "\n".join(lines)
    print(text)
    open("results/week3/compare.md", "w").write(text + "\n")
    json.dump(summary, open("results/week3/summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
