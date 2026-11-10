"""Week 4: end-to-end QA (reader EM/F1) with each retriever, clean and under attack. Needs NaviGator (run on HPG).

Usage (from repo root, after eval/week3.py has written results/week3/preds/):
    python -m generate.qa_week4                              # A, B, C (seed 0) + plain search, clean + edit, 500 q
    python -m generate.qa_week4 --sets none edit connected_q --methods plain_search A B C --n 500

It does not rerun any model: it reads the val rankings in results/week3/preds/{method}_{set}_s{seed}.jsonl, puts the
top k passages (fakes included, if they were ranked there) in the Week 1 prompt, and scores the reader's answer.
Needs data/processed/val_{set}.jsonl (python -m attacks.plant configs/trust_stage.yaml --split val --attack <set>).
Rows go to results/week4.jsonl. Responses are cached in cache/llm.jsonl like Week 1.
"""
import argparse, json, os

import numpy as np

from common.config import load_cfg
from eval.hotpot_metrics import em, f1
from generate.answer import build_prompt
from generate.client import CachedLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["none", "edit"])
    ap.add_argument("--methods", nargs="+", default=["plain_search", "A", "B", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--config", default="configs/baseline_plain.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    llm, k = CachedLLM(cfg["generate"]), cfg["retrieval"]["k"]
    for st in a.sets:
        exs = {e["qid"]: e for e in map(json.loads, open(f'{cfg["data"]["out_dir"]}/val_{st}.jsonl'))}
        for meth in a.methods:
            seed = 0 if meth == "plain_search" else a.seed
            path = f"results/week3/preds/{meth}_{st}_s{seed}.jsonl"
            if meth == "plain_search":  # plain search ranking straight from the cos column
                from eval.second_hop import dense_records
                from graph.dataset import load_split
                recs = dense_records(load_split(load_cfg("configs/gnn_relevance.yaml"), "val", st))
            else:
                recs = [json.loads(l) for l in open(path)]
            recs = recs[: a.n]
            ems, f1s, fake_in_ctx, fake_answer = [], [], [], []
            for r in recs:
                e = exs[r["qid"]]
                ctx = [e["passages"][j] for j in r["order"][:k]]
                pred, _ = llm(build_prompt(e["question"], ctx))
                ems.append(em(pred, e["answer"])); f1s.append(f1(pred, e["answer"]))
                fakes = [p for p in ctx if p.get("fake")]
                fake_in_ctx.append(bool(fakes))
                # the reader repeated something that only a planted passage says (a swapped-in entity)
                real_text = " ".join(p["text"] for p in e["passages"] if not p.get("fake")).lower()
                fake_answer.append(bool(fakes) and len(pred) > 2 and pred.lower() not in real_text
                                   and any(pred.lower() in p["text"].lower() for p in fakes))
            row = {"method": meth, "set": st, "seed": seed, "reader": cfg["generate"]["model"], "n_q": len(recs),
                   "EM": float(np.mean(ems)), "F1": float(np.mean(f1s)),
                   "fake_in_context": float(np.mean(fake_in_ctx)), "answer_from_fake": float(np.mean(fake_answer))}
            print(json.dumps(row))
            os.makedirs("results", exist_ok=True)
            with open("results/week4.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
