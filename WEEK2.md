# Week 2 (Oct 26 – Nov 1): relevance-only model

> **Status:** Steps 0, 1 and 3 were run once on Oct 30, 2026 (CPU, default configs, seeds 0-2). Results are in
> NOTES.md. Tuning, the question-node variant, end-to-end QA and freezing the pre-registration (Steps 2, 4, 5)
> were not done; the project was closed after this run.

Goal for the week, in three lines:
1. Train a GATv2 ranker on relevance alone (no trust, no attacks).
2. Compare it with plain search on **finding the second fact**, with an MLP control to show whether the graph is what helps.
3. Freeze the "clearly better" rule for the make-or-break test (`PREREGISTRATION.md`) before Week 3 produces any trust numbers.

---

## What's new in the repo

```
common/config.py          configs can now say `base: other.yaml` and list only what changes
graph/dataset.py          question-conditioned node features, undirected edges, hop2 labels, optional question node
model/gnn.py              RelevanceGNN (GATv2); layers: 0 makes it an MLP on the same features
model/train.py            training + early stopping on dev + val predictions
eval/second_hop.py        hop2@k and friends, split by bridge / comparison
eval/compare_week2.py     results table, paired bootstrap, rescued/lost counts, seed noise
retrieval/gnn.py          load a checkpoint and use it as a retriever
run_pipeline.py           now supports retrieval.method: gnn (end-to-end QA with the trained ranker)
configs/gnn_relevance.yaml        main model
configs/mlp_relevance.yaml        control: no message passing
configs/gnn_relevance_qnode.yaml  optional: question node for comparison questions
scripts/week2.sh          the whole week in one script
slurm/week2.sbatch        same, as a batch job
PREREGISTRATION.md        the make-or-break rule
```

### Design decisions (so you can explain them in Delivery 1)

- **Node features are question-conditioned.** The Week 1 graphs only store the passage vector. Each node now gets
  `[p, q, p*q, cos(p,q)]`, so the GNN knows what's being asked. Built at load time; the saved graphs are unchanged.
- **Edges are undirected** (`to_undirected`, duplicate edges merged with `max`), which was the open issue from Week 1.
  Edge features are the shared-entity and title flags (cols 0, 1). Week 3 adds agree/contradict as cols 3, 4.
- **"The second fact" (hop2) is defined without the model.** If one gold passage names the other's title, the named one
  is hop2 (that's the bridge target). Otherwise it's the gold the dense retriever scores lower. `graph.dataset` prints
  how many questions used each rule.
- **Loss is listwise**: a softmax over each question's 10 passages, pushing up the gold ones. That matches how it's used
  (ranking within a question). BCE is available as `train.loss: bce`.
- **Val is never used for choices.** 10% of train (200 questions, fixed) is the dev split for early stopping.
  All tuning decisions go off dev numbers (`results/week2/logs/*.csv` and `dev_hop2@4` in `results/week2.jsonl`).
- **The MLP control** (same features, same training, zero GATv2 layers) is the honest comparison. Any trained reranker
  will probably beat untrained MiniLM cosine just from training on HotpotQA; GNN minus MLP is the part the graph adds.

---

## Steps

All commands are from the repo root on a compute node (the VS Code tunnel terminal from `slurm/vscode_b200.sbatch`), with
the env active: `module load conda && conda activate /blue/eee6778/si.seenivasan/conda/envs/gnnrag`.

### Step 0. Pull and quick check (Mon, ~10 min)

```bash
git pull
SEEDS="0" bash scripts/week2.sh     # one seed of GNN + MLP, to make sure everything runs
```

The first run builds `data/processed/graphs_train.pt` (spaCy over 20,000 train passages, a few minutes) and embeds the
train passages and questions (cached in `cache/emb/` afterwards). After that, each training run should take a couple of
minutes. The script stops at the first error.

Look at the `graph.dataset` line. Expect `x: (10, 1153)`, `edge_attr: (E, 2)`, and hop2 rule counts mostly `title`
for bridge questions. If `none` is non-zero, some questions don't have exactly two golds (NOTES says they all do, so it should be 0).

### Step 1. Full runs (Tue–Wed)

```bash
rm -f results/week2.jsonl results/week2/preds/*    # drop the quick-check runs
bash scripts/week2.sh                               # GNN + MLP, seeds 0 1 2
# or walk away:  sbatch slurm/week2.sbatch   (log in /blue/eee6778/si.seenivasan/logs/week2_<jobid>.out)
```

Optional, if there's time, the question-node variant (aimed at the 17% comparison questions):

```bash
for s in 0 1 2; do python -m model.train configs/gnn_relevance_qnode.yaml --seed $s; done
python -m eval.compare_week2 | tee results/week2/compare.md
```

### Step 2. Small tuning, on dev only (Wed, optional)

If GNN ≈ MLP, try a few settings before concluding the graph doesn't help. Make a config that inherits and changes one
thing, train seed 0, and compare `dev_hop2@4` (not val):

```yaml
# configs/gnn_l3.yaml
base: configs/gnn_relevance.yaml
name: gnn_l3
model: {layers: 3}
```

Reasonable range: layers {1, 2, 3}, hidden {128, 256}, lr {1e-3, 3e-4}, `bce_weight: 0.5`. Pick by dev, then run the
chosen one on seeds 0–2. Note in NOTES which settings you tried, even the losers. Delete the extra preds files of configs
you don't report, so `compare_week2` doesn't pick them up.

### Step 3. Compare and look at the results (Thu)

```bash
python -m eval.compare_week2 configs/gnn_relevance.yaml | tee results/week2/compare.md
```

You get:
- a table (mean ± std over seeds) with hop2@2, hop2@4, hop2 MRR, both_gold@2/@4, recall@4, MRR, and hop2@4 for bridge vs comparison;
- paired bootstrap intervals for GNN − plain search, MLP − plain search, and GNN − MLP;
- how many second facts each model rescues and loses relative to plain search;
- `sigma_seed`, the seed noise for the pre-registration.

What to look for:
- **GNN vs plain search on hop2@4** is the headline. Bridge should improve the most, since 93% of bridge gold pairs are linked.
- **GNN vs MLP** is the graph's contribution. If the interval excludes 0, message passing is doing something.
- **Comparison questions** probably won't move much without the question node (their golds are linked only 31% of the time).
- **Lost** second facts: pull 3 examples where plain search had hop2 in the top 4 and the GNN didn't, and see why
  (`results/week2/preds/*.jsonl` has the order per question; `data/processed/val.jsonl` has the text).

### Step 4. End-to-end QA, when NaviGator is back (Thu–Fri)

Week 1's baseline row is still pending. Once NaviGator responds:

```bash
python run_pipeline.py configs/baseline_plain.yaml     # Week 1 row
python run_pipeline.py configs/gnn_relevance.yaml      # same 100 questions, GNN seed 0 as retriever
```

Both rows land in `results/results.jsonl`. This is a nice-to-have for Week 2; retrieval numbers are the actual deliverable.
If NaviGator stays down, leave it for the Week 4 buffer.

### Step 5. Freeze the rule (Fri, before Week 3 starts)

1. Read `PREREGISTRATION.md` and change anything you disagree with **now** (the 3-point floor, the guardrails, which metric is primary).
2. Fill in section 4 with the `sigma_seed` value, the commit hash of the runs, and the date.
3. Paste the compare output into NOTES.md (Week 2 section).
4. Commit: `git add PREREGISTRATION.md NOTES.md results/ && git commit -m "Week 2 results; freeze make-or-break rule"` and push.
   That commit is the proof the rule came before the trust numbers.

---

## End-of-week checklist

- [ ] `graphs_train.pt` built; dataset check looks right
- [ ] GNN and MLP trained on seeds 0, 1, 2
- [ ] `results/week2/compare.md` saved, table pasted in NOTES
- [ ] 3 rescued and 3 lost examples written up in NOTES
- [ ] `sigma_seed` in PREREGISTRATION.md, committed and pushed before any Week 3 runs
- [ ] (optional) question-node variant, end-to-end QA rows

## Known issues / gotchas

- **Don't change `data.n_train`.** `data/load_hotpot.py` samples train and then val from one random stream, so changing the
  train size also changes which 500 val questions you get. If you want more training data later, give train its own seed
  in the loader first.
- **GPU runs aren't bit-for-bit repeatable** even with the same seed (scatter ops in GATv2 aren't deterministic on CUDA).
  The differences are tiny and are part of what sigma_seed measures. CPU runs are repeatable.
- **`results/week2.jsonl` is append-only.** Rerunning a seed adds a row; `compare_week2` reads the preds files instead,
  which are overwritten per name and seed, so the table always reflects the latest run.
- **If you regenerate `val.jsonl`**, rebuild the graphs too; `graph.dataset` checks that graphs and examples line up by qid.
