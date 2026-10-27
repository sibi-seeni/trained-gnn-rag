# Notes

## Setup
- Date of graph-stats run: Oct 26, 2026. Baseline run: pending (NaviGator down on Oct 26).
- Compute: HiPerGator, account/QOS eee6778, node type: B200
- Env: /blue/eee6778/si.seenivasan/conda/envs/gnnrag (versions in requirements.lock.txt)
- Git commit of the graph-stats run: e917b05
- Git commit of the baseline run (`git rev-parse --short HEAD`):

## Reader model
- Llama 3.2 3B not available on NaviGator (checked Oct 20, 2026).
- Main reader: granite-3.3-8b-instruct, temperature 0, max_tokens 64.
- Strong-reader check (later): llama-3.3-70b-instruct.

## Data
- HotpotQA distractor, seed 42: 2000 train / 500 val questions.
- Gold passages per question: 2 for every question.
- Examples with fewer than 10 passages: 7 (handled with offsets, not i*10).
- Question types in val: 413 bridge (83%), 87 comparison (17%).

## Baseline: plain dense search (MiniLM, k = 4)
Command: `python run_pipeline.py configs/baseline_plain.yaml`

| n_q | EM | F1 | recall@2 | recall@4 | recall@10 | both_gold@2 | both_gold@4 | MRR | latency (s, first run) |
|-----|----|----|----------|----------|-----------|-------------|-------------|-----|------------------------|
| 100 |    |    |          |          |           |             |             |     |                        |
| 500 |    |    |          |          |           |             |             |     |                        |

Sanity checks:
- Raw reader outputs short? (look at cache/llm.jsonl):
- 3 example failures (question, gold answer, prediction, were both golds retrieved?):

## Graph stats (val, 500 graphs)
Command: `python -m graph.run_graphs configs/baseline_plain.yaml [--sim 0.5]`

| edges | avg degree | isolated nodes / graph | gold pair linked | linked (bridge) | linked (comparison) | avg edges per graph |
|-------|------------|------------------------|------------------|-----------------|---------------------|---------------------|
| entity + title | 4.43 | 1.73 | 0.820 | 0.927 | 0.310 | 44.3 |
| + sim > 0.5    | 4.60 | 1.47 | 0.828 | 0.937 | 0.310 | 45.9 |

Takeaway:
- Bridge questions (83% of val) have the two gold passages directly linked 93% of the time, so the graph gives the GNN a path from the first fact to the second. This is the case the project is about.
- Comparison questions are linked only 31% of the time. That is expected: their two golds are about two separate entities (e.g. "who was born first, A or B?") that don't mention each other. The question connects them, not the passages.
- Similarity edges at 0.5 add about 1.6 edges per graph and +1 point on bridge linking. Not worth it: use entity + title edges as the main graph.

## Open issues / decisions
- Decision: Week 2 uses the entity + title graph (`graphs_val.pt`).
- Title edges are one-directional (passage i mentions j's title → i→j only). Make graphs undirected (`torch_geometric.utils.to_undirected`) before message passing, and note that "isolated" above counts out-degree only, so it slightly overstates truly isolated nodes. → Done in Week 2: `graph/dataset.py` makes edges undirected at load time (saved graphs unchanged).
- Comparison questions: consider adding a question node linked to every passage, or report bridge and comparison results separately. → Week 2 reports both splits; question node is `configs/gnn_relevance_qnode.yaml` (optional run).
- Baseline row pending: NaviGator reset all connections on Oct 26 (from HiPerGator and off-campus). Rerun when it is back; fallback is vLLM on the B200.

## Week 2: relevance-only model
Runbook: `WEEK2.md`. Command: `bash scripts/week2.sh` (or `sbatch slurm/week2.sbatch`).
- Date of run:              Git commit (`git rev-parse --short HEAD`):
- Device:                   Train/dev/val: 1800 / 200 / 500 questions
- hop2 rule counts (val, from `python -m graph.dataset`): title ___, cos ___, none ___
- Tuning tried on dev (config, dev_hop2@4):

### Results (paste `results/week2/compare.md`)

| method | seeds | hop2@2 | hop2@4 | hop2_mrr | both_gold@2 | both_gold@4 | recall@4 | mrr | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|---|---|---|
| plain_search | - | | | | | | | | | |
| gnn_relevance | 3 | | | | | | | | | |
| mlp_relevance | 3 | | | | | | | | | |

Paired bootstrap (hop2@4): GNN − plain: ___ [ , ]; MLP − plain: ___ [ , ]; GNN − MLP: ___ [ , ]
Rescued / lost vs plain (GNN): ___ / ___
sigma_seed: ___ points (copied into PREREGISTRATION.md)

### Examples
- 3 second facts the GNN rescued (question, hop2 title, plain rank → GNN rank, why):
- 3 it lost:

### End-to-end QA (when NaviGator is back; first 100 val)
| retriever | EM | F1 | both_gold@4 |
|---|---|---|---|
| plain | | | |
| gnn_relevance s0 | | | |

### Takeaway
-

