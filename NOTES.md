# Notes

## Setup
- Date of graph-stats run: Oct 26, 2026. Baseline run: Oct 27, 2026.
- Compute: HiPerGator, account/QOS eee6778, node type: B200
- Env: /blue/eee6778/si.seenivasan/conda/envs/gnnrag (versions in requirements.lock.txt)
- Git commit of the graph-stats run: e917b05
- Git commit of the baseline run: d3d9dba

## Reader model
- Llama 3.2 3B not available on NaviGator (checked Oct 20, 2026).
- Main reader: granite-3.3-8b-instruct, temperature 0, max_tokens 64.
- Strong-reader check (later): llama-3.3-70b-instruct and/or gemma-4-31b-it.

## Data
- HotpotQA distractor, seed 42: 2000 train / 500 val questions.
- Gold passages per question: 2 for every question.
- Examples with fewer than 10 passages: 7 (handled with offsets, not i*10).
- Question types in val: 413 bridge (83%), 87 comparison (17%).

## Baseline: plain dense search (MiniLM, k = 4)
Command: `python run_pipeline.py configs/baseline_plain.yaml`

| n_q | EM | F1 | recall@2 | recall@4 | recall@10 | both_gold@2 | both_gold@4 | both_gold@10 | MRR | latency (ms, first run) |
|-----|----|----|----------|----------|-----------|-------------|-------------|--------------|-----|-------------------------|
| 100 |  0.43  |  0.590  |     0.615     |     0.775     |     1.0      |       0.31      |       0.56      |       1.0      |  0.897   |           276.615             |
| 500 |  0.398  |  0.554  |     0.621     |     0.784     |     1.0      |       0.316      |       0.596      |       1.0        |  0.900   |           611.609             |

Sanity checks:
- Raw reader outputs short? (look at cache/llm.jsonl): Yes, 1-3 words.
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
- Title edges are one-directional (passage i mentions j's title → i→j only). Make graphs undirected (`torch_geometric.utils.to_undirected`) before message passing, and note that "isolated" above counts out-degree only, so it slightly overstates truly isolated nodes.
- Comparison questions: consider adding a question node linked to every passage, or report bridge and comparison results separately.
