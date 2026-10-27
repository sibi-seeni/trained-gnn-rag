# Notes

## Setup
- Date of baseline run:
- Compute: HiPerGator, account/QOS eee6778, node type (B200 / CPU):
- Env: /blue/eee6778/si.seenivasan/conda/envs/gnnrag (versions in requirements.lock.txt)
- Git commit of the baseline run (`git rev-parse --short HEAD`):

## Reader model
- Llama 3.2 3B not available on NaviGator (checked Oct 20, 2026).
- Main reader: granite-3.3-8b-instruct, temperature 0, max_tokens 64.
- Strong-reader check (later): llama-3.3-70b-instruct.

## Data
- HotpotQA distractor, seed 42: 2000 train / 500 val questions.
- Gold passages per question (expect 2): 2
- Examples with fewer than 10 passages: 7
- Bridge vs comparison split in val: {'bridge': 413, 'comparison': 87}

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
| entity + title | 4.433 | 1.732 | 0.82 | 0.927 | 0.310 | 44.27 |
| + sim > 0.5    | 4.597 | 1.468 | 0.828 | 0.937 | 0.310 | 45.9 |

Takeaway (does the graph connect the second fact often enough for the GNN to help?):

## Open issues / decisions
- 