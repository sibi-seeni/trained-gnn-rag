# trained-gnn-rag

Trust-aware graph retrieval for multi-hop QA (HotpotQA). The plan was to retrieve evidence with a GNN (GATv2) over a graph of passages, then add a trust signal inside message passing so poisoned passages get pushed down.

## Project status: closed (Oct 2026)

- **Done (Week 1):** data pipeline, plain dense-search baseline with an LLM reader, and passage-graph analysis. Tables in `NOTES.md`.
- **Done (Week 2):** a relevance-only GATv2 ranker plus an MLP control (same features and training, no message passing), seeds 0-2. On 500 val questions, the second gold passage lands in the top 4 for 63% of questions with plain search, 75% with the MLP and 84% with the GNN. Details and caveats in `NOTES.md` (Week 2 and Conclusion).
- **Not done:** the trust and attack work (Weeks 3-5). `attacks/` is empty and `PREREGISTRATION.md` is a plan that was never carried out. The project was stopped and the effort moved to a related project on prompt-injection-resistant agents.

Code was run on HiPerGator as Slurm jobs (Week 2 was rerun on CPU, see NOTES). Answer generation goes through UF NaviGator (OpenAI-compatible API).

## Layout

```
configs/    one YAML per experiment (a config can inherit with `base:`)
common/     config loading
data/       HotpotQA loading + processed splits (processed/ is gitignored)
graph/      passage graph construction + stats + GNN-ready datasets
retrieval/  embeddings, dense ranking, trained-ranker wrapper
model/      GATv2 ranker + training
attacks/    poisoning (never started)
generate/   NaviGator client + prompt
eval/       EM/F1, retrieval metrics, second-fact metrics, Week 2 comparison
scripts/    one script per week of runs
slurm/      sbatch scripts + env setup
```

## Setup (HiPerGator)

1. Create the env once from a compute node: `bash slurm/setup_env.sh` (exact versions are in `requirements.lock.txt`).
2. Save your NaviGator key on HPG at `~/.secrets/navigator_key` (chmod 600). Never commit it.
3. Check that your model is available, then put its exact id in the config:
   ```bash
   curl -s https://api.ai.it.ufl.edu/v1/models -H "Authorization: Bearer $(cat ~/.secrets/navigator_key)"
   ```
4. Before running anything, from the repo root on a compute node:
   ```bash
   module load conda && conda activate /blue/eee6778/si.seenivasan/conda/envs/gnnrag
   ```

## Reproducing the baseline

```bash
python -m data.load_hotpot configs/baseline_plain.yaml     # 2000 train / 500 val, seed 42
python run_pipeline.py configs/baseline_plain.yaml         # dense search + reader, EM/F1 + retrieval metrics
python -m graph.run_graphs configs/baseline_plain.yaml     # passage graphs + stats
```

Results are appended to `results/results.jsonl` and `results/graph_stats.jsonl`, one row per run.

## Week 2: relevance-only GNN

```bash
bash scripts/week2.sh            # train graphs, GNN + MLP control on seeds 0-2, comparison table
sbatch slurm/week2.sbatch        # same thing as a batch job
python -m model.train configs/gnn_relevance.yaml --seed 0     # a single run
python -m eval.compare_week2 configs/gnn_relevance.yaml       # table + bootstrap + seed noise
```

Per-run summaries go to `results/week2.jsonl`, per-question val rankings to `results/week2/preds/`, checkpoints to `checkpoints/` (gitignored). The runbook is `WEEK2.md`.

## Notes

- LLM responses are cached in `cache/llm.jsonl`, so reruns are free but report about 0 generation latency. Take the latency figure from the first run.
- Graph stats and any model substitutions go in `NOTES.md`.
