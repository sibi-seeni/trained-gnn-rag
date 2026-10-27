# trained-gnn-rag

Trust-aware graph retrieval for multi-hop QA (HotpotQA). Week 1 builds the plain dense-search baseline: load → retrieve → generate → score. Week 2 trains a relevance-only GATv2 ranker over passage graphs (see `WEEK2.md`); the rule for the make-or-break trust test is fixed in `PREREGISTRATION.md`.

Code is edited on the laptop, synced to HiPerGator, and run as Slurm jobs. Answer generation goes through UF NaviGator (OpenAI-compatible API).

## Layout

```
configs/    one YAML per experiment (a config can inherit with `base:`)
common/     config loading
data/       HotpotQA loading + processed splits (processed/ is gitignored)
graph/      passage graph construction + stats + GNN-ready datasets
retrieval/  embeddings, dense ranking, trained-ranker wrapper
model/      GATv2 ranker + training
attacks/    poisoning (later)
generate/   NaviGator client + prompt
eval/       EM/F1, retrieval metrics, second-fact metrics, Week 2 comparison
scripts/    one script per week of runs
slurm/      sbatch scripts + env setup
hpg.sh      laptop-side helper
```

## One-time setup

1. Add an `hpg` host alias to `~/.ssh/config` (pointing at `hpg.rc.ufl.edu`, with ControlMaster so you do Duo once).
2. Set `REMOTE_DIR` in `hpg.sh`.
3. Save your NaviGator key on HPG at `~/.secrets/navigator_key` (chmod 600). Never commit it.
4. Check that your model is available, then put its exact id in the config:
   ```bash
   ssh hpg 'curl -s https://api.ai.it.ufl.edu/v1/models -H "Authorization: Bearer $(cat ~/.secrets/navigator_key)"'
   ```
5. Run the setup:
   ```bash
   chmod +x hpg.sh
   ./hpg.sh setup
   ```

## Running

```bash
./hpg.sh run                     # baseline on CPU (configs/baseline_plain.yaml)
./hpg.sh run configs/X.yaml      # any other experiment
./hpg.sh gpu configs/X.yaml      # B200 job
./hpg.sh status | logs | pull | cancel | shell
```

Results are appended to `results/results.jsonl`, one row per run.

## Reproducing the baseline

```bash
./hpg.sh run configs/baseline_plain.yaml
```

This reports answer EM/F1, recall@k, both-gold recall, MRR, and latency on 100 validation questions. Exact package versions are in `requirements.lock.txt`.

## Week 2: relevance-only GNN

```bash
bash scripts/week2.sh            # train graphs, GNN + MLP control on seeds 0-2, comparison table
sbatch slurm/week2.sbatch        # same thing as a batch job
python -m model.train configs/gnn_relevance.yaml --seed 0     # a single run
python -m eval.compare_week2 configs/gnn_relevance.yaml       # table + bootstrap + seed noise
```

Per-run summaries go to `results/week2.jsonl`, per-question val rankings to `results/week2/preds/`, checkpoints to `checkpoints/` (gitignored).

## Notes

- LLM responses are cached in `cache/llm.jsonl`, so reruns are free but report about 0 generation latency. Take the latency figure from the first run.
- Graph stats and any model substitutions go in `NOTES.md`.
