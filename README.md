# trained-gnn-rag

Trust-aware graph retrieval for multi-hop QA (HotpotQA). Week 1 builds the plain dense-search baseline: load → retrieve → generate → score.

Code is edited on the laptop, synced to HiPerGator, and run as Slurm jobs. Answer generation goes through UF NaviGator (OpenAI-compatible API).

## Layout

```
configs/    one YAML per experiment
data/       HotpotQA loading + processed splits (processed/ is gitignored)
graph/      passage graph construction + stats
retrieval/  embeddings + dense ranking
model/      GNN (Week 2)
attacks/    poisoning (later)
generate/   NaviGator client + prompt
eval/       EM/F1 + retrieval metrics
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

## Notes

- LLM responses are cached in `cache/llm.jsonl`, so reruns are free but report about 0 generation latency. Take the latency figure from the first run.
- Graph stats and any model substitutions go in `NOTES.md`.
