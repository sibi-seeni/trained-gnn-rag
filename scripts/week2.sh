#!/bin/bash
# Week 2, start to finish. Run from the repo root on a compute node (VS Code tunnel terminal or sbatch).
#   bash scripts/week2.sh                    # GNN + MLP control, seeds 0 1 2
#   EXTRA=gnn_relevance_qnode bash scripts/week2.sh   # also the question-node variant
#   SEEDS="0" bash scripts/week2.sh          # quick single-seed pass
set -euo pipefail
BASE=configs/baseline_plain.yaml
SEEDS="${SEEDS:-0 1 2}"
MODELS="gnn_relevance mlp_relevance ${EXTRA:-}"

echo "== 1. graphs (val exists from Week 1; train is new)"
[ -f data/processed/graphs_train.pt ] || python -m graph.run_graphs $BASE --split train
[ -f data/processed/graphs_val.pt ]   || python -m graph.run_graphs $BASE --split val

echo "== 2. dataset check (feature dims, hop2 rule counts)"
python -m graph.dataset configs/gnn_relevance.yaml

echo "== 3. training: $MODELS x seeds $SEEDS"
for m in $MODELS; do
  for s in $SEEDS; do
    python -m model.train configs/$m.yaml --seed $s
  done
done

echo "== 4. comparison"
mkdir -p results/week2
python -m eval.compare_week2 configs/gnn_relevance.yaml | tee results/week2/compare.md
echo "done: paste results/week2/compare.md into NOTES.md, then fill sigma_seed in PREREGISTRATION.md"
