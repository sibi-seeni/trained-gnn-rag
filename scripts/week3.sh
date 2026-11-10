#!/bin/bash
# Weeks 3 + 5, start to finish (CPU is fine; ~1-2 h on 4 cores). Run from the repo root with the env active.
#   bash scripts/week3.sh
# Needs data/processed/{train,val}.jsonl and graphs_{train,val}.pt (scripts/week2.sh builds them) and the Week 2
# checkpoints checkpoints/gnn_relevance_s{0,1,2}.pt (version A); trains them if missing.
set -euo pipefail
CFG=configs/trust_stage.yaml
SEEDS="${SEEDS:-0 1 2}"

echo "== 1. plant fakes (clean sets are rebuilt with agree/contradict edges by the same builder)"
for sp in train val; do
  [ -f data/processed/graphs_${sp}_none.pt ] || python -m attacks.plant $CFG --split $sp --attack none --check
done
[ -f data/processed/graphs_train_edit.pt ] || python -m attacks.plant $CFG --split train --attack edit
for at in edit connected connected_q; do
  [ -f data/processed/graphs_val_${at}.pt ] || python -m attacks.plant $CFG --split val --attack $at
done

echo "== 2. version A (Week 2 model) if missing"
for s in $SEEDS; do
  [ -f checkpoints/gnn_relevance_s$s.pt ] || python -m model.train configs/gnn_relevance.yaml --seed $s
done

echo "== 3. trust stage (B), C, and the exploratory control"
for m in trust_stage gnn_trust_mp gnn_relevance_adv; do
  for s in $SEEDS; do
    python -m model.train_trust configs/$m.yaml --seed $s
  done
done

echo "== 4. comparison + pre-registered rule"
python -m eval.week3
echo "done: results/week3/compare.md"
