#!/bin/bash
# Runs scripts/week2.sh as a batch job (use this instead of the tunnel if you want to walk away).
# Submit from the repo root:   sbatch slurm/week2.sbatch
# Each run is small (~1,800 graphs of 10 nodes), so one GPU for all runs in sequence is plenty.
# Don't turn this into an array of GPU jobs: the eee6778 QOS shares 8 GPUs with the whole class.
#SBATCH --job-name=week2-gnn
#SBATCH --account=eee6778
#SBATCH --qos=eee6778
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/eee6778/si.seenivasan/logs/week2_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load conda
conda activate /blue/eee6778/si.seenivasan/conda/envs/gnnrag
export HF_HOME=/blue/eee6778/si.seenivasan/cache/hf
export PYTHONUNBUFFERED=1
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv
echo "commit: $(git rev-parse --short HEAD)"
bash scripts/week2.sh
date
