#!/bin/bash
# Books a B200 node and opens a VS Code tunnel into it.
# Submit from a HiPerGator terminal:   sbatch /blue/eee6778/si.seenivasan/gnn-rag/slurm/vscode_b200.sbatch
#SBATCH --job-name=vscode-b200
#SBATCH --account=eee6778
#SBATCH --qos=eee6778
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --output=/blue/eee6778/si.seenivasan/logs/vscode_%j.out

hostname; date
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}
export HF_HOME=/blue/eee6778/si.seenivasan/cache/hf
export PIP_NO_CACHE_DIR=1
nvidia-smi --query-gpu=name,memory.total --format=csv

~/bin/code tunnel --name hipergator --accept-server-license-terms
