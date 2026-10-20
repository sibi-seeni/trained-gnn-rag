#!/bin/bash
# One-time env creation on HiPerGator. Run INSIDE the VS Code tunnel terminal (compute node), not the login node.
set -euo pipefail
B=/blue/eee6778/si.seenivasan
ENV=$B/conda/envs/gnnrag
module load conda

if [ ! -d "$ENV" ]; then
  conda create -p "$ENV" python=3.11 -y
fi
conda activate "$ENV"
export PIP_NO_CACHE_DIR=1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric datasets sentence-transformers spacy openai pyyaml pandas tqdm ipykernel
python -m spacy download en_core_web_sm
python -m ipykernel install --user --name gnnrag --display-name gnnrag
pip freeze > requirements.lock.txt
conda clean -a -y
echo "env ready at $ENV"
