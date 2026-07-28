#!/usr/bin/env bash
# One-shot setup for a fresh GPU host. Clones the repo, installs deps, downloads model weights.
#
# Usage:
#   ./setup_host.sh <hf-model-repo> [<hf-model-repo> ...]
# Example:
#   ./setup_host.sh meta-llama/Llama-2-13b-hf
#
# Requires HF_TOKEN set, or run `huggingface-cli login` first (weights are gated).

set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hf-model-repo> [<hf-model-repo> ...]"
    echo "Example: $0 meta-llama/Llama-2-13b-hf"
    exit 1
fi

pip install -r /workspace/geometry-of-morality/requirements.txt

echo "export HF_HOME=/workspace/.hf_home" >> ~/.bashrc
export HF_HOME=/workspace/.hf_home

for repo in "$@"; do
    huggingface-cli download "$repo"
done

echo "Done. Run: python generate_acts.py --model <model> --layers <layers> --datasets <datasets> --device cuda:0. in /workspace/geometry-of-morality to generate acts."
