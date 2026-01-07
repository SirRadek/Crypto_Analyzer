#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  set +u
  export CUDAARCHS_BACKUP="${CUDAARCHS_BACKUP:-}"
  source "$HOME/miniconda/etc/profile.d/conda.sh"
else
  echo "Miniconda not found at $HOME/miniconda/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda activate crypto-gpu
set -u
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:"$CONDA_PREFIX/lib":${LD_LIBRARY_PATH:-}

if [[ "${1:-}" == -* ]]; then
  python "$@"
else
  PYTHONPATH=src python -m "$@"
fi
