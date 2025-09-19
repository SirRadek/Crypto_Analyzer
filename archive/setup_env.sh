#!/usr/bin/env bash
set -euo pipefail

echo "Python version: $(python --version 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA detected"
  nvidia-smi || true
else
  echo "CUDA not detected"
fi

python - <<'PY'
import importlib, json

def log_pkg(name):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"{name} {ver}")
        if name == 'xgboost':
            info = getattr(mod, 'build_info', lambda: {})()
            cuda = info.get('USE_CUDA') or info.get('CUDA')
            print(f"  CUDA support: {cuda}")
    except Exception as exc:  # pragma: no cover
        print(f"{name} not available: {exc}")

for pkg in [
    'numpy',
    'pandas',
    'scipy',
    'sklearn',
    'xgboost',
    'shap',
    'numba',
    'requests',
    'pydantic',
    'dateutil',
    'tqdm',
    'pyarrow',
    'matplotlib',
]:
    log_pkg(pkg)
PY

