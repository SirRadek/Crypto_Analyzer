#!/usr/bin/env bash
set -euo pipefail
MAIN_MODULE=${MAIN_MODULE:-crypto_analyzer}
MAIN_MODULE=$(python - <<'PY'
import importlib, sys
cand = ["crypto_analyzer","Crypto_analyzer"]
for name in cand:
    try:
        importlib.import_module(name)
        print(name); sys.exit(0)
    except Exception:
        pass
print(cand[0])
PY
)
export MAIN_MODULE
mkdir -p ml
python -X importtime -c "import ${MAIN_MODULE}" 2> ml/import.log
python -m tuna ml/import.log -o ml >/dev/null 2>&1 || true
python - <<'PY'
import json, pathlib
log = pathlib.Path('ml/import.log').read_text().splitlines()
pathlib.Path('ml/importtime.json').write_text(json.dumps(log))
PY
