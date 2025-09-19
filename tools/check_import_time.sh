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
mkdir -p artifacts/import_time
python -X importtime -c "import ${MAIN_MODULE}" 2> artifacts/import_time/import.log
python -m tuna artifacts/import_time/import.log -o artifacts/import_time >/dev/null 2>&1 || true
python - <<'PY'
import json, pathlib
log = pathlib.Path('artifacts/import_time/import.log').read_text().splitlines()
pathlib.Path('artifacts/import_time/importtime.json').write_text(json.dumps(log))
PY
