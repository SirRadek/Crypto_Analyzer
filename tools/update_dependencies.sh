#!/usr/bin/env bash
set -euo pipefail

if ! command -v pip-compile >/dev/null 2>&1; then
  echo "pip-compile not found. Install pip-tools via 'pip install pip-tools'." >&2
  exit 1
fi

pip-compile pyproject.toml --extra dev --output-file requirements.txt "$@"
