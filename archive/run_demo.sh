#!/usr/bin/env bash
set -e

PYTHONPATH=src python -m crypto_analyzer.cli --task clf --horizon 120 --use_onchain --split_params '{"test_size":0.2}' --out_dir outputs/demo_clf
# Legacy regression demo removed; classification CLI is the supported entry point.
