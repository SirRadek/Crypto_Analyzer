#!/usr/bin/env bash
set -e

python main.py --task clf --horizon 120 --use_onchain --split_params '{"test_size":0.2}' --out_dir outputs/demo_clf
python main.py --task reg --horizon 120 --use_onchain --split_params '{"test_size":0.2}' --out_dir outputs/demo_reg
