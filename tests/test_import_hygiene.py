import importlib
import os
import sys

MAIN = os.environ.get("MAIN_MODULE", "crypto_analyzer")


def test_no_heavy_modules_on_import():
    importlib.invalidate_caches()
    importlib.import_module(MAIN)
    for m in ("cuml", "cudf", "matplotlib"):
        assert m not in sys.modules
