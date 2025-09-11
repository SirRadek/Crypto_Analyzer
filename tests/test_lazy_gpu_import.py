import sys
import logging

from crypto_analyzer.inference import get_gpu_rf_or_none


def test_gpu_modules_not_imported_when_disabled():
    sys.modules.pop("cuml", None)
    sys.modules.pop("cudf", None)
    res = get_gpu_rf_or_none(use_gpu=False)
    assert res is None
    assert "cuml" not in sys.modules
    assert "cudf" not in sys.modules


def test_gpu_fallback_without_cuda(caplog):
    sys.modules.pop("cuml", None)
    sys.modules.pop("cudf", None)
    with caplog.at_level(logging.WARNING):
        res = get_gpu_rf_or_none(use_gpu=True)
    assert res is None
    assert "cuml" not in sys.modules
    assert any("GPU RF unavailable" in r.getMessage() for r in caplog.records)
