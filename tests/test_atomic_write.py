import importlib
import threading


def test_atomic_write(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTO_ANALYZER_PROJECT_ROOT", str(tmp_path))
    mm = importlib.reload(importlib.import_module("crypto_analyzer.model_manager"))

    target = tmp_path / "artifact.bin"

    def writer():
        mm.atomic_write(target, b"hello")

    t = threading.Thread(target=writer)
    t.start()
    t.join()

    assert target.exists()
    assert target.read_bytes() == b"hello"
    # no temporary files remain
    assert list(tmp_path.glob("artifact.bin*")) == [target]
