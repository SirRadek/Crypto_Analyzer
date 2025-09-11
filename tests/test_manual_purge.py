from pathlib import Path
import importlib
import pytest


def test_list_and_purge(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CRYPTO_ANALYZER_PROJECT_ROOT", str(tmp_path))
    mm = importlib.reload(importlib.import_module("crypto_analyzer.model_manager"))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_reg.joblib").touch()
    (models_dir / "model_reg.v1.pkl").touch()
    (models_dir / "model_clf.joblib").touch()

    # subdirectories are not allowed
    subdir = models_dir / "sub"
    subdir.mkdir()
    with pytest.raises(ValueError):
        mm.list_artifacts("model_reg", subdir)

    files = mm.list_artifacts("model_reg", models_dir)
    assert {f.name for f in files} == {"model_reg.joblib", "model_reg.v1.pkl"}

    # CLI purge without confirm only lists files and returns 0
    code = mm.main(["--purge", "--stem", "model_reg", "--dir", str(models_dir)])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    assert {Path(line).name for line in out} == {"model_reg.joblib", "model_reg.v1.pkl"}
    assert (models_dir / "model_reg.joblib").exists()
    assert (models_dir / "model_reg.v1.pkl").exists()

    dry = mm.purge_artifacts("model_reg", models_dir, confirm=False)
    assert all(f.exists() for f in dry)

    removed = mm.purge_artifacts("model_reg", models_dir, confirm=True)
    assert {f.name for f in removed} == {"model_reg.joblib", "model_reg.v1.pkl"}
    assert not any((models_dir / n).exists() for n in ["model_reg.joblib", "model_reg.v1.pkl"])
    assert (models_dir / "model_clf.joblib").exists()
