import importlib


def test_list_and_purge(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTO_ANALYZER_PROJECT_ROOT", str(tmp_path))
    mm = importlib.reload(importlib.import_module("crypto_analyzer.model_manager"))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_reg.joblib").touch()
    (models_dir / "model_reg.v1.pkl").touch()
    (models_dir / "model_clf.joblib").touch()

    files = mm.list_artifacts("model_reg", models_dir)
    assert {f.name for f in files} == {"model_reg.joblib", "model_reg.v1.pkl"}

    dry = mm.purge_artifacts("model_reg", models_dir, confirm=False)
    assert all(f.exists() for f in dry)

    removed = mm.purge_artifacts("model_reg", models_dir, confirm=True)
    assert {f.name for f in removed} == {"model_reg.joblib", "model_reg.v1.pkl"}
    assert not any((models_dir / n).exists() for n in ["model_reg.joblib", "model_reg.v1.pkl"])
    assert (models_dir / "model_clf.joblib").exists()
