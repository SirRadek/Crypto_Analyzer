import importlib.util


def test_no_cudf_installed() -> None:
    assert importlib.util.find_spec("cudf") is None
