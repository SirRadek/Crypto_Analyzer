import subprocess


def test_vulture_clean():
    res = subprocess.run(
        ["vulture", ".", "--exclude", "tests", "--min-confidence", "70"],
        capture_output=True,
        text=True,
    )
    assert res.stdout.strip() == ""
