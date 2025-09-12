import subprocess, sys, shutil, pytest
import sys
from pathlib import Path

@pytest.mark.skipif(shutil.which("vulture") is None, reason="vulture not installed")
def test_vulture_clean():
    cmd = [
        sys.executable, "-m", "vulture",
        ".",
        "--exclude", "tests,.venv,venv,build,dist",
        "--min-confidence", "90",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Vulture vrací 0 i s nálezy; filtruj šum a whitelist řádek.
    lines = [
        ln for ln in res.stdout.splitlines()
        if ln.strip()
        and not ln.startswith("Checking")
        and not ln.startswith("vulture:")
    ]
    assert res.returncode == 0
    assert not lines, "Vulture findings:\n" + "\n".join(lines)