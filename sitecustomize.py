import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for candidate in (root, root / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
