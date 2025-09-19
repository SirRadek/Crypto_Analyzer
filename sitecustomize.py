import sys
from pathlib import Path

from dotenv import load_dotenv

root = Path(__file__).resolve().parent
load_dotenv(root / ".env", override=False)
for candidate in (root, root / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
