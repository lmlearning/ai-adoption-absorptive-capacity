import sys
from pathlib import Path

# The repository's code directory is a script directory, not the stdlib code module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
