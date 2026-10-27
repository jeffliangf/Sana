"""Allow CPU service tests to import adjacent service modules independently."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
