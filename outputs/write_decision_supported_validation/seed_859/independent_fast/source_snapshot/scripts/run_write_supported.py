"""Run a separately frozen support-aware revision of the Step 1 protocol."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_write_decision as runner

runner.CONFIG_PATH = ROOT / "configs/write_decision_supported.json"
runner.SOURCE_FILES = (*runner.SOURCE_FILES,
                       Path("configs/write_decision_supported.json"),
                       Path("scripts/run_write_supported.py"))

if __name__ == "__main__":
    runner.main()
