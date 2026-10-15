"""Restore a V2 checkpoint using its recorded algorithm factory."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import run_knowledge_loop as loop
from scripts.run_knowledge_v2 import variant


def restore_v2(folder):
    folder = Path(folder)
    runtime = json.loads((folder / "runtime_patches.json").read_text(encoding="utf-8"))
    condition = runtime["condition"]
    with variant(condition):
        agent = loop.restore_agent_checkpoint(folder / "checkpoint_latest.pt", loop.load_base_config())
    return agent
