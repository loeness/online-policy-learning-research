import json

import numpy as np
import torch

from scripts import run_knowledge_loop as loop
from scripts.restore_knowledge_v2 import restore_v2
from online_rl.evidence_router import EvidenceRouter


def test_evidence_router_checkpoints_restore_through_their_recorded_factory():
    before = loop.TaskFreeRouter
    root = loop.ROOT / "outputs/knowledge_v2_pilot/seed_787"
    for condition in ("evidence_router_int8", "evidence_router_old_gate", "evidence_router_nonblocking"):
        folder = root / condition
        payload = torch.load(folder / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        restored = restore_v2(folder)
        assert isinstance(restored.router, EvidenceRouter)
        assert restored.router.state_dict() == payload["router"]
        assert restored.counts == payload["counts"]
        np.testing.assert_array_equal(restored.committed_matrix, restored.compact.decode())
        assert loop.TaskFreeRouter is before
