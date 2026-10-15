import pytest

from scripts import run_knowledge_loop as loop
from scripts.knowledge_v2_components import PreservingTrialArm, nonblocking_consolidate
from scripts.run_knowledge_v2 import variant
from online_rl.evidence_router import EvidenceRouter


def test_variant_restores_every_patched_global_even_on_failure():
    before = loop.OnlineArm, loop.TaskFreeRouter, loop._consolidate, loop.save_checkpoint
    with pytest.raises(RuntimeError):
        with variant("evidence_router_nonblocking"):
            assert loop.OnlineArm is PreservingTrialArm
            assert loop.TaskFreeRouter is EvidenceRouter
            assert loop._consolidate is nonblocking_consolidate
            raise RuntimeError("injected")
    assert (loop.OnlineArm, loop.TaskFreeRouter, loop._consolidate, loop.save_checkpoint) == before
