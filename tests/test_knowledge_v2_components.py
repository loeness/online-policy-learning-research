import numpy as np
import torch

from scripts import run_knowledge_loop as loop
from scripts.knowledge_v2_components import PreservingTrialArm


def test_allocating_one_slot_preserves_uncommitted_updates_in_other_slots():
    config = loop.load_base_config()
    agent = PreservingTrialArm("int8", 787, config)
    trial = loop._read_matrix(agent.core)
    trial[0].fill(.7)
    trial[3].fill(-.2)
    loop._install_matrix(agent.core, trial, shared=False)
    agent._reset_allocated_slot(3, evicted=False)
    actual = loop._read_matrix(agent.core)
    np.testing.assert_array_equal(actual[0], trial[0])
    assert not actual[3].any()
    assert agent.slot_generation[3] == 1
    assert agent.counts["slot_allocations"] == 1
    # The persisted package remains the old committed knowledge for row zero;
    # the local working update is independent until the next consolidation.
    assert not agent.compact.decode()[0].any()


def test_nonblocking_fallback_keeps_the_local_update_when_sharing_fails(monkeypatch):
    from scripts.knowledge_v2_components import nonblocking_consolidate
    agent = PreservingTrialArm("evidence_gate", 787, loop.load_base_config())
    trial = loop._read_matrix(agent.core)
    trial[0, 0] = .8
    loop._install_matrix(agent.core, trial, shared=False)
    result = nonblocking_consolidate(agent, [], cycle=1)
    assert result["decision"] == "independent_fallback"
    assert agent.compact.nbytes <= 320
    assert agent.committed_matrix[0, 0] > .79
    assert agent.counts["rejected"] == 0


def test_surrogate_uses_actual_behavior_probability_for_both_candidates(monkeypatch):
    from types import SimpleNamespace
    from scripts.knowledge_v2_components import behavior_relative_surrogate
    agent = SimpleNamespace(core=object(), committed_matrix=np.zeros((6, 27), np.float32),
                            slot_generation=np.zeros(6, dtype=np.int64))
    row = SimpleNamespace(slot=0, generation=0, obs=np.zeros(13), action=0,
                          log_prob=np.log(.5), advantage=1.)
    monkeypatch.setattr(loop, "_read_matrix", lambda core: np.zeros((6, 27), np.float32))
    def probabilities(agent, matrix, obs, *, restore):
        return np.asarray([[.45, .2, .2, .15] if matrix[0, 0] else [.4, .2, .2, .2]])
    monkeypatch.setattr(loop, "_matrix_probs", probabilities)
    score, count = behavior_relative_surrogate(agent, np.ones((6, 27), np.float32), [row])
    assert count == 1
    assert abs(score - (.45 / .5 - .4 / .5)) < 1e-9
