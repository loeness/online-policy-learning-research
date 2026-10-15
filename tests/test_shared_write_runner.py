"""Check real-factory isolation and complete shared endpoint restoration."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_shared_write import make_agent, save, restore_run, write_json


def test_factory_and_endpoint_restore(tmp_path):
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/shared_write.json").read_text())
    config.update(condition="oracle_active", seed=809)
    agent, slots, base = make_agent("oracle_active", 809, config)
    agent.core.install_shared_matrix(np.arange(18, dtype=np.float32).reshape(2, 9) * .01)
    agent.core.install_local_matrix(np.arange(162, dtype=np.float32).reshape(6, 27) * .001)
    observations = torch.tensor([[.1, .2, .8, .9, *([0.] * 8), slots["A2"] + 1],
                                 [.8, .5, .1, .4, *([0.] * 8), slots["B0"] + 1]])
    expected = agent.core.get_distribution(observations).distribution.probs.detach().clone()
    write_json(tmp_path / "config.json", config)
    sizes = save(agent, tmp_path, "final", config, slots, base)
    restored, other_slots, _ = restore_run(tmp_path)
    actual = restored.core.get_distribution(observations).distribution.probs.detach()
    assert other_slots == slots
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert sizes["actor_bytes"] == (tmp_path / "actor_final.bin").stat().st_size == 738
    assert sizes["whole_package_bytes"] <= config["whole_package_cap_bytes"]
    assert restored.core.source_basis.requires_grad is False


def test_independent_has_no_shared_parameters_and_fast_control_changes_only_local_lr():
    config = json.loads((ROOT / "configs/shared_write.json").read_text())
    regular, slots, _ = make_agent("independent", 809, config)
    fast, fast_slots, _ = make_agent("independent_fast", 809, config)
    assert not list(regular.core.shared_parameters()) and fast_slots == slots
    rates = {g["name"]: g["lr"] for g in regular.optimizer.param_groups}
    fast_rates = {g["name"]: g["lr"] for g in fast.optimizer.param_groups}
    assert rates["critic"] == fast_rates["critic"]
    assert fast_rates["local"] == config["fast_local_learning_rate"]
    for (name, a), (_, b) in zip(regular.core.named_parameters(), fast.core.named_parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
