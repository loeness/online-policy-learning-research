import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.c1_supplement_experiment import run_arm, _restore_payload, _make_core

def test_optimizer_count_is_one_per_leaf_update_after_adam_is_wrapped():
    from online_rl.policy import LeafAdam
    from online_rl.c1_supplement_experiment import _count_optimizer_steps
    torch.optim.Adam([torch.nn.Parameter(torch.ones(1))])
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = LeafAdam([parameter])
    optimizer.step_count = 0
    counts = {"steps": 0}
    handle = _count_optimizer_steps(optimizer, counts)
    for _ in range(3):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    handle.remove()
    assert counts["steps"] == optimizer.step_count == 3

@pytest.mark.parametrize("method,control", [("tree", "none"), ("grouped_dictionary", "none"), ("flat", "none"), ("dictionary", "none"), ("full", "none"), ("full", "ewc"), ("tree", "bc")])
def test_real_stream_phase_restore_and_auxiliary_accounting(method, control, tmp_path):
    config = json.loads(Path("configs/c1_supplement.json").read_text())
    base = json.loads(Path(config["base_config"]).read_text())
    base["seen_sequence"] = ["A0", "B0", "A1", "B1"]
    base["history_sequence"] = ["B2", "A0", "B1", "A2"]
    config["validation_schedule"] = {"seen_bout_steps": 64, "heldout_bout_steps": 128, "revisit_bout_steps": 64, "history_bout_steps": 64, "eval_steps": 16, "cold_eval_steps": 16}
    result = run_arm(supplement=config, base=base, output=tmp_path, seed=801, method=method, tier_name="high", tier=base["tiers"]["high"], params={"learning_rate": .003, "ent_coef": .06}, stage_name="integration_smoke", control=control, coefficient=1, validation=True)
    assert result["status"] == "complete", result.get("error")
    assert result["training_interactions"] == 1024
    assert result["actual_optimizer_steps"] > 0
    assert result["base_actor_unchanged"] == (method != "full")
    restored = _restore_payload(tmp_path / "history/agent.pt")
    assert type(restored["policy"]).__name__
    assert len((tmp_path / "history/steps.csv").read_text().splitlines()) == 257
    if control == "bc":
        assert result["continual_state"]["behavior_reservoir_bytes"] == 8704
        assert restored["continual_state"]["behavior_reservoir"]["seen"] == 1024
    if control == "ewc":
        assert result["continual_state"]["ewc_fisher_bytes"] > 0
        assert restored["continual_state"]["online_ewc"] is not None
