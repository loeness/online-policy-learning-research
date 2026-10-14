import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.c1_model_experiment import (_foundation_actor, _epsilon_mix, run_trajectory, restore_controller, restore_action_generator)
from online_rl.c1_strong_models import C1ModelController

def test_actual_transition_restore_and_common_mixture(tmp_path):
    config = json.loads(Path("configs/c1.json").read_text())
    actor = _foundation_actor(config)
    controller = C1ModelController("window_assignment")
    result = run_trajectory(actor=actor, controller=controller, router=None,
        schedule=[("seen", "A0", 32), ("heldout", "B2", 32)], seed=811,
        output=tmp_path, run_kind="test")
    assert result["online_steps"] == result["statistical_updates"] == 64
    checkpoint = torch.load(tmp_path / "checkpoint_heldout.pt", weights_only=False)
    restored = restore_controller(checkpoint)
    obs = np.asarray(result["last_observation"])
    base = np.array([.7, .1, .1, .1])
    np.testing.assert_array_equal(controller.action_probabilities(obs, base), restored.action_probabilities(obs, base))
    assert restore_action_generator(checkpoint).bit_generator.state == checkpoint["rng"]["action_generator"]
    np.testing.assert_allclose(_epsilon_mix(base, .05), .95 * base + .0125)
    assert result["max_serialized_agent_bytes"] < 188416
