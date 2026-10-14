"""Integration checks for compact residuals in the original PPO training loop."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from online_rl.compression import PackedResidualPolicy
from online_rl.compression_experiment import PackedSB3Adapter, actor_hash
from online_rl.controls import load_terminal_run, clear_router_working_memory
from online_rl.experiment import StreamEnv, make_model
from stable_baselines3.common.vec_env import DummyVecEnv


def test_compact_policy_can_collect_rollouts_and_train_original_ppo(tmp_path):
    import csv
    from online_rl.experiment import FIELDS
    torch.set_num_threads(1)
    cfg = {"learning_rate": .01, "ent_coef": .06, "bout_steps": 4096}
    source, router = load_terminal_run("outputs/validated/B_history_keep_s41", cfg)
    core = PackedResidualPolicy.from_dense(source.policy, [1, 2], retain_fraction=.5)
    optimizer = core.make_optimizer(source.policy.optimizer)
    source.env.close()
    clear_router_working_memory(router)
    base = actor_hash(core)
    start_steps = optimizer.step_count
    with (tmp_path / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        stream = StreamEnv("test", "history_keep", 999, router, cfg, writer, ["cw"])
        vec = DummyVecEnv([lambda: stream])
        model = make_model(vec, {**cfg, "seed": 999})
        model.policy = PackedSB3Adapter(core, optimizer)
        stream.model = model
        model.learn(total_timesteps=512)
        assert stream.total_steps == 512
        assert optimizer.step_count > start_steps
        assert actor_hash(core) == base
        for parameter, state in optimizer.state.items():
            if "exp_avg" in state:
                assert state["exp_avg"].shape == parameter.shape
        assert core.storage_report()["retained_coordinates"] == 260
        assert not any(k.startswith("residuals.") or k == "path_masks" for k in core.state_dict())
        vec.close()
