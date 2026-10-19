"""SB3 PPO policy with fixed foundation and persistent path residuals.

The actor sees four observable navigation coordinates. The critic additionally
sees a causal history summary. The final observation coordinate is a route ID
computed by the agent, never the simulator's hidden context ID.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

HISTORY_DIM = 8
OBS_DIM = 4 + HISTORY_DIM + 1
MAX_NODES = 8


class LeafAdam(torch.optim.Adam):
    """Do not advance momentum for a residual absent from the current loss."""

    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if getattr(parameter, "tree_residual", False):
                    if parameter.grad is not None and not torch.count_nonzero(parameter.grad):
                        parameter.grad = None
        result = super().step(closure)
        self.step_count = getattr(self, "step_count", 0) + 1
        return result


class SplitNetwork(nn.Module):
    def __init__(self, features_dim: int, width: int = 64):
        super().__init__()
        self.latent_dim_pi = width
        self.latent_dim_vf = width
        self.policy_net = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(4 + HISTORY_DIM, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())

    def forward_actor(self, features):
        return self.policy_net(features[:, :4])

    def forward_critic(self, features):
        return self.value_net(features[:, :4 + HISTORY_DIM])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class TreePolicy(ActorCriticPolicy):
    def __init__(self, *args, max_nodes=MAX_NODES, **kwargs):
        self.max_nodes = max_nodes
        kwargs["optimizer_class"] = LeafAdam
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SplitNetwork(self.features_dim)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.residuals = nn.ParameterList([
            nn.Parameter(torch.zeros(self.action_space.n, self.mlp_extractor.latent_dim_pi + 1))
            for _ in range(self.max_nodes)
        ])
        for parameter in self.residuals:
            parameter.tree_residual = True
        masks = torch.eye(self.max_nodes)
        masks[0, :] = 0
        masks[:, 0] = 0
        self.register_buffer("path_masks", masks)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def distribution(self, obs, latent_pi):
        ids = obs[:, -1].round().long().clamp(0, self.max_nodes - 1)
        phi = torch.cat([latent_pi, torch.ones_like(latent_pi[:, :1])], dim=1)
        weights = torch.stack(tuple(self.residuals), dim=0)
        correction = torch.einsum("bn,nac,bc->ba", self.path_masks[ids], weights, phi)
        logits = self.action_net(latent_pi) + correction
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self.distribution(obs, pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(vf), distribution.log_prob(actions)

    def get_distribution(self, obs):
        latent = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self.distribution(obs, latent)

    def evaluate_actions(self, obs, actions):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self.distribution(obs, pi)
        return self.value_net(vf), distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs):
        return self.value_net(self.mlp_extractor.forward_critic(self.extract_features(obs)))

    def configure_learning(self, mode: str, learning_rate: float):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if mode == "full":
            for parameter in self.mlp_extractor.policy_net.parameters():
                parameter.requires_grad_(True)
            for parameter in self.action_net.parameters():
                parameter.requires_grad_(True)
        elif mode != "frozen":
            for parameter in self.residuals[1:]:
                parameter.requires_grad_(True)
        if mode != "frozen":
            for parameter in self.mlp_extractor.value_net.parameters():
                parameter.requires_grad_(True)
            for parameter in self.value_net.parameters():
                parameter.requires_grad_(True)
        trainable = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = LeafAdam(trainable or [self.value_net.weight], lr=learning_rate, eps=1e-5)

    def zero_node(self, node: int):
        if node == 0:
            return
        parameter = self.residuals[node]
        with torch.no_grad():
            parameter.zero_()
        self.optimizer.state.pop(parameter, None)

    def set_path(self, node: int, ancestors: list[int]):
        with torch.no_grad():
            self.path_masks[node].zero_()
            if node != 0:
                self.path_masks[node, node] = 1
            for ancestor in ancestors:
                if ancestor != 0:
                    self.path_masks[node, ancestor] = 1


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def capacity(model, router, rollout=True):
    policy_bytes = tensor_bytes(model.policy.state_dict())
    optimizer_bytes = tensor_bytes(model.policy.optimizer.state_dict())
    rollout_bytes = 0
    if rollout and hasattr(model, "rollout_buffer"):
        rollout_bytes = sum(v.nbytes for v in vars(model.rollout_buffer).values() if isinstance(v, np.ndarray))
    packed = io.BytesIO()
    torch.save({"policy": model.policy.state_dict(), "optimizer": model.policy.optimizer.state_dict(),
                "router": router.snapshot(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": __import__("random").getstate(),
                "last_observation": getattr(model, "_last_obs", None),
                "last_episode_starts": getattr(model, "_last_episode_starts", None),
                "optimizer_step_count": getattr(model.policy.optimizer, "step_count", 0)}, packed)
    serialized = packed.tell()
    router_bytes = len(__import__("json").dumps(router.snapshot()).encode("utf-8"))
    return {"policy_bytes": policy_bytes, "optimizer_bytes": optimizer_bytes,
            "router_bytes": router_bytes, "rollout_bytes": rollout_bytes,
            "serialized_core_bytes": serialized,
            "agent_state_bytes": serialized + rollout_bytes}
