"""Full fine tuning and bounded replay controls for C1.

These controls copy the common C1 foundation exactly once.  The conditioned
policy adds a zero initialized 64-value shared hidden-layer bias for each of
the six endpoint routes; it does not keep a separate actor per context. The
replay control is PPO with bounded policy behavior cloning, inspired by (but
not a reproduction of) CLEAR.
"""
from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.preprocessing import preprocess_obs
from torch import nn
from torch.nn import functional as F

from .policy import TreePolicy


_FORMAT_VERSION = 1
_ARCHITECTURE = "C1FullFineTunePolicy-v1"
_OBS_DIM = 13
_ACTION_COUNT = 4
_ENDPOINTS = tuple(range(1, 7))


def _as_tensor_bytes(values: Any) -> int:
    if torch.is_tensor(values):
        return int(values.numel() * values.element_size())
    if isinstance(values, np.ndarray):
        return int(values.nbytes)
    if isinstance(values, Mapping):
        return sum(_as_tensor_bytes(value) for value in values.values())
    if isinstance(values, (tuple, list)):
        return sum(_as_tensor_bytes(value) for value in values)
    return 0


class C1FullFineTunePolicy(nn.Module):
    """A full-parameter PPO actor-critic initialized from the C1 foundation.

    ``route_conditioned=True`` adds a trainable 6 by 64 bias to the first
    hidden preactivation of the original shared actor MLP. This is equivalent
    to adding six task one-hot input columns to the first linear layer. The
    bias starts at zero, so the initial actor is exactly the foundation for
    every route. Route 0 uses no shift. Parent routes 7 and 8 use the mean
    embedding of currently known endpoints in their family. With
    ``route_conditioned=False``, the actor ignores route IDs entirely.

    Both the actor and critic are trainable.  The critic retains the original
    12-coordinate input (navigation coordinates plus eight history values).
    The actor's context is the explicit route coordinate only; this is an
    oracle-compatible upper-control when the caller supplies oracle routes.
    """

    def __init__(self, *, epsilon: float = 0.05, temperature: float = 2.0) -> None:
        super().__init__()
        self._epsilon = self._validate_epsilon(epsilon)
        self._temperature = self._validate_temperature(temperature)
        # Instances are populated by from_reference/from_spec.

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        value = float(epsilon)
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("epsilon must be finite and in [0, 1)")
        return value

    @staticmethod
    def _validate_temperature(temperature: float) -> float:
        value = float(temperature)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("temperature must be finite and positive")
        return value

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        route_conditioned: bool = True,
        band_conditioned: bool = True,
        epsilon: float = 0.05,
        temperature: float = 2.0,
    ) -> "C1FullFineTunePolicy":
        """Copy a trained C1-compatible ``TreePolicy`` without changing W0."""
        if not isinstance(reference, TreePolicy):
            raise TypeError("reference must be an online_rl.policy.TreePolicy")
        if tuple(reference.observation_space.shape) != (_OBS_DIM,):
            raise ValueError("C1 full fine tuning requires a 13 value observation")
        if not isinstance(reference.action_space, spaces.Discrete) or int(reference.action_space.n) != _ACTION_COUNT:
            raise ValueError("C1 full fine tuning requires four discrete actions")
        if int(reference.mlp_extractor.latent_dim_pi) != 64:
            raise ValueError("C1 full fine tuning requires 64 actor features")

        result = cls(epsilon=epsilon, temperature=temperature)
        result.features_extractor = copy.deepcopy(reference.features_extractor)
        result.mlp_extractor = copy.deepcopy(reference.mlp_extractor)
        result.action_net = copy.deepcopy(reference.action_net)
        result.value_net = copy.deepcopy(reference.value_net)
        result.action_dist = make_proba_distribution(reference.action_space)
        result.observation_space = copy.deepcopy(reference.observation_space)
        result.action_space = copy.deepcopy(reference.action_space)
        result.normalize_images = bool(getattr(reference, "normalize_images", False))
        result.features_dim = int(reference.features_dim)
        result.route_conditioned = bool(route_conditioned)
        result.band_conditioned = bool(band_conditioned)
        like = next(result.mlp_extractor.policy_net.parameters())
        if result.band_conditioned:
            result.band_embedding = nn.Parameter(torch.zeros((3, 64), dtype=like.dtype, device=like.device))
        else:
            result.register_parameter("band_embedding", None)
        if result.route_conditioned:
            like = next(result.mlp_extractor.policy_net.parameters())
            result.route_embedding = nn.Parameter(torch.zeros((6, 64), dtype=like.dtype, device=like.device))
            # Initial C1 training exposes the first two endpoints in each
            # family. This mask prevents parent routes from averaging in the
            # zero rows reserved for held-out endpoints.
            known = torch.tensor([True, True, False, True, True, False], dtype=torch.bool, device=like.device)
            result.register_buffer("known_endpoint_mask", known)
        else:
            result.register_parameter("route_embedding", None)
            result.register_buffer("known_endpoint_mask", torch.zeros(6, dtype=torch.bool))
        for parameter in result.parameters():
            parameter.requires_grad_(True)
        return result

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def parameter_update_scope(self) -> str:
        return "full_actor_critic" if self.route_conditioned else "full_actor_critic_plain"

    def set_known_endpoints(self, endpoints: list[int] | tuple[int, ...] | set[int]) -> None:
        """Set which endpoint embeddings contribute to parent-route means."""
        if not self.route_conditioned:
            raise ValueError("plain full fine tuning has no route embeddings")
        selected = set()
        for endpoint in endpoints:
            if isinstance(endpoint, bool) or not isinstance(endpoint, (int, np.integer)) or int(endpoint) not in _ENDPOINTS:
                raise ValueError("known endpoints must be integer IDs one through six")
            selected.add(int(endpoint))
        with torch.no_grad():
            self.known_endpoint_mask.zero_()
            if selected:
                indices = torch.as_tensor(sorted(selected), dtype=torch.long, device=self.known_endpoint_mask.device) - 1
                self.known_endpoint_mask[indices] = True

    def initialize_heldout(self, endpoint: int) -> None:
        """Initialize an unseen family endpoint from its two known siblings."""
        if not self.route_conditioned:
            raise ValueError("plain full fine tuning has no route embeddings")
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, np.integer)) or int(endpoint) not in _ENDPOINTS:
            raise ValueError("endpoint must be an integer ID one through six")
        endpoint = int(endpoint)
        index = endpoint - 1
        if bool(self.known_endpoint_mask[index]):
            raise ValueError("endpoint is already marked as known")
        siblings = range(0, 3) if endpoint <= 3 else range(3, 6)
        sibling_indices = [sibling for sibling in siblings if sibling != index and bool(self.known_endpoint_mask[sibling])]
        if not sibling_indices:
            raise ValueError("held-out endpoint has no known siblings to initialize from")
        with torch.no_grad():
            self.route_embedding[index].copy_(self.route_embedding[sibling_indices].mean(dim=0))
            self.known_endpoint_mask[index] = True

    def _as_batch(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        reference = next(self.parameters())
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=reference.dtype, device=reference.device)
        else:
            obs = obs.to(device=reference.device, dtype=reference.dtype)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[1] != _OBS_DIM:
            raise ValueError("C1 policy observations must have shape (batch, 13)")
        return obs

    def extract_features(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        obs = self._as_batch(obs)
        preprocessed = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return self.features_extractor(preprocessed)

    def _route_shift(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.route_conditioned:
            return obs.new_zeros((obs.shape[0], 64))
        route_ids = obs[:, -1].round().long().clamp(0, 8)
        embedding = self.route_embedding
        known = self.known_endpoint_mask.to(dtype=embedding.dtype)
        family_a = (embedding[:3] * known[:3, None]).sum(dim=0) / known[:3].sum().clamp_min(1.0)
        family_b = (embedding[3:] * known[3:, None]).sum(dim=0) / known[3:].sum().clamp_min(1.0)
        route_table = torch.cat((embedding.new_zeros((1, 64)), embedding, family_a[None], family_b[None]), dim=0)
        return route_table.index_select(0, route_ids)

    def actor_logits(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Return unscaled action logits; useful for initialization checks."""
        obs = self._as_batch(obs)
        features = self.extract_features(obs)
        return self._actor_logits_from_features(obs, features)

    def _actor_logits_from_features(self, obs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        first_layer = self.mlp_extractor.policy_net[0]
        hidden_pre_activation = F.linear(features[:, :4], first_layer.weight, first_layer.bias)
        hidden_pre_activation = hidden_pre_activation + self._route_shift(obs)
        if self.band_embedding is not None:
            band = (obs[:, 0] * 8).round().long().clamp(0, 8) // 3
            hidden_pre_activation = hidden_pre_activation + self.band_embedding[band]
        latent_pi = self.mlp_extractor.policy_net[1:](hidden_pre_activation)
        return self.action_net(latent_pi)

    def _distribution_from_probs(self, probabilities: torch.Tensor):
        self.action_dist.distribution = torch.distributions.Categorical(probs=probabilities)
        return self.action_dist

    def get_distribution(self, obs: torch.Tensor | np.ndarray):
        obs = self._as_batch(obs)
        logits = self._actor_logits_from_features(obs, self.extract_features(obs))
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        if self.epsilon:
            probabilities = (1.0 - self.epsilon) * probabilities + self.epsilon / _ACTION_COUNT
        return self._distribution_from_probs(probabilities)

    def _value(self, features: torch.Tensor) -> torch.Tensor:
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def forward(self, obs, deterministic: bool = False):
        obs = self._as_batch(obs)
        features = self.extract_features(obs)
        logits = self._actor_logits_from_features(obs, features)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        if self.epsilon:
            probabilities = (1.0 - self.epsilon) * probabilities + self.epsilon / _ACTION_COUNT
        distribution = self._distribution_from_probs(probabilities)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self._value(features), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        obs = self._as_batch(obs)
        features = self.extract_features(obs)
        logits = self._actor_logits_from_features(obs, features)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        if self.epsilon:
            probabilities = (1.0 - self.epsilon) * probabilities + self.epsilon / _ACTION_COUNT
        distribution = self._distribution_from_probs(probabilities)
        actions = actions.to(device=obs.device, dtype=torch.long).flatten()
        return self._value(features), distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs):
        return self._value(self.extract_features(obs))

    def actor_named_parameters(self) -> dict[str, nn.Parameter]:
        result = {
            name: parameter
            for name, parameter in self.named_parameters()
            if name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net.")
        }
        if self.route_embedding is not None:
            result["route_embedding"] = self.route_embedding
        if self.band_embedding is not None:
            result["band_embedding"] = self.band_embedding
        return result

    def optimizer_manifest(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        names = {id(parameter): name for name, parameter in self.named_parameters()}
        groups = []
        for group in optimizer.param_groups:
            param_names = []
            for parameter in group["params"]:
                if id(parameter) not in names:
                    raise ValueError("optimizer contains a parameter outside this policy")
                param_names.append(names[id(parameter)])
            groups.append({
                "param_names": param_names,
                "options": {key: copy.deepcopy(value) for key, value in group.items() if key != "params"},
            })
        state = {}
        for parameter, values in optimizer.state.items():
            name = names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer state contains a parameter outside this policy")
            state[name] = {
                key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in values.items()
            }
        name = "Adam" if isinstance(optimizer, torch.optim.Adam) else "SGD" if isinstance(optimizer, torch.optim.SGD) else None
        if name is None:
            raise ValueError("only torch Adam and SGD optimizer states are supported")
        return {"class": name, "param_groups": groups, "state": state}

    def _optimizer_from_manifest(self, manifest: Mapping[str, Any]) -> torch.optim.Optimizer:
        named = dict(self.named_parameters())
        groups = []
        for saved_group in manifest["param_groups"]:
            params = []
            for name in saved_group["param_names"]:
                if name not in named:
                    raise ValueError(f"optimizer refers to unknown full-policy parameter {name!r}")
                params.append(named[name])
            groups.append({**copy.deepcopy(saved_group["options"]), "params": params})
        if not groups:
            raise ValueError("optimizer manifest has no parameter groups")
        if manifest.get("class") == "Adam":
            optimizer = torch.optim.Adam(groups)
        elif manifest.get("class") == "SGD":
            optimizer = torch.optim.SGD(groups)
        else:
            raise ValueError("only Adam and SGD optimizer manifests are supported")
        for saved_name, values in manifest.get("state", {}).items():
            parameter = named.get(saved_name)
            if parameter is None:
                raise ValueError(f"optimizer state refers to unknown full-policy parameter {saved_name!r}")
            optimizer.state[parameter] = {
                key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in values.items()
            }
        return optimizer

    def make_optimizer(
        self,
        *,
        learning_rate: float = 1e-2,
        optimizer: str = "adam",
        momentum: float = 0.0,
    ) -> torch.optim.Optimizer:
        """Build a full actor and critic optimizer; SGD is a lower-state option."""
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if optimizer == "adam":
            return torch.optim.Adam(trainable, lr=learning_rate, eps=1e-5)
        if optimizer == "sgd":
            momentum = float(momentum)
            if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
                raise ValueError("momentum must be finite and in [0, 1)")
            return torch.optim.SGD(trainable, lr=learning_rate, momentum=momentum)
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    def persistent_tensor_bytes(self) -> int:
        return _as_tensor_bytes(self.state_dict())

    def storage_report(self, optimizer: torch.optim.Optimizer | None = None) -> dict[str, int | str | bool]:
        actor_named = self.actor_named_parameters()
        actor_bytes = sum(int(parameter.numel() * parameter.element_size()) for parameter in actor_named.values())
        route_bytes = 0 if self.route_embedding is None else int(self.route_embedding.numel() * self.route_embedding.element_size())
        critic_bytes = sum(
            int(parameter.numel() * parameter.element_size())
            for name, parameter in self.named_parameters()
            if name.startswith("mlp_extractor.value_net.") or name.startswith("value_net.")
        )
        optimizer_total = _as_tensor_bytes(optimizer.state_dict()) if optimizer is not None else 0
        optimizer_actor = 0
        if optimizer is not None:
            actor_ids = {id(parameter) for parameter in actor_named.values()}
            optimizer_actor = sum(_as_tensor_bytes(values) for parameter, values in optimizer.state.items() if id(parameter) in actor_ids)
        policy_bytes = self.persistent_tensor_bytes()
        return {
            "kind": "full_finetune_route_conditioned" if self.route_conditioned else "full_finetune_plain",
            "persistent_policy_tensor_bytes": policy_bytes,
            "total_policy_bytes": policy_bytes,
            "trainable_actor_parameter_bytes": actor_bytes,
            "route_embedding_bytes": route_bytes,
            "band_embedding_bytes": 0 if self.band_embedding is None else self.band_embedding.numel() * self.band_embedding.element_size(),
            "critic_parameter_bytes": critic_bytes,
            "optimizer_tensor_bytes": optimizer_total,
            "actor_optimizer_tensor_bytes": optimizer_actor,
            "adaptive_actor_tensor_bytes": actor_bytes + optimizer_actor,
            "full_optimizer_state_included": optimizer is not None,
            "shared_actor_copies": 1,
        }

    def to_spec(self) -> dict[str, Any]:
        if not isinstance(self.observation_space, spaces.Box) or not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("only Box observations and Discrete actions are supported")
        return {
            "format_version": _FORMAT_VERSION,
            "architecture": _ARCHITECTURE,
            "route_conditioned": self.route_conditioned,
            "band_conditioned": self.band_conditioned,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "observation_space": {
                "kind": "Box",
                "dtype": np.dtype(self.observation_space.dtype).name,
                "low": np.asarray(self.observation_space.low).tolist(),
                "high": np.asarray(self.observation_space.high).tolist(),
            },
            "action_space": {"kind": "Discrete", "n": int(self.action_space.n)},
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "C1FullFineTunePolicy":
        if int(spec.get("format_version", -1)) != _FORMAT_VERSION or spec.get("architecture") != _ARCHITECTURE:
            raise ValueError("unsupported C1 full fine tuning policy spec")
        observation = spec["observation_space"]
        action = spec["action_space"]
        if observation.get("kind") != "Box" or action.get("kind") != "Discrete" or int(action.get("n", -1)) != 4:
            raise ValueError("C1 full fine tuning policy spec requires Box observations and four actions")
        low = np.asarray(observation["low"], dtype=np.dtype(observation["dtype"]))
        high = np.asarray(observation["high"], dtype=np.dtype(observation["dtype"]))
        observation_space = spaces.Box(low=low, high=high, dtype=np.dtype(observation["dtype"]))
        reference = TreePolicy(
            observation_space=observation_space,
            action_space=spaces.Discrete(4),
            lr_schedule=lambda _: 1e-2,
        )
        result = cls.from_reference(
            reference,
            route_conditioned=bool(spec["route_conditioned"]),
            band_conditioned=bool(spec.get("band_conditioned", False)),
            epsilon=float(spec["epsilon"]),
            temperature=float(spec["temperature"]),
        )
        del reference
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        if int(payload.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported C1 full fine tuning payload version")
        policy = cls.from_spec(payload["spec"])
        policy.load_state_dict(payload["policy"], strict=True)
        optimizer = policy._optimizer_from_manifest(payload["optimizer"])
        return {
            "policy": policy,
            "optimizer": optimizer,
            "router": copy.deepcopy(payload.get("router")),
            "metadata": copy.deepcopy(payload.get("metadata", {})),
            "rng": copy.deepcopy(payload.get("rng", {})),
            "replay": copy.deepcopy(payload.get("replay")),
            "ewc": copy.deepcopy(payload.get("ewc")),
        }


def export_payload(
    policy: C1FullFineTunePolicy,
    optimizer: torch.optim.Optimizer,
    router_snapshot: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    rng: Mapping[str, Any] | None = None,
    replay_state: Mapping[str, Any] | None = None,
    ewc_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a serializable full-policy, named-optimizer checkpoint."""
    return {
        "format_version": _FORMAT_VERSION,
        "spec": policy.to_spec(),
        "policy": {key: value.detach().clone() for key, value in policy.state_dict().items()},
        "optimizer": policy.optimizer_manifest(optimizer),
        "router": copy.deepcopy(router_snapshot),
        "metadata": copy.deepcopy(dict(metadata or {})),
        "rng": copy.deepcopy(dict(rng) if rng is not None else _capture_rng_state()),
        "replay": copy.deepcopy(replay_state),
        "ewc": copy.deepcopy(ewc_state),
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


class C1BoundedBehaviorReservoir:
    """Algorithm R reservoir of online observations and their behavior policy.

    Each item stores the 13-value policy input and four teacher probabilities
    in float32 arrays. The buffer never stores task labels, rewards, or a model
    snapshot. Call :meth:`observe` when the current policy generated an action,
    before the next PPO update changes that policy.
    """

    def __init__(self, capacity: int = 256, *, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or int(capacity) <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, _OBS_DIM), dtype=np.float32)
        self.teacher_probabilities = np.zeros((self.capacity, _ACTION_COUNT), dtype=np.float32)
        self.size = 0
        self.seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, observation: np.ndarray | torch.Tensor, teacher_probabilities: np.ndarray | torch.Tensor) -> bool:
        obs = np.asarray(observation.detach().cpu() if torch.is_tensor(observation) else observation, dtype=np.float32)
        probs = np.asarray(
            teacher_probabilities.detach().cpu() if torch.is_tensor(teacher_probabilities) else teacher_probabilities,
            dtype=np.float32,
        )
        if obs.shape != (_OBS_DIM,) or not np.all(np.isfinite(obs)):
            raise ValueError("reservoir observations must be finite 13 value arrays")
        if probs.shape != (_ACTION_COUNT,) or not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
            raise ValueError("teacher probabilities must be a finite nonnegative four value array")
        if not np.isclose(float(probs.sum()), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("teacher probabilities must sum to one")
        self.seen += 1
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            index = int(self._rng.integers(self.seen))
            if index >= self.capacity:
                return False
        self.observations[index] = obs
        self.teacher_probabilities[index] = probs
        return True

    @torch.no_grad()
    def observe(self, policy: Any, observation: np.ndarray | torch.Tensor) -> bool:
        """Snapshot the current policy distribution for one online input."""
        core = getattr(policy, "core", policy)
        obs = np.asarray(observation.detach().cpu() if torch.is_tensor(observation) else observation, dtype=np.float32)
        if obs.shape != (_OBS_DIM,):
            raise ValueError("reservoir observations must have shape (13,)")
        probs = core.get_distribution(torch.as_tensor(obs)).distribution.probs.squeeze(0)
        return self.add(obs, probs)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample an empty behavior reservoir")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        replace = int(batch_size) > self.size
        indices = self._rng.choice(self.size, size=int(batch_size), replace=replace)
        return self.observations[indices].copy(), self.teacher_probabilities[indices].copy()

    def storage_report(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "seen": self.seen,
            "observation_bytes": int(self.observations.nbytes),
            "teacher_probability_bytes": int(self.teacher_probabilities.nbytes),
            "array_storage_bytes": int(self.observations.nbytes + self.teacher_probabilities.nbytes),
            "bytes_per_item": int(_OBS_DIM * np.dtype(np.float32).itemsize + _ACTION_COUNT * np.dtype(np.float32).itemsize),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the bounded data and sampler RNG needed to resume replay."""
        return {
            "capacity": self.capacity,
            "observations": self.observations.copy(),
            "teacher_probabilities": self.teacher_probabilities.copy(),
            "size": self.size,
            "seen": self.seen,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("reservoir checkpoint capacity does not match")
        observations = np.asarray(state["observations"], dtype=np.float32)
        probabilities = np.asarray(state["teacher_probabilities"], dtype=np.float32)
        size, seen = int(state["size"]), int(state["seen"])
        if observations.shape != self.observations.shape or probabilities.shape != self.teacher_probabilities.shape:
            raise ValueError("reservoir checkpoint arrays have invalid shapes")
        if not 0 <= size <= self.capacity or seen < size:
            raise ValueError("reservoir checkpoint counters are invalid")
        if not np.all(np.isfinite(observations[:size])) or not np.all(np.isfinite(probabilities[:size])):
            raise ValueError("reservoir checkpoint contains non-finite values")
        if size and not np.allclose(probabilities[:size].sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("reservoir checkpoint teacher probabilities must sum to one")
        self.observations[:] = observations
        self.teacher_probabilities[:] = probabilities
        self.size = size
        self.seen = seen
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class C1OnlineEWC:
    """Online diagonal EWC over shared actor parameters only.

    The Fisher estimate is the mean of per-example score-gradient squares,
    ``mean_i((d log pi(a_i|s_i) / d theta) ** 2)``.  It deliberately does not
    square a minibatch-mean gradient.  The caller supplies observations and
    actions from an existing online rollout (or offline source observations
    for an initial source prior); this class never reads task IDs or steps an
    environment.
    """

    def __init__(self, policy: C1FullFineTunePolicy, *, decay: float = 1.0) -> None:
        value = float(decay)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("decay must be finite and in [0, 1]")
        self.decay = value
        self.reference = {
            name: parameter.detach().clone()
            for name, parameter in policy.actor_named_parameters().items()
        }
        self.fisher = {name: torch.zeros_like(value) for name, value in self.reference.items()}
        self.enabled = False
        self.consolidations = 0
        self.observations_used = 0
        self.source_prior_observations = 0

    @staticmethod
    def estimate_diagonal_fisher(
        policy: C1FullFineTunePolicy,
        observations: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor | None = None,
        *,
        sample_size: int | None = None,
        seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Estimate diagonal empirical Fisher from per-sample log scores."""
        device = next(policy.parameters()).device
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        if obs.ndim == 2 and obs.shape[1] == 4:
            # Source pretraining data stores only navigation coordinates;
            # source rollouts used zero history and root route 0.
            obs = torch.cat((obs, obs.new_zeros((obs.shape[0], _OBS_DIM - 4))), dim=1)
        if obs.ndim != 2 or tuple(obs.shape[1:]) != (_OBS_DIM,) or obs.shape[0] == 0:
            raise ValueError("Fisher observations must have shape (N, 4) or (N, 13) with N positive")
        if sample_size is not None:
            if isinstance(sample_size, bool) or int(sample_size) <= 0:
                raise ValueError("sample_size must be positive")
            rng = np.random.default_rng(seed)
            count = min(int(sample_size), int(obs.shape[0]))
            indices = rng.choice(obs.shape[0], size=count, replace=False)
            obs = obs[torch.as_tensor(indices, dtype=torch.long, device=device)]
            if actions is not None:
                actions = torch.as_tensor(actions, dtype=torch.long, device=device).reshape(-1)[
                    torch.as_tensor(indices, dtype=torch.long, device=device)
                ]
        parameters = policy.actor_named_parameters()
        names = list(parameters)
        values = list(parameters.values())
        estimate = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
        if actions is None:
            with torch.no_grad():
                probabilities = policy.get_distribution(obs).distribution.probs
                selected_actions = torch.multinomial(probabilities, num_samples=1).flatten()
        else:
            selected_actions = torch.as_tensor(actions, dtype=torch.long, device=device).reshape(-1)
            if selected_actions.numel() != obs.shape[0]:
                raise ValueError("Fisher actions must have one entry per observation")
            if torch.any((selected_actions < 0) | (selected_actions >= _ACTION_COUNT)):
                raise ValueError("Fisher actions must be in the four-action space")

        for index in range(obs.shape[0]):
            distribution = policy.get_distribution(obs[index])
            log_probability = distribution.log_prob(selected_actions[index].reshape(1)).sum()
            gradients = torch.autograd.grad(log_probability, values, allow_unused=True)
            for name, gradient in zip(names, gradients):
                if gradient is not None:
                    estimate[name].add_(gradient.detach().square())
        scale = 1.0 / float(obs.shape[0])
        for name in estimate:
            estimate[name].mul_(scale)
        return estimate

    def initialize_source_prior(
        self,
        policy: C1FullFineTunePolicy,
        source_observations: np.ndarray | torch.Tensor,
        source_actions: np.ndarray | torch.Tensor | None = None,
        *,
        sample_size: int = 64,
        seed: int = 0,
    ) -> None:
        """Estimate the initial W0 prior from existing source observations."""
        fisher = self.estimate_diagonal_fisher(
            policy, source_observations, source_actions, sample_size=sample_size, seed=seed
        )
        self.reference = {
            name: parameter.detach().clone()
            for name, parameter in policy.actor_named_parameters().items()
        }
        self.fisher = {name: value.detach().clone() for name, value in fisher.items()}
        self.enabled = True
        self.source_prior_observations = int(min(sample_size, len(source_observations)))
        self.observations_used += self.source_prior_observations

    def consolidate(
        self,
        policy: C1FullFineTunePolicy,
        fisher: Mapping[str, torch.Tensor],
        *,
        sample_count: int,
    ) -> None:
        current = policy.actor_named_parameters()
        if set(current) != set(self.reference) or set(fisher) != set(self.fisher):
            raise ValueError("EWC actor parameter names do not match the policy")
        for name, parameter in current.items():
            value = fisher[name].to(device=parameter.device, dtype=parameter.dtype)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"EWC Fisher shape does not match actor parameter {name!r}")
            self.fisher[name].mul_(self.decay).add_(value)
            self.reference[name] = parameter.detach().clone()
        self.enabled = True
        self.consolidations += 1
        self.observations_used += int(sample_count)

    def regularization_loss(self, policy: C1FullFineTunePolicy) -> torch.Tensor:
        """Return the unscaled half-Fisher-weighted quadratic penalty."""
        parameters = policy.actor_named_parameters()
        if set(parameters) != set(self.reference):
            raise ValueError("EWC actor parameter names do not match the policy")
        if not self.enabled:
            return next(policy.parameters()).new_zeros(())
        terms = []
        for name, parameter in parameters.items():
            fisher = self.fisher[name].to(device=parameter.device, dtype=parameter.dtype)
            reference = self.reference[name].to(device=parameter.device, dtype=parameter.dtype)
            terms.append((fisher * (parameter - reference).square()).sum())
        return 0.5 * torch.stack(terms).sum()

    def register_gradient_hooks(
        self,
        policy: C1FullFineTunePolicy,
        coefficient: float,
    ) -> list[Any]:
        """Add the EWC gradient to every actor gradient during a PPO update."""
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("coefficient must be finite and nonnegative")
        if not self.enabled or coefficient == 0.0:
            return []
        handles = []
        for name, parameter in policy.actor_named_parameters().items():
            fisher = self.fisher[name].to(device=parameter.device, dtype=parameter.dtype)
            reference = self.reference[name].to(device=parameter.device, dtype=parameter.dtype)

            def add_ewc_gradient(gradient, *, parameter=parameter, fisher=fisher, reference=reference):
                return gradient + coefficient * fisher * (parameter.detach() - reference)

            handles.append(parameter.register_hook(add_ewc_gradient))
        return handles

    def storage_report(self) -> dict[str, int | bool]:
        fisher_bytes = _as_tensor_bytes(self.fisher)
        reference_bytes = _as_tensor_bytes(self.reference)
        return {
            "enabled": self.enabled,
            "fisher_tensor_bytes": fisher_bytes,
            "reference_actor_bytes": reference_bytes,
            "ewc_state_tensor_bytes": fisher_bytes + reference_bytes,
            "consolidations": self.consolidations,
            "observations_used": self.observations_used,
            "source_prior_observations": self.source_prior_observations,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "enabled": self.enabled,
            "consolidations": self.consolidations,
            "observations_used": self.observations_used,
            "source_prior_observations": self.source_prior_observations,
            "reference": {name: value.detach().clone() for name, value in self.reference.items()},
            "fisher": {name: value.detach().clone() for name, value in self.fisher.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any], policy: C1FullFineTunePolicy) -> None:
        parameters = policy.actor_named_parameters()
        reference, fisher = state["reference"], state["fisher"]
        if set(reference) != set(parameters) or set(fisher) != set(parameters):
            raise ValueError("EWC checkpoint actor parameter names do not match the policy")
        for name, parameter in parameters.items():
            if tuple(reference[name].shape) != tuple(parameter.shape) or tuple(fisher[name].shape) != tuple(parameter.shape):
                raise ValueError(f"EWC checkpoint shape does not match actor parameter {name!r}")
        self.decay = float(state["decay"])
        if not math.isfinite(self.decay) or not 0.0 <= self.decay <= 1.0:
            raise ValueError("EWC checkpoint decay is invalid")
        self.reference = {name: value.detach().clone().to(device=parameters[name].device) for name, value in reference.items()}
        self.fisher = {name: value.detach().clone().to(device=parameters[name].device) for name, value in fisher.items()}
        self.enabled = bool(state["enabled"])
        self.consolidations = int(state["consolidations"])
        self.observations_used = int(state["observations_used"])
        self.source_prior_observations = int(state["source_prior_observations"])


def behavior_cloning_kl(
    policy: C1FullFineTunePolicy,
    observations: np.ndarray | torch.Tensor,
    teacher_probabilities: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Mean KL(teacher || current policy) over a replay batch."""
    device = next(policy.parameters()).device
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    teacher = torch.as_tensor(teacher_probabilities, dtype=torch.float32, device=device)
    student = policy.get_distribution(obs).distribution.probs
    teacher = teacher.clamp_min(1e-8)
    return F.kl_div(student.clamp_min(1e-8).log(), teacher, reduction="batchmean")


class C1ContinualPPO(PPO):
    """SB3 PPO with optional bounded behavior replay and full-policy EWC.

    With ``ewc_coefficient=0``, the adapter core may be a full policy or a
    frozen-base residual policy that implements ``get_distribution`` and
    ``evaluate_actions``. EWC is restricted to :class:`C1FullFineTunePolicy`.
    Its gradient is included in each actor update. Behavior cloning takes a
    separate optimizer step after the ordinary PPO epochs, using a uniform
    sample from the bounded online reservoir.
    """

    def __init__(
        self,
        *args,
        replay_capacity: int = 256,
        bc_coefficient: float = 0.05,
        bc_batch_size: int = 64,
        replay_seed: int = 0,
        ewc_coefficient: float = 1.0,
        ewc_interval: int = 2048,
        ewc_batch_size: int = 64,
        ewc_decay: float = 1.0,
        ewc_seed: int = 0,
        **kwargs,
    ) -> None:
        self.replay_capacity = int(replay_capacity)
        self.bc_coefficient = float(bc_coefficient)
        self.bc_batch_size = int(bc_batch_size)
        if self.replay_capacity <= 0 or self.bc_batch_size <= 0:
            raise ValueError("replay_capacity and bc_batch_size must be positive")
        if not math.isfinite(self.bc_coefficient) or self.bc_coefficient < 0.0:
            raise ValueError("bc_coefficient must be finite and nonnegative")
        self.ewc_coefficient = float(ewc_coefficient)
        self.ewc_interval = int(ewc_interval)
        self.ewc_batch_size = int(ewc_batch_size)
        self.ewc_decay = float(ewc_decay)
        if not math.isfinite(self.ewc_coefficient) or self.ewc_coefficient < 0.0:
            raise ValueError("ewc_coefficient must be finite and nonnegative")
        if self.ewc_interval <= 0 or self.ewc_batch_size <= 0:
            raise ValueError("ewc_interval and ewc_batch_size must be positive")
        if not math.isfinite(self.ewc_decay) or not 0.0 <= self.ewc_decay <= 1.0:
            raise ValueError("ewc_decay must be finite and in [0, 1]")
        self.behavior_reservoir = C1BoundedBehaviorReservoir(self.replay_capacity, seed=replay_seed)
        self.continual_bc_updates = 0
        self.online_ewc: C1OnlineEWC | None = None
        self.ewc_consolidation_count = 0
        self._next_ewc_step = self.ewc_interval
        self._ewc_rng = np.random.default_rng(ewc_seed)
        super().__init__(*args, **kwargs)

    def observe_online(self, observation: np.ndarray | torch.Tensor) -> bool:
        """Record the current behavior probabilities for one online state."""
        return self.behavior_reservoir.observe(self.policy, observation)

    def _continual_core(self) -> C1FullFineTunePolicy:
        core = self._policy_core()
        if not isinstance(core, C1FullFineTunePolicy):
            raise TypeError("C1ContinualPPO requires a policy adapter with a C1FullFineTunePolicy core")
        return core

    def _policy_core(self) -> Any:
        core = getattr(self.policy, "core", self.policy)
        if not callable(getattr(core, "get_distribution", None)) or not callable(getattr(core, "evaluate_actions", None)):
            raise TypeError("C1ContinualPPO requires a policy core with get_distribution and evaluate_actions")
        return core

    def initialize_source_ewc(
        self,
        source_observations: np.ndarray | torch.Tensor,
        source_actions: np.ndarray | torch.Tensor | None = None,
        *,
        sample_size: int = 64,
        seed: int = 0,
    ) -> None:
        """Set a W0 anchor and source Fisher from previously collected data."""
        core = self._continual_core()
        self.online_ewc = C1OnlineEWC(core, decay=self.ewc_decay)
        self.online_ewc.initialize_source_prior(
            core, source_observations, source_actions, sample_size=sample_size, seed=seed
        )

    def regularization_loss(self) -> torch.Tensor:
        """Return the unweighted online-EWC penalty for the current actor."""
        core = self._continual_core()
        if self.online_ewc is None:
            self.online_ewc = C1OnlineEWC(core, decay=self.ewc_decay)
        return self.online_ewc.regularization_loss(core)

    def continual_state_dict(self) -> dict[str, Any]:
        return {
            "behavior_reservoir": self.behavior_reservoir.state_dict(),
            "online_ewc": None if self.online_ewc is None else self.online_ewc.state_dict(),
            "next_ewc_step": self._next_ewc_step,
            "ewc_consolidation_count": self.ewc_consolidation_count,
            "continual_bc_updates": self.continual_bc_updates,
            "ewc_rng_state": copy.deepcopy(self._ewc_rng.bit_generator.state),
        }

    def load_continual_state_dict(self, state: Mapping[str, Any]) -> None:
        self.behavior_reservoir.load_state_dict(state["behavior_reservoir"])
        ewc_state = state.get("online_ewc")
        if ewc_state is None:
            self.online_ewc = None
        else:
            core = self._continual_core()
            self.online_ewc = C1OnlineEWC(core, decay=float(ewc_state["decay"]))
            self.online_ewc.load_state_dict(ewc_state, core)
        self._next_ewc_step = int(state["next_ewc_step"])
        self.ewc_consolidation_count = int(state["ewc_consolidation_count"])
        self.continual_bc_updates = int(state["continual_bc_updates"])
        self._ewc_rng.bit_generator.state = copy.deepcopy(state["ewc_rng_state"])

    def train(self) -> None:
        core = self._policy_core()
        full_core = isinstance(core, C1FullFineTunePolicy)
        if self.ewc_coefficient > 0.0 and not full_core:
            raise TypeError("online EWC is supported only with C1FullFineTunePolicy; set ewc_coefficient=0 for replay-only use")
        if self.ewc_coefficient > 0.0 and self.online_ewc is None:
            self.online_ewc = C1OnlineEWC(core, decay=self.ewc_decay)

        pending_fisher = None
        pending_fisher_samples = 0
        if self.ewc_coefficient > 0.0 and self.num_timesteps >= self._next_ewc_step:
            observations = np.asarray(self.rollout_buffer.observations, dtype=np.float32).reshape(-1, _OBS_DIM)
            actions = np.asarray(self.rollout_buffer.actions, dtype=np.int64).reshape(-1)
            count = min(self.ewc_batch_size, len(observations))
            pending_fisher_samples = count
            indices = self._ewc_rng.choice(len(observations), size=count, replace=False)
            pending_fisher = (observations[indices].copy(), actions[indices].copy())
            while self._next_ewc_step <= self.num_timesteps:
                self._next_ewc_step += self.ewc_interval

        ewc_loss = self.regularization_loss() if self.ewc_coefficient > 0.0 else None
        handles = (
            self.online_ewc.register_gradient_hooks(core, self.ewc_coefficient)
            if self.online_ewc is not None and full_core else []
        )
        try:
            super().train()
            if self.bc_coefficient > 0.0 and self.behavior_reservoir.size:
                obs, teacher = self.behavior_reservoir.sample(self.bc_batch_size)
                replay_loss = behavior_cloning_kl(core, obs, teacher)
                optimizer = self.policy.optimizer
                optimizer.zero_grad()
                (self.bc_coefficient * replay_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                optimizer.step()
                self.continual_bc_updates += 1
                self.logger.record("train/continual_behavior_kl", float(replay_loss.detach().cpu()))
                self.logger.record("train/continual_bc_updates", self.continual_bc_updates, exclude="tensorboard")
        finally:
            for handle in handles:
                handle.remove()
        if self.online_ewc is not None and self.online_ewc.enabled and ewc_loss is not None:
            self.logger.record("train/ewc_penalty", float(ewc_loss.detach().cpu()))

        if pending_fisher is not None:
            assert self.online_ewc is not None
            # Use the same collected samples, with Fisher and reference both
            # evaluated at the policy being consolidated after the update.
            fisher = C1OnlineEWC.estimate_diagonal_fisher(core, *pending_fisher)
            self.online_ewc.consolidate(core, fisher, sample_count=pending_fisher_samples)
            self.ewc_consolidation_count += 1
            self.logger.record("train/ewc_consolidations", self.ewc_consolidation_count, exclude="tensorboard")
            self.logger.record("train/ewc_fisher_samples", pending_fisher_samples, exclude="tensorboard")


__all__ = [
    "C1BoundedBehaviorReservoir",
    "C1ContinualPPO",
    "C1FullFineTunePolicy",
    "C1OnlineEWC",
    "behavior_cloning_kl",
    "export_payload",
]
