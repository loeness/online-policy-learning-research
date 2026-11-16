"""Factorized residual policies for the six-context C1 experiment.

The pretrained ``TreePolicy`` supplies a frozen actor and a trainable critic.
Residuals are expressed in a fixed three dimensional action contrast space
and use the observed x coordinate to select one of three local bands.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import make_proba_distribution
from torch import nn
from torch.nn import functional as F

from .compression import _FORMAT_VERSION, _SUPPORTED_ARCHITECTURE, PackedResidualPolicy
from .policy import LeafAdam, TreePolicy


_ENDPOINTS = tuple(range(1, 7))
_PARENTS = {1: 7, 2: 7, 3: 7, 4: 8, 5: 8, 6: 8}
_LAYOUTS = {"flat", "tree", "dictionary", "dense"}
_BASE_FEATURES = 64
_BASE_PHI = _BASE_FEATURES + 1
_LOCAL_PHI = 3 * _BASE_PHI


def _contrast_basis(*, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """Return the fixed orthonormal four action contrast basis."""
    basis = torch.zeros((4, 3), dtype=dtype, device=device)
    for column in range(3):
        scale = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


def _mark_residual(parameter: nn.Parameter) -> nn.Parameter:
    parameter.tree_residual = True
    return parameter


class _LowRankResidual(nn.Module):
    """Three contrast rows represented as U @ V."""

    def __init__(self, rank: int, width: int, *, like: torch.Tensor) -> None:
        super().__init__()
        self.U = _mark_residual(nn.Parameter(torch.empty((3, rank), device=like.device, dtype=like.dtype)))
        self.V = _mark_residual(nn.Parameter(torch.zeros((rank, width), device=like.device, dtype=like.dtype)))
        nn.init.normal_(self.U, mean=0.0, std=0.02)


class _DenseResidual(nn.Module):
    """Three contrast rows stored directly."""

    def __init__(self, width: int, *, like: torch.Tensor) -> None:
        super().__init__()
        self.values = _mark_residual(nn.Parameter(torch.zeros((3, width), device=like.device, dtype=like.dtype)))


class _DictionaryResidual(nn.Module):
    """Endpoint coefficients over a shared learned dictionary."""

    def __init__(self, size: int, *, like: torch.Tensor) -> None:
        super().__init__()
        self.A = _mark_residual(nn.Parameter(torch.zeros((3, size), device=like.device, dtype=like.dtype)))


class C1ResidualPolicy(PackedResidualPolicy):
    """Frozen-base actor critic with C1 local residual layouts.

    Use :meth:`from_dense` to copy a pretrained ``TreePolicy``. ``method`` is
    one of ``flat``, ``tree``, ``dictionary`` or ``dense``. Flat and tree use
    ranks one or two; dictionary uses six or twelve shared atoms.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        # Algorithmic constant. It is recreated from code and adds no checkpoint bytes.
        self.register_buffer("_contrast_basis", _contrast_basis(), persistent=False)
        self._storage = "flat"
        self.rank = 1
        self.dictionary_size = 6
        self.epsilon = 0.05
        self.temperature = self._validate_temperature(temperature)
        self._parents = nn.ModuleDict()
        self._dictionary_V: nn.Parameter | None = None

    @classmethod
    def from_dense(
        cls,
        policy: TreePolicy,
        *,
        method: str = "flat",
        rank: int = 1,
        dictionary_size: int = 6,
        epsilon: float = 0.05,
        temperature: float = 1.0,
        train_critic: bool = True,
    ) -> "C1ResidualPolicy":
        """Copy a pretrained tree actor critic and create zero-output residuals."""
        if not isinstance(policy, TreePolicy):
            raise TypeError("from_dense currently supports online_rl.policy.TreePolicy")
        if tuple(policy.observation_space.shape) != (13,):
            raise ValueError("C1ResidualPolicy requires a 13 value observation")
        if not isinstance(policy.action_space, spaces.Discrete) or int(policy.action_space.n) != 4:
            raise ValueError("C1ResidualPolicy requires four discrete actions")
        if int(policy.mlp_extractor.latent_dim_pi) != _BASE_FEATURES:
            raise ValueError("C1ResidualPolicy requires 64 frozen actor features")
        cls._validate_options(method, rank, dictionary_size, epsilon)
        temperature = cls._validate_temperature(temperature)

        for name, parameter in policy.named_parameters():
            parameter._packed_source_name = name
        paths = cls._paths_for(method)
        result = cls._copy_base(policy, paths, storage=method, train_critic=train_critic)
        result.max_nodes = 9
        result._paths = paths
        result._storage = method
        result.rank = int(rank)
        result.dictionary_size = int(dictionary_size)
        result.epsilon = float(epsilon)
        result.temperature = temperature
        result._make_residual_layout(like=next(policy.mlp_extractor.policy_net.parameters()))
        result._set_trainable_flags(train_critic=train_critic)
        return result

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        method: str = "flat",
        leaf_rank: int = 1,
        dictionary_rank: int = 6,
        exploration_epsilon: float = 0.05,
        temperature: float = 1.0,
        train_critic: bool = True,
    ) -> "C1ResidualPolicy":
        """Runner-facing alias with names matching the C1 experiment config."""
        return cls.from_dense(
            reference,
            method=method,
            rank=leaf_rank,
            dictionary_size=dictionary_rank,
            epsilon=exploration_epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )

    @staticmethod
    def _validate_options(method: str, rank: int, dictionary_size: int, epsilon: float) -> None:
        if method not in _LAYOUTS:
            raise ValueError(f"method must be one of {sorted(_LAYOUTS)}")
        if isinstance(rank, bool) or int(rank) not in (1, 2):
            raise ValueError("rank must be one or two")
        if isinstance(dictionary_size, bool) or int(dictionary_size) not in (6, 12):
            raise ValueError("dictionary_size must be six or twelve")
        if not math.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) < 1.0:
            raise ValueError("epsilon must be finite and in [0, 1)")

    @staticmethod
    def _validate_temperature(temperature: float) -> float:
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        return temperature

    @staticmethod
    def _paths_for(method: str) -> dict[int, tuple[int, ...]]:
        if method == "tree":
            return {endpoint: (_PARENTS[endpoint], endpoint) for endpoint in _ENDPOINTS}
        return {endpoint: (endpoint,) for endpoint in _ENDPOINTS}

    def _make_residual_layout(self, *, like: torch.Tensor) -> None:
        self._nodes = nn.ModuleDict()
        self._parents = nn.ModuleDict()
        self._node_shapes = {}
        self._dictionary_V = None
        if self._storage == "dictionary":
            shared = nn.Parameter(torch.empty((self.dictionary_size, _LOCAL_PHI), device=like.device, dtype=like.dtype))
            self._dictionary_V = _mark_residual(shared)
            nn.init.normal_(self._dictionary_V, mean=0.0, std=0.02)
            for endpoint in _ENDPOINTS:
                self._nodes[str(endpoint)] = _DictionaryResidual(self.dictionary_size, like=like)
                self._node_shapes[endpoint] = (3, _LOCAL_PHI)
        elif self._storage == "dense":
            for endpoint in _ENDPOINTS:
                self._nodes[str(endpoint)] = _DenseResidual(_LOCAL_PHI, like=like)
                self._node_shapes[endpoint] = (3, _LOCAL_PHI)
        elif self._storage == "flat":
            for endpoint in _ENDPOINTS:
                self._nodes[str(endpoint)] = _LowRankResidual(self.rank, _LOCAL_PHI, like=like)
                self._node_shapes[endpoint] = (3, _LOCAL_PHI)
        else:  # tree
            for endpoint in _ENDPOINTS:
                self._nodes[str(endpoint)] = _LowRankResidual(self.rank, _LOCAL_PHI, like=like)
                self._node_shapes[endpoint] = (3, _LOCAL_PHI)
            for parent in (7, 8):
                self._parents[str(parent)] = _LowRankResidual(1, _BASE_PHI, like=like)
                self._node_shapes[parent] = (3, _BASE_PHI)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "C1ResidualPolicy":
        """Recreate the declared module layout; payload tensors fill its values."""
        if int(spec.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported C1 policy spec version")
        if spec.get("architecture") != _SUPPORTED_ARCHITECTURE:
            raise ValueError("unsupported C1 policy architecture")
        config = spec.get("c1_residual")
        if not isinstance(config, Mapping):
            raise ValueError("C1 policy spec has no residual configuration")
        method = str(config.get("method", ""))
        rank = int(config.get("rank", 1))
        dictionary_size = int(config.get("dictionary_size", 6))
        epsilon = float(config.get("epsilon", 0.05))
        temperature = cls._validate_temperature(config.get("temperature", 1.0))
        cls._validate_options(method, rank, dictionary_size, epsilon)

        obs = spec["observation_space"]
        if obs.get("kind") != "Box":
            raise ValueError("only Box observation spaces are supported")
        observation_space = spaces.Box(
            low=np.asarray(obs["low"], dtype=np.dtype(obs["dtype"])),
            high=np.asarray(obs["high"], dtype=np.dtype(obs["dtype"])),
            dtype=np.dtype(obs["dtype"]),
        )
        action = spec["action_space"]
        if action.get("kind") != "Discrete" or int(action["n"]) != 4:
            raise ValueError("C1 policy spec requires four discrete actions")
        paths = {int(endpoint): tuple(int(node) for node in path) for endpoint, path in spec["paths"].items()}
        base = TreePolicy(
            observation_space=observation_space,
            action_space=spaces.Discrete(4),
            lr_schedule=lambda _: 1e-2,
            max_nodes=9,
        )
        result = cls._copy_base(
            base, paths, storage=method, train_critic=bool(spec.get("train_critic", True))
        )
        result.max_nodes = 9
        result._paths = paths
        result._storage = method
        result.rank = rank
        result.dictionary_size = dictionary_size
        result.epsilon = epsilon
        result.temperature = temperature
        result._make_residual_layout(like=next(base.mlp_extractor.policy_net.parameters()))
        result._set_trainable_flags(train_critic=result.train_critic)
        expected = spec.get("residual_parameter_shapes")
        actual = result._residual_parameter_shapes()
        if expected is not None and expected != actual:
            raise ValueError("C1 residual parameter shapes do not match the declared layout")
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Restore policy, named LeafAdam state and companion payload fields."""
        if int(payload.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported C1 policy payload version")
        policy = cls.from_spec(payload["spec"])
        policy.load_state_dict(payload["policy"], strict=True)
        optimizer = policy._optimizer_from_manifest(payload["optimizer"])
        return {
            "policy": policy,
            "optimizer": optimizer,
            "router": copy.deepcopy(payload.get("router")),
            "metadata": copy.deepcopy(payload.get("metadata", {})),
            "rng": copy.deepcopy(payload.get("rng")),
        }

    def _residual_parameters(self) -> list[nn.Parameter]:
        parameters = [parameter for parameter in self.parameters() if getattr(parameter, "tree_residual", False)]
        return parameters

    def _set_trainable_flags(self, *, train_critic: bool) -> None:
        self.train_critic = bool(train_critic)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if train_critic:
            for parameter in self.mlp_extractor.value_net.parameters():
                parameter.requires_grad_(True)
            for parameter in self.value_net.parameters():
                parameter.requires_grad_(True)
        for parameter in self._residual_parameters():
            parameter.requires_grad_(True)
            parameter.tree_residual = True

    def _residual_parameter_shapes(self) -> dict[str, list[int]]:
        return {
            name: list(parameter.shape)
            for name, parameter in self.named_parameters()
            if getattr(parameter, "tree_residual", False)
        }

    def to_spec(self) -> dict[str, Any]:
        if not isinstance(self.observation_space, spaces.Box):
            raise ValueError("only Box observation spaces can be described")
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("only Discrete action spaces can be described")
        return {
            "format_version": _FORMAT_VERSION,
            "architecture": _SUPPORTED_ARCHITECTURE,
            "layout": "c1_residual_v1",
            "storage": self._storage,
            "max_nodes": 9,
            "train_critic": self.train_critic,
            "observation_space": {
                "kind": "Box",
                "dtype": np.dtype(self.observation_space.dtype).name,
                "low": np.asarray(self.observation_space.low).tolist(),
                "high": np.asarray(self.observation_space.high).tolist(),
            },
            "action_space": {"kind": "Discrete", "n": int(self.action_space.n)},
            "feature_dim": self.features_dim,
            "paths": {str(endpoint): list(path) for endpoint, path in sorted(self._paths.items())},
            "c1_residual": {
                "method": self._storage,
                "rank": self.rank,
                "dictionary_size": self.dictionary_size,
                "epsilon": self.epsilon,
                "temperature": self.temperature,
                "leaf_rank": self.rank,
                "dictionary_rank": self.dictionary_size,
                "exploration_epsilon": self.epsilon,
                "max_nodes": 9,
                "endpoint_ids": list(_ENDPOINTS),
                "parent_ids": {str(endpoint): parent for endpoint, parent in _PARENTS.items()},
                "base_phi_dim": _BASE_PHI,
                "local_phi_dim": _LOCAL_PHI,
                "gate": "floor(clamp(observation[0],0,1)*3), clamped to 0..2",
            },
            "residual_parameter_shapes": self._residual_parameter_shapes(),
        }

    def _feature_banks(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_phi = torch.cat([latent_pi, torch.ones_like(latent_pi[:, :1])], dim=1)
        x_band = torch.floor(obs[:, 0].clamp(0.0, 1.0) * 3.0).long().clamp(0, 2)
        gates = F.one_hot(x_band, num_classes=3).to(dtype=latent_pi.dtype)
        local_phi = (gates.unsqueeze(-1) * base_phi.unsqueeze(1)).reshape(obs.shape[0], _LOCAL_PHI)
        return base_phi, local_phi

    def _effective_matrix(self, endpoint: int) -> torch.Tensor:
        node = self._nodes[str(endpoint)]
        if self._storage == "flat" or self._storage == "tree":
            reduced = node.U @ node.V
        elif self._storage == "dictionary":
            reduced = node.A @ self._dictionary_V
        else:
            reduced = node.values
        return self._contrast_basis.to(device=reduced.device, dtype=reduced.dtype) @ reduced

    def _parent_matrix(self, parent: int) -> torch.Tensor:
        node = self._parents[str(parent)]
        reduced = node.U @ node.V
        return self._contrast_basis.to(device=reduced.device, dtype=reduced.dtype) @ reduced

    def _single_correction(
        self,
        endpoint: int,
        base_phi: torch.Tensor,
        local_phi: torch.Tensor,
    ) -> torch.Tensor:
        if self._storage == "tree":
            parent = _PARENTS[endpoint]
            parent_out = F.linear(base_phi, self._parent_matrix(parent))
            leaf_out = F.linear(local_phi, self._effective_matrix(endpoint))
            return parent_out + leaf_out
        return F.linear(local_phi, self._effective_matrix(endpoint))

    def _fallback_correction(self, family_parent: int, local_phi: torch.Tensor) -> torch.Tensor:
        # These routes are evaluation-only summaries of the first two seen leaves.
        siblings = (1, 2) if family_parent == 7 else (4, 5)
        matrix = (self._effective_matrix(siblings[0]).detach() + self._effective_matrix(siblings[1]).detach()) * 0.5
        return F.linear(local_phi, matrix)

    def _correction(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        obs = self._as_batch(obs)
        ids = obs[:, -1].round().long().clamp(0, 8)
        base_phi, local_phi = self._feature_banks(obs, latent_pi)
        result = latent_pi.new_zeros((latent_pi.shape[0], 4))
        for endpoint in _ENDPOINTS:
            batch_indices = torch.nonzero(ids == endpoint, as_tuple=False).reshape(-1)
            if batch_indices.numel() == 0:
                continue
            local_result = self._single_correction(
                endpoint,
                base_phi.index_select(0, batch_indices),
                local_phi.index_select(0, batch_indices),
            )
            result = result.index_add(0, batch_indices, local_result)
        for parent in (7, 8):
            batch_indices = torch.nonzero(ids == parent, as_tuple=False).reshape(-1)
            if batch_indices.numel() == 0:
                continue
            local_base = base_phi.index_select(0, batch_indices)
            local = local_phi.index_select(0, batch_indices)
            if self._storage == "tree":
                summary = F.linear(local_base, self._parent_matrix(parent))
            else:
                summary = self._fallback_correction(parent, local)
            result = result.index_add(0, batch_indices, summary)
        return result

    def distribution(self, obs: torch.Tensor | np.ndarray, latent_pi: torch.Tensor):
        obs = self._as_batch(obs)
        logits = self.action_net(latent_pi) + self._correction(obs, latent_pi)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        if self.epsilon:
            probabilities = (1.0 - self.epsilon) * probabilities + self.epsilon / 4.0
        # SB3's CategoricalDistribution wrapper retains get_actions/log_prob/entropy.
        self.action_dist.distribution = torch.distributions.Categorical(probs=probabilities)
        return self.action_dist

    def make_optimizer(
        self,
        dense_optimizer: torch.optim.Optimizer | None = None,
        *,
        learning_rate: float = 1e-2,
        transfer: bool = True,
        train_critic: bool | None = None,
    ) -> LeafAdam:
        """Build LeafAdam, optionally copying compatible critic moments by name."""
        if train_critic is not None:
            self.train_critic = bool(train_critic)
        self._set_trainable_flags(train_critic=self.train_critic)
        trainable = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if dense_optimizer is None or not transfer:
            optimizer = LeafAdam(list(trainable.values()), lr=float(learning_rate), eps=1e-5)
            optimizer.step_count = 0
            return optimizer

        # Parameters retain their TreePolicy names from _copy_base's source.
        groups: list[dict[str, Any]] = []
        pending_states: dict[nn.Parameter, dict[str, Any]] = {}
        known: set[int] = set()
        all_source_groups = dense_optimizer.param_groups
        for source_group in all_source_groups:
            targets = []
            for source_parameter in source_group["params"]:
                source_name = getattr(source_parameter, "_packed_source_name", None)
                if source_name is None:
                    # Normal TreePolicy optimizer parameters do not carry tags; infer by identity name map.
                    source_name = getattr(source_parameter, "_c1_source_name", None)
                if source_name is None:
                    continue
                target = trainable.get(source_name)
                if target is None or id(target) in known:
                    continue
                known.add(id(target))
                targets.append(target)
                source_state = dense_optimizer.state.get(source_parameter)
                if source_state:
                    pending_states[target] = {
                        key: value.detach().clone().to(device=target.device) if torch.is_tensor(value)
                        else copy.deepcopy(value)
                        for key, value in source_state.items()
                    }
            if targets:
                options = {key: copy.deepcopy(value) for key, value in source_group.items() if key != "params"}
                groups.append({**options, "params": targets})
        residuals = [parameter for parameter in trainable.values() if getattr(parameter, "tree_residual", False) and id(parameter) not in known]
        if residuals:
            if groups:
                groups[0]["params"].extend(residuals)
            else:
                groups.append({"params": residuals, "lr": float(learning_rate), "eps": 1e-5})
        residual_ids = {id(parameter) for parameter in residuals}
        uncovered = [
            parameter
            for parameter in trainable.values()
            if id(parameter) not in known and id(parameter) not in residual_ids
        ]
        if uncovered:
            if groups:
                groups[0]["params"].extend(uncovered)
            else:
                groups.append({"params": uncovered, "lr": float(learning_rate), "eps": 1e-5})
        if not groups:
            groups = [{"params": list(trainable.values()), "lr": float(learning_rate), "eps": 1e-5}]
        optimizer = LeafAdam(groups)
        optimizer.state.update(pending_states)
        optimizer.step_count = int(getattr(dense_optimizer, "step_count", 0))
        return optimizer

    def initialize_heldout(self, endpoint: int) -> None:
        """Initialize an unseen endpoint from the mean of its two siblings."""
        endpoint = int(endpoint)
        if endpoint not in _ENDPOINTS:
            raise ValueError("endpoint must be one of IDs one through six")
        if endpoint <= 3:
            siblings = tuple(node for node in (1, 2, 3) if node != endpoint)[:2]
        else:
            siblings = tuple(node for node in (4, 5, 6) if node != endpoint)[:2]
        node = self._nodes[str(endpoint)]
        with torch.no_grad():
            if self._storage == "tree":
                # Keep a nonzero left factor so the zero-start leaf can learn V.
                # A deterministic fallback handles a caller that cleared U first.
                if not torch.count_nonzero(node.U):
                    node.U.zero_()
                    for column in range(min(node.U.shape[1], node.U.shape[0])):
                        node.U[column, column] = 0.02
                node.V.zero_()
            elif self._storage == "dense":
                node.values.copy_(torch.stack([self._nodes[str(sibling)].values for sibling in siblings]).mean(0))
            elif self._storage == "dictionary":
                node.A.copy_(torch.stack([self._nodes[str(sibling)].A for sibling in siblings]).mean(0))
            else:
                matrices = torch.stack([
                    self._nodes[str(sibling)].U @ self._nodes[str(sibling)].V
                    for sibling in siblings
                ]).mean(0)
                left, singular, right = torch.linalg.svd(matrices, full_matrices=False)
                node.U.zero_()
                node.V.zero_()
                rank = min(self.rank, singular.numel())
                root_singular = singular[:rank].clamp_min(0).sqrt()
                node.U[:, :rank].copy_(left[:, :rank] * root_singular.unsqueeze(0))
                node.V[:rank, :].copy_(root_singular.unsqueeze(1) * right[:rank, :])

    def storage_report(self) -> dict[str, int | str]:
        residual_parameters = self._residual_parameters()
        residual_bytes = sum(parameter.numel() * parameter.element_size() for parameter in residual_parameters)
        total = self.persistent_tensor_bytes()
        return {
            "storage": self._storage,
            "persistent_policy_tensor_bytes": total,
            "total_policy_bytes": total,
            "base_and_head_tensor_bytes": total - residual_bytes,
            "residual_value_bytes": residual_bytes,
            "residual_weight_factor_bytes": residual_bytes,
            "weight_factor_bytes": residual_bytes,
            "coordinate_index_bytes": 0,
            "index_bytes": 0,
            "retained_coordinates": sum(parameter.numel() for parameter in residual_parameters),
            "learned_coordinate_count": sum(parameter.numel() for parameter in residual_parameters),
        }


__all__ = ["C1ResidualPolicy"]
