"""Alternative active-leaf residual layouts for storage comparisons.

These policies keep the same frozen SB3 actor and value network as
``PackedResidualPolicy`` and replace only its active residual matrices.  The
layouts are fixed at construction time; training updates their stored values
or factors directly and never keeps a dense residual copy as a backup.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from .compression import (
    PackedResidualPolicy,
    _FORMAT_VERSION,
    _SUPPORTED_ARCHITECTURE,
)
from .policy import LeafAdam, TreePolicy


_METHODS = {
    "dense",
    "topk50",
    "contrast",
    "svd2",
    "columns32",
    "bf16",
    "center_topk50",
    "activation_topk50",
}
_ACTIVE_SHAPE = (4, 65)
_TOPK_COUNT = math.ceil(0.5 * math.prod(_ACTIVE_SHAPE))
_COLUMN_COUNT = 32


def _helmert_basis(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a fixed orthonormal basis for the four-action contrast space."""
    basis = torch.zeros((4, 3), dtype=dtype)
    for column in range(3):
        scale = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


class _AlternativeResidualStorage(nn.Module):
    """One active residual represented in a selected fixed layout."""

    def __init__(
        self,
        shape: tuple[int, int],
        method: str,
        *,
        values: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        left: torch.Tensor | None = None,
        right: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.method = str(method)
        self.shape = tuple(int(part) for part in shape)
        rows, columns = self.shape

        if method in {"dense", "bf16"}:
            dtype = torch.bfloat16 if method == "bf16" else torch.float32
            if values is None:
                values = torch.zeros(self.shape, dtype=dtype)
            self.values = nn.Parameter(values.detach().to(dtype=dtype).reshape(self.shape).clone())
        elif method in {"topk50", "center_topk50", "activation_topk50"}:
            count = int(indices.numel()) if indices is not None else _TOPK_COUNT
            if indices is None:
                indices = torch.zeros(count, dtype=torch.uint16)
            if values is None:
                values = torch.zeros(count, dtype=torch.float32)
            self.register_buffer("indices", indices.detach().to(dtype=torch.uint16).reshape(-1).clone())
            self.values = nn.Parameter(values.detach().to(dtype=torch.float32).reshape(-1).clone())
        elif method == "contrast":
            if values is None:
                values = torch.zeros((3, columns), dtype=torch.float32)
            self.values = nn.Parameter(values.detach().to(dtype=torch.float32).reshape(3, columns).clone())
        elif method == "svd2":
            if left is None:
                left = torch.zeros((3, 2), dtype=torch.float32)
            if right is None:
                right = torch.zeros((2, columns), dtype=torch.float32)
            self.left = nn.Parameter(left.detach().to(dtype=torch.float32).reshape(3, 2).clone())
            self.right = nn.Parameter(right.detach().to(dtype=torch.float32).reshape(2, columns).clone())
        elif method == "columns32":
            if indices is None:
                indices = torch.zeros(_COLUMN_COUNT, dtype=torch.uint16)
            if values is None:
                values = torch.zeros((rows, _COLUMN_COUNT), dtype=torch.float32)
            self.register_buffer("indices", indices.detach().to(dtype=torch.uint16).reshape(-1).clone())
            self.values = nn.Parameter(
                values.detach().to(dtype=torch.float32).reshape(rows, _COLUMN_COUNT).clone()
            )
        else:
            raise ValueError(f"unsupported alternative residual method {method!r}")

        for parameter in self.parameters(recurse=False):
            parameter.tree_residual = True

    def residual_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.parameters(recurse=False))


class AlternativeResidualPolicy(PackedResidualPolicy):
    """Frozen-base actor critic with one of the alternative residual layouts."""

    def __init__(self) -> None:
        super().__init__()
        # This fixed algorithm basis is reconstructed on load and excluded from
        # state_dict byte accounting.
        self.register_buffer("_contrast_basis", _helmert_basis(), persistent=False)
        self._storage = "dense"

    @classmethod
    def from_reference(
        cls,
        reference: PackedResidualPolicy,
        method: str,
        *,
        calibration_obs: torch.Tensor | np.ndarray | Mapping[int | str, Any] | None = None,
    ) -> tuple["AlternativeResidualPolicy", dict[str, Any]]:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        if not isinstance(reference, PackedResidualPolicy):
            raise TypeError("reference must be a PackedResidualPolicy")
        if reference._storage != "dense":
            raise ValueError("reference must use storage='dense'")

        route_phi = cls._calibration_features(reference, calibration_obs)
        result = cls._copy_base(
            reference,
            reference._paths,
            storage=method,
            train_critic=reference.train_critic,
        )
        result._storage = method
        result._node_shapes = {}
        result._nodes = nn.ModuleDict()

        node_ids = sorted({node for path in reference._paths.values() for node in path if node != 0})
        metadata: dict[str, Any] = {
            "method": method,
            "active_nodes": node_ids,
            "dtype": "bfloat16" if method == "bf16" else "float32",
            "nodes": {},
        }

        for node_id in node_ids:
            source = reference._nodes[str(node_id)].values.detach().to(dtype=torch.float32)
            shape = tuple(int(part) for part in source.shape)
            if shape != _ACTIVE_SHAPE:
                raise ValueError(f"residual {node_id} has shape {shape}; expected {_ACTIVE_SHAPE}")
            result._node_shapes[node_id] = shape
            centered = source - source.mean(dim=0, keepdim=True)
            node_phi = cls._node_calibration_phi(reference, node_id, route_phi)

            if method == "dense":
                node = _AlternativeResidualStorage(shape, method, values=source)
            elif method == "topk50":
                selected = cls._top_coordinates(source.abs(), _TOPK_COUNT)
                node = _AlternativeResidualStorage(
                    shape,
                    method,
                    indices=selected.to(torch.uint16),
                    values=source.reshape(-1).index_select(0, selected),
                )
            elif method == "contrast":
                basis = result._contrast_basis.to(device=source.device, dtype=source.dtype)
                node = _AlternativeResidualStorage(shape, method, values=basis.T @ centered)
            elif method == "svd2":
                basis = result._contrast_basis.to(device=source.device, dtype=source.dtype)
                contrast = basis.T @ centered
                left_vectors, singular_values, right_vectors = torch.linalg.svd(
                    contrast, full_matrices=False
                )
                root_singular = singular_values[:2].clamp_min(0).sqrt()
                left = left_vectors[:, :2] * root_singular.unsqueeze(0)
                right = root_singular.unsqueeze(1) * right_vectors[:2, :]
                node = _AlternativeResidualStorage(shape, method, left=left, right=right)
            elif method == "columns32":
                rms = cls._feature_rms(node_phi, source)
                scores = torch.linalg.vector_norm(centered, dim=0) * rms
                non_bias = torch.arange(shape[1] - 1, device=source.device)
                order = torch.argsort(scores.index_select(0, non_bias), descending=True, stable=True)
                selected = non_bias.index_select(0, order[: _COLUMN_COUNT - 1])
                selected = torch.cat([selected, torch.tensor([shape[1] - 1], device=source.device)]).sort().values
                node = _AlternativeResidualStorage(
                    shape,
                    method,
                    indices=selected.to(torch.uint16),
                    values=centered.index_select(1, selected),
                )
            elif method == "bf16":
                node = _AlternativeResidualStorage(shape, method, values=source.to(torch.bfloat16))
            elif method == "center_topk50":
                selected = cls._top_coordinates(centered.abs(), _TOPK_COUNT)
                node = _AlternativeResidualStorage(
                    shape,
                    method,
                    indices=selected.to(torch.uint16),
                    values=centered.reshape(-1).index_select(0, selected),
                )
            else:  # activation_topk50
                rms = cls._feature_rms(node_phi, source)
                selected = cls._top_coordinates(centered.abs() * rms.unsqueeze(0), _TOPK_COUNT)
                node = _AlternativeResidualStorage(
                    shape,
                    method,
                    indices=selected.to(torch.uint16),
                    values=centered.reshape(-1).index_select(0, selected),
                )

            result._nodes[str(node_id)] = node
            metadata["nodes"][str(node_id)] = {
                "learned_coordinates": sum(parameter.numel() for parameter in node.residual_parameters()),
                "index_count": int(node.indices.numel()) if hasattr(node, "indices") else 0,
                "calibration_observations": int(node_phi.shape[0]) if node_phi is not None else 0,
            }

        result._set_trainable_flags(train_critic=result.train_critic)
        return result, metadata

    @staticmethod
    def _top_coordinates(scores: torch.Tensor, count: int) -> torch.Tensor:
        flat = scores.reshape(-1)
        selected = torch.argsort(flat, descending=True, stable=True)[:count]
        return selected.sort().values

    @staticmethod
    def _calibration_features(
        reference: PackedResidualPolicy,
        calibration_obs: torch.Tensor | np.ndarray | Mapping[int | str, Any] | None,
    ) -> dict[int, torch.Tensor]:
        if calibration_obs is None:
            return {}

        batches: list[tuple[int | None, Any]] = []
        if isinstance(calibration_obs, Mapping):
            batches.extend((int(route), obs) for route, obs in calibration_obs.items())
        else:
            batches.append((None, calibration_obs))

        by_route: dict[int, list[torch.Tensor]] = {}
        with torch.no_grad():
            for fixed_route, observations in batches:
                batch = reference._as_batch(observations)
                if fixed_route is None:
                    routes = batch[:, -1].round().long()
                else:
                    routes = torch.full(
                        (batch.shape[0],), int(fixed_route), dtype=torch.long, device=batch.device
                    )
                features = reference.extract_features(batch)
                latent_pi = reference.mlp_extractor.forward_actor(features)
                phi = torch.cat([latent_pi, torch.ones_like(latent_pi[:, :1])], dim=1)
                for route in torch.unique(routes).tolist():
                    selected = phi[routes == int(route)].detach()
                    by_route.setdefault(int(route), []).append(selected)
        return {route: torch.cat(values, dim=0) for route, values in by_route.items()}

    @staticmethod
    def _node_calibration_phi(
        reference: PackedResidualPolicy,
        node_id: int,
        route_phi: Mapping[int, torch.Tensor],
    ) -> torch.Tensor | None:
        # Use calibration from the node's own route where available. For an
        # inherited residual, descendants that include the node are the next
        # closest route-specific sample.
        if node_id in route_phi and route_phi[node_id].numel():
            return route_phi[node_id]
        descendant_routes = [
            route_phi[endpoint]
            for endpoint, path in reference._paths.items()
            if node_id in path and endpoint in route_phi and route_phi[endpoint].numel()
        ]
        if descendant_routes:
            return torch.cat(descendant_routes, dim=0)
        if route_phi:
            return torch.cat(list(route_phi.values()), dim=0)
        return None

    @staticmethod
    def _feature_rms(phi: torch.Tensor | None, source: torch.Tensor) -> torch.Tensor:
        if phi is None or phi.shape[0] == 0:
            return torch.ones(source.shape[1], dtype=source.dtype, device=source.device)
        return phi.to(device=source.device, dtype=source.dtype).square().mean(dim=0).sqrt()

    def _set_trainable_flags(self, *, train_critic: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if train_critic:
            for parameter in self.mlp_extractor.value_net.parameters():
                parameter.requires_grad_(True)
            for parameter in self.value_net.parameters():
                parameter.requires_grad_(True)
        for node in self._nodes.values():
            for parameter in node.residual_parameters():
                parameter.requires_grad_(True)
                parameter.tree_residual = True

    def _node_contribution(self, node_id: int, phi: torch.Tensor) -> torch.Tensor:
        node = self._nodes[str(node_id)]
        if node.method in {"dense", "bf16"}:
            return F.linear(phi, node.values.to(dtype=phi.dtype))
        if node.method in {"topk50", "center_topk50", "activation_topk50"}:
            indices = node.indices.to(device=phi.device, dtype=torch.long)
            columns = torch.remainder(indices, node.shape[1])
            rows = torch.div(indices, node.shape[1], rounding_mode="floor")
            terms = phi.index_select(1, columns) * node.values.to(dtype=phi.dtype).unsqueeze(0)
            row_indices = rows.unsqueeze(0).expand(phi.shape[0], -1)
            return phi.new_zeros((phi.shape[0], node.shape[0])).scatter_add(1, row_indices, terms)
        if node.method == "contrast":
            basis = self._contrast_basis.to(device=phi.device, dtype=phi.dtype)
            contrasts = F.linear(phi, node.values.to(dtype=phi.dtype))
            return F.linear(contrasts, basis)
        if node.method == "svd2":
            basis = self._contrast_basis.to(device=phi.device, dtype=phi.dtype)
            reduced = F.linear(phi, node.right.to(dtype=phi.dtype))
            contrasts = F.linear(reduced, node.left.to(dtype=phi.dtype))
            return F.linear(contrasts, basis)
        if node.method == "columns32":
            indices = node.indices.to(device=phi.device, dtype=torch.long)
            return F.linear(phi.index_select(1, indices), node.values.to(dtype=phi.dtype))
        raise RuntimeError(f"unknown residual layout {node.method!r}")

    def make_optimizer(
        self,
        *,
        learning_rate: float = 1e-2,
        transfer: bool = False,
    ) -> LeafAdam:
        """Create a fresh LeafAdam; alternative layouts never transfer moments."""
        if transfer:
            raise ValueError("alternative residual policies always start with fresh optimizer state")
        self._set_trainable_flags(train_critic=self.train_critic)
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("alternative residual policy has no trainable parameters")
        optimizer = LeafAdam(trainable, lr=float(learning_rate), eps=1e-5)
        optimizer.step_count = 0
        return optimizer

    def to_spec(self) -> dict[str, Any]:
        if not isinstance(self.observation_space, spaces.Box):
            raise ValueError("only Box observation spaces can be described")
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("only Discrete action spaces can be described")
        return {
            "format_version": _FORMAT_VERSION,
            "architecture": _SUPPORTED_ARCHITECTURE,
            "layout": "alternative_residual_v1",
            "alternative_method": self._storage,
            "storage": self._storage,
            "max_nodes": self.max_nodes,
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
            "node_shapes": {str(node): list(shape) for node, shape in sorted(self._node_shapes.items())},
            "coordinate_counts": {
                str(node_id): sum(parameter.numel() for parameter in node.residual_parameters())
                for node_id, node in sorted((int(key), value) for key, value in self._nodes.items())
            },
            "index_counts": {
                str(node_id): int(node.indices.numel()) if hasattr(node, "indices") else 0
                for node_id, node in sorted((int(key), value) for key, value in self._nodes.items())
            },
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "AlternativeResidualPolicy":
        if int(spec.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported alternative policy spec version")
        if spec.get("architecture") != _SUPPORTED_ARCHITECTURE:
            raise ValueError("unsupported alternative policy architecture")
        method = str(spec.get("alternative_method", ""))
        if method not in _METHODS:
            raise ValueError(f"unsupported alternative policy method {method!r}")
        obs = spec["observation_space"]
        if obs.get("kind") != "Box":
            raise ValueError("only Box observation spaces are supported")
        observation_space = spaces.Box(
            low=np.asarray(obs["low"], dtype=np.dtype(obs["dtype"])),
            high=np.asarray(obs["high"], dtype=np.dtype(obs["dtype"])),
            dtype=np.dtype(obs["dtype"]),
        )
        action = spec["action_space"]
        if action.get("kind") != "Discrete":
            raise ValueError("only Discrete action spaces are supported")
        max_nodes = int(spec["max_nodes"])
        base = TreePolicy(
            observation_space=observation_space,
            action_space=spaces.Discrete(int(action["n"])),
            lr_schedule=lambda _: 1e-2,
            max_nodes=max_nodes,
        )
        paths = {int(endpoint): tuple(int(node) for node in path) for endpoint, path in spec["paths"].items()}
        result = cls._copy_base(
            base, paths, storage=method, train_critic=bool(spec.get("train_critic", True))
        )
        result._storage = method
        result._node_shapes = {
            int(node): tuple(int(part) for part in shape)
            for node, shape in spec["node_shapes"].items()
        }
        result._nodes = nn.ModuleDict()
        index_counts = {int(node): int(count) for node, count in spec.get("index_counts", {}).items()}
        for node_id, shape in sorted(result._node_shapes.items()):
            count = index_counts.get(node_id, 0)
            indices = torch.zeros(count, dtype=torch.uint16) if count else None
            result._nodes[str(node_id)] = _AlternativeResidualStorage(shape, method, indices=indices)
        result._set_trainable_flags(train_critic=result.train_critic)
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        if int(payload.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported alternative policy payload version")
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

    def storage_report(self) -> dict[str, int | str]:
        residual_parameters = [
            parameter
            for node in self._nodes.values()
            for parameter in node.residual_parameters()
        ]
        weight_factor_bytes = sum(parameter.numel() * parameter.element_size() for parameter in residual_parameters)
        index_bytes = sum(
            node.indices.numel() * node.indices.element_size()
            for node in self._nodes.values()
            if hasattr(node, "indices")
        )
        total_bytes = self.persistent_tensor_bytes()
        return {
            "storage": self._storage,
            "dtype": "bfloat16" if self._storage == "bf16" else "float32",
            "persistent_policy_tensor_bytes": total_bytes,
            "total_policy_bytes": total_bytes,
            "base_and_head_tensor_bytes": total_bytes - weight_factor_bytes - index_bytes,
            "residual_weight_factor_bytes": weight_factor_bytes,
            "weight_factor_bytes": weight_factor_bytes,
            "residual_value_bytes": weight_factor_bytes,
            "coordinate_index_bytes": index_bytes,
            "index_bytes": index_bytes,
            "retained_coordinates": sum(parameter.numel() for parameter in residual_parameters),
            "learned_coordinate_count": sum(parameter.numel() for parameter in residual_parameters),
        }


def compress_policy(
    reference: PackedResidualPolicy,
    method: str,
    calibration_obs: torch.Tensor | np.ndarray | Mapping[int | str, Any] | None = None,
) -> tuple[AlternativeResidualPolicy, dict[str, Any]]:
    """Build an alternative active-only residual policy and its construction report."""
    return AlternativeResidualPolicy.from_reference(reference, method, calibration_obs=calibration_obs)


def restore_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Restore either an alternative layout or a legacy packed policy payload."""
    spec = payload.get("spec", {})
    if spec.get("layout") == "alternative_residual_v1" or spec.get("alternative_method"):
        return AlternativeResidualPolicy.from_payload(payload)
    return PackedResidualPolicy.from_payload(payload)

