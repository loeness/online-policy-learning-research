"""Compact storage for the path residuals used by :mod:`online_rl.policy`.

``PackedResidualPolicy.from_dense`` copies a ``TreePolicy`` foundation and
stores only residuals reachable from the requested route endpoints.  The
default ``storage="packed"`` keeps each residual's selected float32 values and
one uint16 flat-coordinate buffer.  ``storage="dense"`` is the active-only
dense control: it stores the same reachable residuals as full matrices and has
no coordinate buffers.

The packed module computes residual contributions from coordinates directly;
it never keeps a dense residual matrix or dense path mask in its state.  Its
coordinate conversion to int64 is a temporary runtime tensor.  It does not
claim sparse hardware acceleration.

Example::

    packed = PackedResidualPolicy.from_dense(policy, active_nodes=[1, 2])
    optimizer = packed.make_optimizer(dense_optimizer=policy.optimizer)
    bundle = export_payload(packed, optimizer, router.snapshot(), metadata={})
    torch.save(bundle, "packed.pt")

    saved = torch.load("packed.pt", map_location="cpu", weights_only=False)
    restored = PackedResidualPolicy.from_payload(saved)
    policy, optimizer = restored["policy"], restored["optimizer"]
"""
from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.preprocessing import preprocess_obs
from torch import nn

from .policy import LeafAdam, TreePolicy


_FORMAT_VERSION = 1
_SUPPORTED_ARCHITECTURE = "TreePolicy-SplitNetwork-v1"


class _ResidualStorage(nn.Module):
    """One residual in either packed-coordinate or active-only dense form."""

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        storage: str,
        values: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        count: int | None = None,
    ) -> None:
        super().__init__()
        self.storage = storage
        self.shape = tuple(int(part) for part in shape)
        if storage == "packed":
            if indices is None:
                indices = torch.zeros(int(count or 0), dtype=torch.uint16)
            if values is None:
                values = torch.zeros(int(indices.numel()), dtype=torch.float32)
            self.register_buffer("indices", indices.to(dtype=torch.uint16), persistent=True)
            values = values.reshape(-1)
        elif storage == "dense":
            if values is None:
                values = torch.zeros(self.shape, dtype=torch.float32)
            values = values.reshape(self.shape)
        else:
            raise ValueError("storage must be 'packed' or 'dense'")
        self.values = nn.Parameter(values.detach().clone())
        # LeafAdam recognizes these parameters to avoid advancing idle moments.
        self.values.tree_residual = True


class PackedResidualPolicy(nn.Module):
    """A frozen-base actor critic with compact path residuals.

    ``active_nodes`` are route endpoint IDs.  If ``path_map`` is omitted,
    paths are read from ``TreePolicy.path_masks``.  ``path_map`` maps each
    endpoint to the residual node IDs that contribute, in any order; node 0
    is always removed because the base actor is the root.  Alternatively,
    ``parents`` maps each residual node to its single parent and paths are
    built from root to endpoint.

    ``storage="packed"`` stores only the largest-magnitude nonzero
    coordinates.  ``retain_fraction=1`` preserves every exactly nonzero
    value.  For a fraction below 1, ``ceil(fraction * nonzero_count)`` values
    are kept per residual node.  ``storage="dense"`` stores a full matrix for
    each reachable node and requires a retention fraction of 1; it is useful
    as the active-only dense control.

    The base ``TreePolicy`` modules are copied once: feature extractor,
    ``SplitNetwork`` actor and critic, action head, and value head.  The
    original ``residuals`` and ``path_masks`` modules are not retained.
    """

    def __init__(self) -> None:
        super().__init__()
        # Instances are created by from_dense/from_spec after setting modules.
        self._paths: dict[int, tuple[int, ...]] = {}
        self._node_shapes: dict[int, tuple[int, int]] = {}
        self._storage = "packed"
        self.max_nodes = 0
        self.train_critic = True
        self._nodes = nn.ModuleDict()

    @classmethod
    def from_dense(
        cls,
        policy: TreePolicy,
        active_nodes: Sequence[int],
        *,
        retain_fraction: float = 1.0,
        parents: Mapping[int, int] | None = None,
        path_map: Mapping[int, Sequence[int]] | None = None,
        storage: str = "packed",
        train_critic: bool = True,
    ) -> "PackedResidualPolicy":
        """Copy a ``TreePolicy`` into packed or active-only dense storage.

        ``active_nodes`` lists route endpoints, including endpoint IDs which
        may inherit one or more residuals.  Path entries refer to residual
        node IDs, not route-mask rows.  Without an explicit path description,
        the path mask rows in ``policy`` are used.
        """
        if not isinstance(policy, TreePolicy):
            raise TypeError("from_dense currently supports online_rl.policy.TreePolicy")
        if storage not in {"packed", "dense"}:
            raise ValueError("storage must be 'packed' or 'dense'")
        retain_fraction = float(retain_fraction)
        if not 0.0 < retain_fraction <= 1.0:
            raise ValueError("retain_fraction must be in (0, 1]")
        if storage == "dense" and retain_fraction != 1.0:
            raise ValueError("storage='dense' is the full active-only dense control")
        if parents is not None and path_map is not None:
            raise ValueError("pass either parents or path_map, not both")

        endpoints = sorted({int(node) for node in active_nodes})
        if any(node < 0 or node >= policy.max_nodes for node in endpoints):
            raise ValueError("active route endpoint is outside policy.max_nodes")
        paths = cls._resolve_paths(policy, endpoints, parents=parents, path_map=path_map)
        stored_node_ids = sorted({node for path in paths.values() for node in path if node != 0})

        # Parameter names are attached to source parameters so optimizer state
        # can be transferred by name without retaining the dense policy.
        for name, parameter in policy.named_parameters():
            parameter._packed_source_name = name

        result = cls._copy_base(policy, paths, storage=storage, train_critic=train_critic)
        for node_id in stored_node_ids:
            source = policy.residuals[node_id].detach()
            shape = tuple(int(part) for part in source.shape)
            if len(shape) != 2:
                raise ValueError(f"residual {node_id} must be a matrix, got {shape}")
            result._node_shapes[node_id] = shape
            flat = source.reshape(-1)
            if storage == "dense":
                node = _ResidualStorage(shape, storage="dense", values=source)
            else:
                nonzero = torch.nonzero(flat != 0, as_tuple=False).reshape(-1)
                if nonzero.numel() == 0:
                    selected = nonzero
                elif retain_fraction == 1.0:
                    selected = nonzero
                else:
                    keep = max(1, int(math.ceil(retain_fraction * nonzero.numel())))
                    magnitudes = flat.index_select(0, nonzero).abs()
                    order = torch.argsort(magnitudes, descending=True, stable=True)[:keep]
                    selected = nonzero.index_select(0, order).sort().values
                indices = selected.to(dtype=torch.uint16)
                values = flat.index_select(0, selected.to(dtype=torch.long))
                node = _ResidualStorage(shape, storage="packed", values=values, indices=indices)
            result._nodes[str(node_id)] = node
        result._set_trainable_flags(train_critic=train_critic)
        return result

    @classmethod
    def _copy_base(
        cls,
        policy: TreePolicy,
        paths: Mapping[int, Sequence[int]],
        *,
        storage: str,
        train_critic: bool,
    ) -> "PackedResidualPolicy":
        result = cls()
        result.features_extractor = copy.deepcopy(policy.features_extractor)
        result.mlp_extractor = copy.deepcopy(policy.mlp_extractor)
        result.action_net = copy.deepcopy(policy.action_net)
        result.value_net = copy.deepcopy(policy.value_net)
        result.action_dist = make_proba_distribution(policy.action_space)
        result.observation_space = copy.deepcopy(policy.observation_space)
        result.action_space = copy.deepcopy(policy.action_space)
        result.normalize_images = bool(getattr(policy, "normalize_images", False))
        result.features_dim = int(policy.features_dim)
        result.max_nodes = int(policy.max_nodes)
        result._storage = storage
        result._paths = {int(endpoint): tuple(int(n) for n in path if int(n) != 0)
                         for endpoint, path in paths.items()}
        result.train_critic = bool(train_critic)
        return result

    @staticmethod
    def _resolve_paths(
        policy: TreePolicy,
        endpoints: Sequence[int],
        *,
        parents: Mapping[int, int] | None,
        path_map: Mapping[int, Sequence[int]] | None,
    ) -> dict[int, tuple[int, ...]]:
        paths: dict[int, tuple[int, ...]] = {}
        for endpoint in endpoints:
            if path_map is not None:
                if endpoint not in path_map and str(endpoint) not in path_map:
                    raise ValueError(f"path_map has no entry for endpoint {endpoint}")
                source_path = path_map.get(endpoint, path_map.get(str(endpoint)))
                path = tuple(int(node) for node in source_path)
            elif parents is not None:
                chain = []
                node = endpoint
                visited = set()
                while node != 0:
                    if node in visited:
                        raise ValueError(f"cycle detected in parents at node {node}")
                    visited.add(node)
                    chain.append(node)
                    parent = parents.get(node, parents.get(str(node), 0))
                    node = int(parent)
                path = tuple(reversed(chain))
            else:
                path = tuple(
                    int(node)
                    for node in torch.nonzero(policy.path_masks[endpoint], as_tuple=False).reshape(-1).tolist()
                    if int(node) != 0
                )
            if any(node < 0 or node >= policy.max_nodes for node in path):
                raise ValueError(f"path for endpoint {endpoint} contains an invalid node ID")
            paths[endpoint] = tuple(dict.fromkeys(node for node in path if node != 0))
        return paths

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "PackedResidualPolicy":
        """Recreate the module layout described by :meth:`to_spec`.

        The current descriptor intentionally supports the existing
        ``TreePolicy``/``SplitNetwork`` architecture.  Packed coordinate values
        are placeholders here; ``load_state_dict`` fills the one serialized
        coordinate buffer and parameter values for each node.
        """
        if int(spec.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported packed policy spec version")
        if spec.get("architecture") != _SUPPORTED_ARCHITECTURE:
            raise ValueError("unsupported packed policy architecture")
        obs = spec["observation_space"]
        if obs.get("kind") != "Box":
            raise ValueError("only Box observation spaces are supported by from_spec")
        observation_space = spaces.Box(
            low=np.asarray(obs["low"], dtype=np.dtype(obs["dtype"])),
            high=np.asarray(obs["high"], dtype=np.dtype(obs["dtype"])),
            dtype=np.dtype(obs["dtype"]),
        )
        action = spec["action_space"]
        if action.get("kind") != "Discrete":
            raise ValueError("only Discrete action spaces are supported by from_spec")
        max_nodes = int(spec["max_nodes"])
        base = TreePolicy(
            observation_space=observation_space,
            action_space=spaces.Discrete(int(action["n"])),
            lr_schedule=lambda _: 1e-2,
            max_nodes=max_nodes,
        )
        paths = {int(endpoint): tuple(int(node) for node in path)
                 for endpoint, path in spec["paths"].items()}
        storage = str(spec["storage"])
        if storage not in {"packed", "dense"}:
            raise ValueError(f"invalid storage kind {storage!r}")
        result = cls._copy_base(
            base, paths, storage=storage, train_critic=bool(spec.get("train_critic", True))
        )
        node_shapes = {int(node): tuple(int(n) for n in shape)
                       for node, shape in spec["node_shapes"].items()}
        counts = {int(node): int(count) for node, count in spec["coordinate_counts"].items()}
        for node_id in sorted(node_shapes):
            shape = node_shapes[node_id]
            result._node_shapes[node_id] = shape
            if storage == "packed":
                count = counts[node_id]
                result._nodes[str(node_id)] = _ResidualStorage(
                    shape, storage="packed", indices=torch.zeros(count, dtype=torch.uint16), count=count
                )
            else:
                if counts[node_id] != math.prod(shape):
                    raise ValueError(f"dense coordinate count for node {node_id} does not match its shape")
                result._nodes[str(node_id)] = _ResidualStorage(shape, storage="dense")
        result._set_trainable_flags(train_critic=result.train_critic)
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Recreate policy, named optimizer state, and companion metadata."""
        if int(payload.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported packed payload version")
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

    def _set_trainable_flags(self, *, train_critic: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if train_critic:
            for parameter in self.mlp_extractor.value_net.parameters():
                parameter.requires_grad_(True)
            for parameter in self.value_net.parameters():
                parameter.requires_grad_(True)
        for node in self._nodes.values():
            node.values.requires_grad_(True)
            node.values.tree_residual = True

    def to_spec(self) -> dict[str, Any]:
        """Return layout metadata without duplicating packed coordinates."""
        if not isinstance(self.observation_space, spaces.Box):
            raise ValueError("only Box observation spaces can be described")
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("only Discrete action spaces can be described")
        dtype = np.dtype(self.observation_space.dtype).name
        return {
            "format_version": _FORMAT_VERSION,
            "architecture": _SUPPORTED_ARCHITECTURE,
            "storage": self._storage,
            "max_nodes": self.max_nodes,
            "train_critic": self.train_critic,
            "observation_space": {
                "kind": "Box",
                "dtype": dtype,
                "low": np.asarray(self.observation_space.low).tolist(),
                "high": np.asarray(self.observation_space.high).tolist(),
            },
            "action_space": {"kind": "Discrete", "n": int(self.action_space.n)},
            "feature_dim": self.features_dim,
            "paths": {str(endpoint): list(path) for endpoint, path in sorted(self._paths.items())},
            "node_shapes": {str(node): list(shape) for node, shape in sorted(self._node_shapes.items())},
            # Counts shape placeholder tensors on reconstruction; actual indices
            # occur only in the policy state_dict.
            "coordinate_counts": {
                str(node): int(storage.indices.numel()) if self._storage == "packed"
                else int(storage.values.numel())
                for node, storage in sorted((int(k), v) for k, v in self._nodes.items())
            },
        }

    def _as_batch(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        reference = next(self.parameters())
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=reference.dtype, device=reference.device)
        else:
            obs = obs.to(device=reference.device, dtype=reference.dtype)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs

    def extract_features(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        obs = self._as_batch(obs)
        preprocessed = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return self.features_extractor(preprocessed)

    def _node_contribution(self, node_id: int, phi: torch.Tensor) -> torch.Tensor:
        node = self._nodes[str(node_id)]
        if self._storage == "dense":
            return torch.nn.functional.linear(phi, node.values)
        indices = node.indices.to(device=phi.device, dtype=torch.long)
        if indices.numel() == 0:
            return phi.new_zeros((phi.shape[0], node.shape[0]))
        rows = torch.div(indices, node.shape[1], rounding_mode="floor")
        columns = torch.remainder(indices, node.shape[1])
        terms = phi.index_select(1, columns) * node.values.to(dtype=phi.dtype).unsqueeze(0)
        row_index = rows.unsqueeze(0).expand(phi.shape[0], -1)
        result = phi.new_zeros((phi.shape[0], node.shape[0]))
        return result.scatter_add(1, row_index, terms)

    def _correction(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        ids = obs[:, -1].round().long().clamp(0, self.max_nodes - 1)
        phi = torch.cat([latent_pi, torch.ones_like(latent_pi[:, :1])], dim=1)
        result = latent_pi.new_zeros((latent_pi.shape[0], self.action_space.n))
        for endpoint, path in self._paths.items():
            if not path:
                continue
            batch_indices = torch.nonzero(ids == endpoint, as_tuple=False).reshape(-1)
            if batch_indices.numel() == 0:
                continue
            local_phi = phi.index_select(0, batch_indices)
            local = latent_pi.new_zeros((batch_indices.numel(), self.action_space.n))
            for node_id in path:
                local = local + self._node_contribution(node_id, local_phi)
            result = result.index_add(0, batch_indices, local)
        return result

    def distribution(self, obs: torch.Tensor | np.ndarray, latent_pi: torch.Tensor):
        obs = self._as_batch(obs)
        logits = self.action_net(latent_pi) + self._correction(obs, latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic: bool = False):
        obs = self._as_batch(obs)
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        distribution = self.distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(latent_vf), distribution.log_prob(actions)

    def get_distribution(self, obs):
        obs = self._as_batch(obs)
        latent_pi = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self.distribution(obs, latent_pi)

    def evaluate_actions(self, obs, actions):
        obs = self._as_batch(obs)
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        distribution = self.distribution(obs, latent_pi)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs):
        features = self.extract_features(obs)
        return self.value_net(self.mlp_extractor.forward_critic(features))

    def make_optimizer(
        self,
        dense_optimizer: torch.optim.Optimizer | None = None,
        *,
        learning_rate: float = 1e-2,
        transfer: bool = True,
        train_critic: bool | None = None,
    ) -> LeafAdam:
        """Create the sparse optimizer, optionally mapping dense Adam state.

        Base actor/head parameters remain frozen.  Critic parameters are
        trainable by default and can be disabled with ``train_critic=False``.
        Dense residual moments are gathered at each stored coordinate by their
        source parameter name.  Per-parameter Adam steps, group options, and
        the optimizer's custom ``step_count`` are retained when transfer is
        enabled.
        """
        if train_critic is not None:
            self.train_critic = bool(train_critic)
        self._set_trainable_flags(train_critic=self.train_critic)
        target_named = dict(self.named_parameters())
        groups: list[dict[str, Any]] = []
        if dense_optimizer is not None and transfer:
            for dense_group in dense_optimizer.param_groups:
                target_parameters = []
                source_pairs = []
                for source_parameter in dense_group["params"]:
                    source_name = getattr(source_parameter, "_packed_source_name", None)
                    if source_name is None:
                        continue
                    target = self._target_for_dense_name(source_name, target_named)
                    if target is None or not target.requires_grad:
                        continue
                    if any(target is prior for prior in target_parameters):
                        continue
                    target_parameters.append(target)
                    source_pairs.append((source_parameter, target, source_name))
                if not target_parameters:
                    continue
                options = {key: copy.deepcopy(value) for key, value in dense_group.items() if key != "params"}
                groups.append({**options, "params": target_parameters})
                for source_parameter, target, source_name in source_pairs:
                    state = dense_optimizer.state.get(source_parameter)
                    if state:
                        self._copy_dense_optimizer_state(source_name, source_parameter, target, state)
            if not groups:
                raise ValueError("dense optimizer has no parameters matching the packed trainable policy")
            optimizer = LeafAdam(groups)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    state = getattr(parameter, "_packed_pending_optimizer_state", None)
                    if state is not None:
                        optimizer.state[parameter] = state
                        delattr(parameter, "_packed_pending_optimizer_state")
            optimizer.step_count = int(getattr(dense_optimizer, "step_count", 0))
            return optimizer

        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("packed policy has no trainable parameters")
        optimizer = LeafAdam(trainable, lr=float(learning_rate), eps=1e-5)
        optimizer.step_count = 0
        return optimizer

    def _target_for_dense_name(
        self, source_name: str, target_named: Mapping[str, nn.Parameter]
    ) -> nn.Parameter | None:
        if source_name.startswith("residuals."):
            try:
                node_id = int(source_name.split(".", 1)[1])
            except ValueError:
                return None
            if node_id == 0 or str(node_id) not in self._nodes:
                return None
            return self._nodes[str(node_id)].values
        return target_named.get(source_name)

    def _copy_dense_optimizer_state(
        self,
        source_name: str,
        source_parameter: nn.Parameter,
        target_parameter: nn.Parameter,
        state: Mapping[str, Any],
    ) -> None:
        copied = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                copied[key] = copy.deepcopy(value)
                continue
            mapped = value.detach().clone()
            if (
                source_name.startswith("residuals.")
                and self._storage == "packed"
                and tuple(value.shape) == tuple(source_parameter.shape)
            ):
                node_id = int(source_name.split(".", 1)[1])
                coordinates = self._nodes[str(node_id)].indices.to(device=value.device, dtype=torch.long)
                mapped = value.reshape(-1).index_select(0, coordinates)
            copied[key] = mapped
        self._nodes_or_target_state(target_parameter, copied)

    @staticmethod
    def _nodes_or_target_state(parameter: nn.Parameter, state: dict[str, Any]) -> None:
        # Assignment is made by caller's optimizer state in make_optimizer.
        parameter._packed_pending_optimizer_state = state

    def _optimizer_from_manifest(self, manifest: Mapping[str, Any]) -> LeafAdam:
        self._set_trainable_flags(train_critic=self.train_critic)
        named = dict(self.named_parameters())
        groups = []
        for saved_group in manifest["param_groups"]:
            params = []
            for name in saved_group["param_names"]:
                if name not in named:
                    raise ValueError(f"optimizer refers to unknown packed parameter {name!r}")
                params.append(named[name])
            options = copy.deepcopy(saved_group["options"])
            groups.append({**options, "params": params})
        if not groups:
            raise ValueError("packed optimizer manifest has no parameter groups")
        optimizer = LeafAdam(groups)
        for saved_name, saved_state in manifest.get("state", {}).items():
            parameter = named.get(saved_name)
            if parameter is None:
                raise ValueError(f"optimizer state refers to unknown packed parameter {saved_name!r}")
            optimizer.state[parameter] = {
                key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in saved_state.items()
            }
        optimizer.step_count = int(manifest.get("step_count", 0))
        return optimizer

    def optimizer_manifest(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Return optimizer groups and states keyed by packed parameter names."""
        names_by_id = {id(parameter): name for name, parameter in self.named_parameters()}
        param_groups = []
        for group in optimizer.param_groups:
            names = []
            for parameter in group["params"]:
                name = names_by_id.get(id(parameter))
                if name is None:
                    raise ValueError("optimizer contains a parameter outside this policy")
                names.append(name)
            options = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
            param_groups.append({"param_names": names, "options": options})
        state = {}
        for parameter, values in optimizer.state.items():
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer state contains a parameter outside this policy")
            state[name] = {
                key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in values.items()
            }
        return {
            "class": "LeafAdam",
            "param_groups": param_groups,
            "state": state,
            "step_count": int(getattr(optimizer, "step_count", 0)),
        }

    def persistent_tensor_bytes(self) -> int:
        """Count exact persistent policy tensor bytes, including coordinates."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self.state_dict().values())

    def storage_report(self) -> dict[str, int | str]:
        """Report tensor bytes without counting transient runtime workspaces."""
        values_bytes = sum(node.values.numel() * node.values.element_size() for node in self._nodes.values())
        indices_bytes = (
            sum(node.indices.numel() * node.indices.element_size() for node in self._nodes.values())
            if self._storage == "packed" else 0
        )
        return {
            "storage": self._storage,
            "persistent_policy_tensor_bytes": self.persistent_tensor_bytes(),
            "base_and_head_tensor_bytes": self.persistent_tensor_bytes() - values_bytes - indices_bytes,
            "residual_value_bytes": values_bytes,
            "coordinate_index_bytes": indices_bytes,
            "retained_coordinates": sum(node.values.numel() for node in self._nodes.values()),
        }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def export_payload(
    policy: PackedResidualPolicy,
    optimizer: torch.optim.Optimizer,
    router_snapshot: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
    rng: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Package the compact policy, named optimizer, router, and RNG state.

    The returned object is ready for ``torch.save``.  State tensors are cloned
    so subsequent optimization cannot mutate the snapshot in memory.
    """
    return {
        "format_version": _FORMAT_VERSION,
        "spec": policy.to_spec(),
        "policy": {key: value.detach().clone() for key, value in policy.state_dict().items()},
        "optimizer": policy.optimizer_manifest(optimizer),
        "router": copy.deepcopy(router_snapshot),
        "metadata": copy.deepcopy(dict(metadata or {})),
        "rng": copy.deepcopy(dict(rng) if rng is not None else _capture_rng_state()),
    }


def save_payload(
    path: str | Path,
    policy: PackedResidualPolicy,
    optimizer: torch.optim.Optimizer,
    router_snapshot: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
    rng: Mapping[str, Any] | None = None,
) -> int:
    """Write a packed payload with ``torch.save`` and return serialized bytes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = export_payload(policy, optimizer, router_snapshot, metadata, rng)
    torch.save(payload, destination)
    return destination.stat().st_size


def load_payload(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a payload produced by :func:`save_payload`."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return PackedResidualPolicy.from_payload(payload)
