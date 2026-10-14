"""Flat group controls for the six-context C1 residual policy.

``FlatGroupedAdditivePolicy`` stores the same family and leaf adapters as the
existing C1 tree layout, then selects them through flat membership tables.
It is an implementation control for tree traversal, not a new adapter
algorithm.

``GroupedDictionaryPolicy`` gives each family one learned feature basis and
each leaf its own coefficient matrix.  It has no parent path: family fallback
routes average the residuals of the leaves marked as seen in that family.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .c1_policy import (
    C1ResidualPolicy,
    _ENDPOINTS,
    _LOCAL_PHI,
    _PARENTS,
    _contrast_basis,
    _mark_residual,
)
from .policy import TreePolicy


_GROUPS = {7: (1, 2, 3), 8: (4, 5, 6)}
_FAMILY_FOR_ENDPOINT = {endpoint: parent for parent, members in _GROUPS.items() for endpoint in members}
_GROUPED_METHOD = "grouped_additive"
_DICTIONARY_METHOD = "grouped_dictionary"


class FlatGroupedAdditivePolicy(C1ResidualPolicy):
    """Flat membership-table equivalent of ``C1ResidualPolicy(method='tree')``.

    The family adapter and leaf adapter tensors, shapes, names, and initial
    values match the tree control.  Only the route-to-adapter lookup differs.
    """

    @classmethod
    def from_dense(
        cls,
        policy: TreePolicy,
        *,
        method: str = _GROUPED_METHOD,
        rank: int = 1,
        dictionary_size: int = 6,
        epsilon: float = 0.05,
        temperature: float = 1.0,
        train_critic: bool = True,
    ) -> "FlatGroupedAdditivePolicy":
        if method != _GROUPED_METHOD:
            raise ValueError(f"method must be {_GROUPED_METHOD!r}")
        result = super().from_dense(
            policy,
            method="tree",
            rank=rank,
            dictionary_size=dictionary_size,
            epsilon=epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )
        result._paths = {}
        return result

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        method: str = _GROUPED_METHOD,
        leaf_rank: int = 1,
        exploration_epsilon: float = 0.05,
        temperature: float = 1.0,
        train_critic: bool = True,
    ) -> "FlatGroupedAdditivePolicy":
        """Create the flat grouped control from the shared frozen foundation."""
        return cls.from_dense(
            reference,
            method=method,
            rank=leaf_rank,
            epsilon=exploration_epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )

    @classmethod
    def from_tree(cls, treecore: C1ResidualPolicy) -> "FlatGroupedAdditivePolicy":
        """Copy a tree core exactly while retaining transferable parameter names."""
        if not isinstance(treecore, C1ResidualPolicy) or treecore._storage != "tree":
            raise TypeError("from_tree requires a C1ResidualPolicy with method='tree'")
        paths = {endpoint: (_PARENTS[endpoint], endpoint) for endpoint in _ENDPOINTS}
        result = cls._copy_base(
            treecore,
            paths,
            storage="tree",
            train_critic=treecore.train_critic,
        )
        result.max_nodes = treecore.max_nodes
        result.rank = treecore.rank
        result.dictionary_size = treecore.dictionary_size
        result.epsilon = treecore.epsilon
        result.temperature = treecore.temperature
        result._paths = {}
        result._make_residual_layout(like=next(treecore.mlp_extractor.policy_net.parameters()))
        result.load_state_dict(treecore.state_dict(), strict=True)
        result._set_trainable_flags(train_critic=treecore.train_critic)
        return result

    def _correction(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        """Add the selected family and leaf adapters using flat lookup tables."""
        obs = self._as_batch(obs)
        ids = obs[:, -1].round().long().clamp(0, 8)
        base_phi, local_phi = self._feature_banks(obs, latent_pi)
        result = latent_pi.new_zeros((latent_pi.shape[0], 4))

        for endpoint in _ENDPOINTS:
            batch_indices = torch.nonzero(ids == endpoint, as_tuple=False).reshape(-1)
            if batch_indices.numel() == 0:
                continue
            parent = _FAMILY_FOR_ENDPOINT[endpoint]
            local_base = base_phi.index_select(0, batch_indices)
            local = local_phi.index_select(0, batch_indices)
            family = F.linear(local_base, self._parent_matrix(parent))
            leaf = F.linear(local, self._effective_matrix(endpoint))
            result = result.index_add(0, batch_indices, family + leaf)

        for parent in _GROUPS:
            batch_indices = torch.nonzero(ids == parent, as_tuple=False).reshape(-1)
            if batch_indices.numel() == 0:
                continue
            local_base = base_phi.index_select(0, batch_indices)
            family = F.linear(local_base, self._parent_matrix(parent))
            result = result.index_add(0, batch_indices, family)
        return result

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        spec["layout"] = "c1_grouped_additive_v1"
        spec["storage"] = _GROUPED_METHOD
        spec["paths"] = {}
        spec["group_membership"] = {str(group): list(members) for group, members in _GROUPS.items()}
        spec["c1_residual"]["method"] = _GROUPED_METHOD
        spec["c1_residual"]["group_membership"] = copy.deepcopy(spec["group_membership"])
        return spec

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "FlatGroupedAdditivePolicy":
        if spec.get("layout") != "c1_grouped_additive_v1":
            raise ValueError("unsupported flat grouped additive layout")
        membership = spec.get("group_membership")
        expected = {str(group): list(members) for group, members in _GROUPS.items()}
        if membership != expected:
            raise ValueError("flat grouped additive membership does not match C1")
        normalized = copy.deepcopy(dict(spec))
        normalized["storage"] = "tree"
        normalized["paths"] = {
            str(endpoint): [_PARENTS[endpoint], endpoint] for endpoint in _ENDPOINTS
        }
        normalized["c1_residual"]["method"] = "tree"
        result = super().from_spec(normalized)
        result._paths = {}
        return result


class _FamilyBasis(nn.Module):
    """One family's learned local-feature dictionary."""

    def __init__(self, rank: int, *, like: torch.Tensor) -> None:
        super().__init__()
        self.values = _mark_residual(
            nn.Parameter(torch.empty((rank, _LOCAL_PHI), device=like.device, dtype=like.dtype))
        )
        nn.init.normal_(self.values, mean=0.0, std=0.02)


class _LeafCoefficients(nn.Module):
    """A leaf's three contrast rows over its family's shared basis."""

    def __init__(self, rank: int, *, like: torch.Tensor) -> None:
        super().__init__()
        self.A = _mark_residual(nn.Parameter(torch.zeros((3, rank), device=like.device, dtype=like.dtype)))


class GroupedDictionaryPolicy(C1ResidualPolicy):
    """Two family dictionaries with per-leaf coefficients and no tree paths."""

    @staticmethod
    def _validate_options(method: str, rank: int, dictionary_size: int, epsilon: float) -> None:
        if method != _DICTIONARY_METHOD:
            raise ValueError(f"method must be {_DICTIONARY_METHOD!r}")
        if isinstance(rank, bool) or int(rank) not in (1, 2):
            raise ValueError("leaf_rank must be one or two")
        if isinstance(dictionary_size, bool) or int(dictionary_size) not in (3, 6):
            raise ValueError("dictionary_rank must be three or six")
        if not np.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) < 1.0:
            raise ValueError("epsilon must be finite and in [0, 1)")

    @staticmethod
    def _paths_for(method: str) -> dict[int, tuple[int, ...]]:
        if method != _DICTIONARY_METHOD:
            raise ValueError(f"method must be {_DICTIONARY_METHOD!r}")
        return {}

    def _make_residual_layout(self, *, like: torch.Tensor) -> None:
        self._nodes = nn.ModuleDict()
        self._parents = nn.ModuleDict()
        self._group_bases = nn.ModuleDict()
        self._node_shapes = {}
        self._dictionary_V = None
        for parent in _GROUPS:
            self._group_bases[str(parent)] = _FamilyBasis(self.dictionary_size, like=like)
        for endpoint in _ENDPOINTS:
            self._nodes[str(endpoint)] = _LeafCoefficients(self.dictionary_size, like=like)
        if "seen_mask" in self._buffers:
            del self._buffers["seen_mask"]
        # The experiment's initial seen cohort is the first two leaves in each family.
        self.register_buffer(
            "seen_mask",
            torch.tensor([True, True, False, True, True, False], dtype=torch.bool, device=like.device),
            persistent=True,
        )

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        method: str = _DICTIONARY_METHOD,
        leaf_rank: int = 1,
        dictionary_rank: int = 3,
        exploration_epsilon: float = 0.05,
        temperature: float = 1.0,
        train_critic: bool = True,
    ) -> "GroupedDictionaryPolicy":
        """Copy the frozen C1 foundation and create a rank-three or rank-six layout."""
        return cls.from_dense(
            reference,
            method=method,
            rank=leaf_rank,
            dictionary_size=dictionary_rank,
            epsilon=exploration_epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )

    @property
    def group_membership(self) -> dict[int, tuple[int, ...]]:
        return {group: tuple(members) for group, members in _GROUPS.items()}

    @property
    def shared_V(self) -> torch.Tensor:
        """Stacked view with shape ``(2, dictionary_rank, 195)``.

        The two family slices are separate parameters so LeafAdam can leave an
        inactive family's value and moments untouched.
        """
        return torch.stack([self._group_bases[str(group)].values for group in _GROUPS], dim=0)

    def _effective_matrix(self, endpoint: int) -> torch.Tensor:
        family = _FAMILY_FOR_ENDPOINT[int(endpoint)]
        reduced = self._nodes[str(endpoint)].A @ self._group_bases[str(family)].values
        return self._contrast_basis.to(device=reduced.device, dtype=reduced.dtype) @ reduced

    def _fallback_matrix(self, parent: int) -> torch.Tensor:
        members = _GROUPS[int(parent)]
        seen = [endpoint for endpoint in members if bool(self.seen_mask[endpoint - 1])]
        if not seen:
            raise RuntimeError(f"family {parent} has no seen leaves for fallback")
        # A parent route is a detached summary; its fallback must not update leaves.
        with torch.no_grad():
            return torch.stack([self._effective_matrix(endpoint) for endpoint in seen], dim=0).mean(0)

    def _fallback_correction(self, family_parent: int, local_phi: torch.Tensor) -> torch.Tensor:
        return F.linear(local_phi, self._fallback_matrix(family_parent))

    def _correction(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        obs = self._as_batch(obs)
        ids = obs[:, -1].round().long().clamp(0, 8)
        _, local_phi = self._feature_banks(obs, latent_pi)
        result = latent_pi.new_zeros((latent_pi.shape[0], 4))
        for endpoint in _ENDPOINTS:
            batch_indices = torch.nonzero(ids == endpoint, as_tuple=False).reshape(-1)
            if batch_indices.numel() == 0:
                continue
            local = local_phi.index_select(0, batch_indices)
            correction = F.linear(local, self._effective_matrix(endpoint))
            result = result.index_add(0, batch_indices, correction)
        for parent in _GROUPS:
            batch_indices = torch.nonzero(ids == parent, as_tuple=False).reshape(-1)
            if batch_indices.numel() == 0:
                continue
            local = local_phi.index_select(0, batch_indices)
            correction = self._fallback_correction(parent, local)
            result = result.index_add(0, batch_indices, correction)
        return result

    def initialize_heldout(self, endpoint: int) -> None:
        """Initialize a held-out coefficient from seen siblings and mark it seen."""
        endpoint = int(endpoint)
        if endpoint not in _ENDPOINTS:
            raise ValueError("endpoint must be one of IDs one through six")
        family = _FAMILY_FOR_ENDPOINT[endpoint]
        siblings = [
            sibling for sibling in _GROUPS[family]
            if sibling != endpoint and bool(self.seen_mask[sibling - 1])
        ]
        if not siblings:
            raise ValueError(f"endpoint {endpoint} has no seen sibling for initialization")
        with torch.no_grad():
            target = self._nodes[str(endpoint)].A
            target.copy_(torch.stack([self._nodes[str(sibling)].A for sibling in siblings]).mean(0))
            self.seen_mask[endpoint - 1] = True

    def mark_seen(self, endpoint: int) -> None:
        """Include an already adapted leaf in future family fallback averages."""
        endpoint = int(endpoint)
        if endpoint not in _ENDPOINTS:
            raise ValueError("endpoint must be one of IDs one through six")
        self.seen_mask[endpoint - 1] = True

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        groups = {str(group): list(members) for group, members in _GROUPS.items()}
        spec["layout"] = "c1_grouped_dictionary_v1"
        spec["storage"] = _DICTIONARY_METHOD
        spec["paths"] = {}
        spec["group_membership"] = groups
        config = spec["c1_residual"]
        config["method"] = _DICTIONARY_METHOD
        config["dictionary_rank"] = self.dictionary_size
        config["group_membership"] = copy.deepcopy(groups)
        config["group_ids"] = list(_GROUPS)
        config["seen_mask"] = self.seen_mask.detach().cpu().tolist()
        config["shared_basis_shape"] = [2, self.dictionary_size, _LOCAL_PHI]
        config["leaf_coefficient_shape"] = [6, 3, self.dictionary_size]
        return spec

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "GroupedDictionaryPolicy":
        if spec.get("layout") != "c1_grouped_dictionary_v1":
            raise ValueError("unsupported grouped dictionary layout")
        expected = {str(group): list(members) for group, members in _GROUPS.items()}
        if spec.get("group_membership") != expected:
            raise ValueError("grouped dictionary membership does not match C1")
        config = spec.get("c1_residual")
        if not isinstance(config, Mapping) or config.get("method") != _DICTIONARY_METHOD:
            raise ValueError("grouped dictionary spec has no matching residual configuration")
        if config.get("group_membership") != expected:
            raise ValueError("grouped dictionary residual membership does not match C1")
        result = super().from_spec(spec)
        seen = config.get("seen_mask", [True, True, False, True, True, False])
        if len(seen) != len(_ENDPOINTS):
            raise ValueError("grouped dictionary seen mask must contain six entries")
        result.seen_mask.copy_(torch.as_tensor(seen, dtype=torch.bool, device=result.seen_mask.device))
        return result

    def storage_report(self) -> dict[str, int | str]:
        report = super().storage_report()
        report["storage"] = _DICTIONARY_METHOD
        report["dictionary_rank"] = self.dictionary_size
        report["shared_basis_parameters"] = 2 * self.dictionary_size * _LOCAL_PHI
        report["leaf_coefficient_parameters"] = len(_ENDPOINTS) * 3 * self.dictionary_size
        report["fully_warmed_residual_and_adam_bytes"] = 3 * int(report["residual_value_bytes"])
        return report


__all__ = ["FlatGroupedAdditivePolicy", "GroupedDictionaryPolicy"]
