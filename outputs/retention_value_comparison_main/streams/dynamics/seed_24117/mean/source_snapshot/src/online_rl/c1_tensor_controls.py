"""Shared source-basis tensor controls for the C1 residual experiment.

All policies use one 3 by 65 basis across the three visible x bands and
context-specific 3 by 3 band coefficients.  ``tensor_learned`` adapts the
source basis along with its coefficients; ``output_mixer`` freezes the source
basis and trains only the coefficients.  These are C1 controls: the basis is
computed from the frozen W0 action head and receives only agent-visible
observations during optimization.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .c1_policy import C1ResidualPolicy, _ENDPOINTS, _LOCAL_PHI, _mark_residual
from .policy import TreePolicy


_METHODS = {"tensor_learned", "output_mixer", "hierarchical_output_mixer"}
_INITIAL_SEEN = (True, True, False, True, True, False)
_FAMILY_MEMBERS = {7: (1, 2, 3), 8: (4, 5, 6)}


class _BandCoefficients(nn.Module):
    """One leaf's separate coefficient matrices for the three x bands."""

    def __init__(self, *, like: torch.Tensor) -> None:
        super().__init__()
        self.A = _mark_residual(nn.Parameter(torch.zeros((3, 3, 3), device=like.device, dtype=like.dtype)))


class _FamilyCoefficients(nn.Module):
    """One family's shared contrast-space mixer, separate for idle-family Adam."""

    def __init__(self, *, like: torch.Tensor) -> None:
        super().__init__()
        self.C = _mark_residual(nn.Parameter(torch.zeros((3, 3), device=like.device, dtype=like.dtype)))


class TensorSourceBasisPolicy(C1ResidualPolicy):
    """C1 residual policy with a shared source-head basis and per-leaf mixers."""

    @staticmethod
    def _validate_options(method: str, rank: int, dictionary_size: int, epsilon: float) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        if isinstance(rank, bool) or int(rank) not in (1, 2):
            raise ValueError("leaf_rank must be one or two")
        if isinstance(dictionary_size, bool) or int(dictionary_size) != 3:
            raise ValueError("source-basis rank is fixed at three")
        if not np.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) < 1.0:
            raise ValueError("epsilon must be finite and in [0, 1)")

    @staticmethod
    def _paths_for(method: str) -> dict[int, tuple[int, ...]]:
        if method not in _METHODS:
            raise ValueError(f"unsupported tensor-basis method {method!r}")
        # This representation is addressed by endpoint IDs, not path objects.
        return {}

    @classmethod
    def from_dense(
        cls,
        policy: TreePolicy,
        *,
        method: str = "tensor_learned",
        rank: int = 1,
        dictionary_size: int = 3,
        epsilon: float = 0.05,
        temperature: float = 2.0,
        train_critic: bool = True,
    ) -> "TensorSourceBasisPolicy":
        return super().from_dense(
            policy,
            method=method,
            rank=rank,
            dictionary_size=dictionary_size,
            epsilon=epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        method: str = "tensor_learned",
        leaf_rank: int = 1,
        dictionary_rank: int = 3,
        exploration_epsilon: float = 0.05,
        temperature: float = 2.0,
        train_critic: bool = True,
    ) -> "TensorSourceBasisPolicy":
        """Copy W0 and build the declared source-basis control."""
        return cls.from_dense(
            reference,
            method=method,
            rank=leaf_rank,
            dictionary_size=dictionary_rank,
            epsilon=exploration_epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )

    def _make_residual_layout(self, *, like: torch.Tensor) -> None:
        self._nodes = nn.ModuleDict()
        self._parents = nn.ModuleDict()
        self._families = nn.ModuleDict()
        self._node_shapes = {}
        self._dictionary_V = None
        self.source_basis = _mark_residual(
            nn.Parameter(torch.empty((3, 65), device=like.device, dtype=like.dtype))
        )
        with torch.no_grad():
            head = torch.cat([self.action_net.weight, self.action_net.bias[:, None]], dim=1)
            basis = self._contrast_basis.to(device=head.device, dtype=head.dtype)
            self.source_basis.copy_(basis.T @ head)
        if self._storage == "hierarchical_output_mixer":
            for family in (7, 8):
                self._families[str(family)] = _FamilyCoefficients(like=like)
        for endpoint in _ENDPOINTS:
            self._nodes[str(endpoint)] = _BandCoefficients(like=like)
            self._node_shapes[endpoint] = (3, _LOCAL_PHI)
        if "seen_mask" in self._buffers:
            del self._buffers["seen_mask"]
        self.register_buffer(
            "seen_mask",
            torch.tensor(_INITIAL_SEEN, dtype=torch.bool, device=like.device),
            persistent=True,
        )

    def _set_trainable_flags(self, *, train_critic: bool) -> None:
        super()._set_trainable_flags(train_critic=train_critic)
        self.source_basis.requires_grad_(self._storage == "tensor_learned")
        self.source_basis.tree_residual = True

    @property
    def basis_trainable(self) -> bool:
        return self._storage == "tensor_learned"

    def _effective_matrix(self, endpoint: int) -> torch.Tensor:
        coefficients = self._nodes[str(int(endpoint))].A
        if self._storage == "hierarchical_output_mixer":
            family = "7" if int(endpoint) <= 3 else "8"
            coefficients = coefficients + self._families[family].C.unsqueeze(0)
        reduced = torch.cat(
            [coefficients[band] @ self.source_basis for band in range(3)], dim=1
        )
        basis = self._contrast_basis.to(device=reduced.device, dtype=reduced.dtype)
        return basis @ reduced

    def _correction(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        """Apply the source basis once, then mix only the route's active head."""
        obs = self._as_batch(obs)
        ids = obs[:, -1].round().long().clamp(0, 8)
        base_phi, _ = self._feature_banks(obs, latent_pi)
        bands = torch.floor(obs[:, 0].clamp(0.0, 1.0) * 3.0).long().clamp(0, 2)
        result = latent_pi.new_zeros((latent_pi.shape[0], 4))
        contrast_basis = self._contrast_basis.to(device=latent_pi.device, dtype=latent_pi.dtype)

        # Loop only over routes in this batch. Absent endpoint/family tensors
        # keep grad=None, allowing LeafAdam to leave their values and moments idle.
        for route in torch.unique(ids, sorted=True).tolist():
            route = int(route)
            if route not in _ENDPOINTS and route not in (7, 8):
                continue
            batch_indices = torch.nonzero(ids == route, as_tuple=False).reshape(-1)
            phi = base_phi.index_select(0, batch_indices)
            active_bands = bands.index_select(0, batch_indices)
            if route in _ENDPOINTS:
                psi = F.linear(phi, self.source_basis)
                selected_coefficients = self._nodes[str(route)].A.index_select(0, active_bands)
                if self._storage == "hierarchical_output_mixer":
                    family = "7" if route <= 3 else "8"
                    selected_coefficients = selected_coefficients + self._families[family].C.unsqueeze(0)
                contrast = torch.bmm(selected_coefficients, psi.unsqueeze(-1)).squeeze(-1)
            elif route in (7, 8):
                if self._storage == "hierarchical_output_mixer":
                    # Match C1 tree parent adapters: uncertain family routes
                    # remain trainable through their shared family coefficient.
                    family = str(route)
                    psi = F.linear(phi, self.source_basis)
                    contrast = F.linear(psi, self._families[family].C)
                else:
                    members = _FAMILY_MEMBERS[route]
                    selected = [member for member in members if bool(self.seen_mask[member - 1])]
                    if not selected:
                        raise RuntimeError(f"family {route} has no seen endpoint for fallback")
                    # Match flat dictionary fallback: neither coefficients nor
                    # the shared feature basis receive gradients on this route.
                    with torch.no_grad():
                        coefficients = torch.stack(
                            [self._nodes[str(member)].A for member in selected]
                        ).mean(0)
                        detached_psi = F.linear(phi, self.source_basis)
                        contrast = torch.bmm(
                            coefficients.index_select(0, active_bands),
                            detached_psi.unsqueeze(-1),
                        ).squeeze(-1)
            else:
                continue
            result = result.index_add(0, batch_indices, contrast @ contrast_basis.T)
        return result

    def _fallback_correction(self, family_parent: int, local_phi: torch.Tensor) -> torch.Tensor:
        if self._storage == "hierarchical_output_mixer":
            reduced = self._families[str(int(family_parent))].C @ self.source_basis
            basis = self._contrast_basis.to(device=reduced.device, dtype=reduced.dtype)
            family_matrix = basis @ reduced
            repeated = torch.cat([family_matrix, family_matrix, family_matrix], dim=1)
            return F.linear(local_phi, repeated)
        members = _FAMILY_MEMBERS[int(family_parent)]
        selected = [member for member in members if bool(self.seen_mask[member - 1])]
        if not selected:
            raise RuntimeError(f"family {family_parent} has no seen endpoint for fallback")
        with torch.no_grad():
            matrix = torch.stack([self._effective_matrix(endpoint) for endpoint in selected]).mean(0)
        return F.linear(local_phi, matrix)

    def initialize_heldout(self, endpoint: int) -> None:
        """Initialize a held-out leaf from the declared family/leaf prior."""
        endpoint = int(endpoint)
        if endpoint not in _ENDPOINTS:
            raise ValueError("endpoint must be one through six")
        if self._storage == "hierarchical_output_mixer":
            # The family coefficient is the retained parent prior. The new
            # leaf starts with no additional local correction.
            with torch.no_grad():
                self._nodes[str(endpoint)].A.zero_()
                self.seen_mask[endpoint - 1] = True
            return
        family = 7 if endpoint <= 3 else 8
        siblings = [
            member for member in _FAMILY_MEMBERS[family]
            if member != endpoint and bool(self.seen_mask[member - 1])
        ]
        if not siblings:
            raise ValueError(f"endpoint {endpoint} has no seen sibling")
        with torch.no_grad():
            target = self._nodes[str(endpoint)].A
            target.copy_(torch.stack([self._nodes[str(member)].A for member in siblings]).mean(0))
            self.seen_mask[endpoint - 1] = True

    def mark_seen(self, endpoint: int) -> None:
        endpoint = int(endpoint)
        if endpoint not in _ENDPOINTS:
            raise ValueError("endpoint must be one through six")
        with torch.no_grad():
            self.seen_mask[endpoint - 1] = True

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        spec["supplemental_layout"] = "c1_tensor_source_basis_v1"
        spec["supplemental_seen_mask"] = self.seen_mask.detach().cpu().tolist()
        config = spec["c1_residual"]
        config["method"] = self._storage
        config["dictionary_rank"] = 3
        config["source_basis_shape"] = [3, 65]
        config["source_basis_initialization"] = "centered_W0_action_head"
        config["source_basis_trainable"] = self.basis_trainable
        config["band_coefficient_shape"] = [6, 3, 3, 3]
        config["band_mode"] = "one_hot_visible_x_band"
        config["group_membership"] = {"7": [1, 2, 3], "8": [4, 5, 6]}
        if self._storage == "hierarchical_output_mixer":
            config["family_coefficient_shape"] = [2, 3, 3]
            config["family_ids"] = [7, 8]
            config["parent_fallback"] = "trainable_family_coefficients_only"
        return spec

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "TensorSourceBasisPolicy":
        if spec.get("supplemental_layout") != "c1_tensor_source_basis_v1":
            raise ValueError("unsupported C1 tensor source-basis layout")
        config = spec.get("c1_residual")
        if not isinstance(config, Mapping) or config.get("method") not in _METHODS:
            raise ValueError("tensor source-basis spec has no matching method")
        if config.get("source_basis_shape") != [3, 65]:
            raise ValueError("source-basis shape must be 3 by 65")
        if config.get("band_coefficient_shape") != [6, 3, 3, 3]:
            raise ValueError("band coefficient shape must be six by three by three by three")
        if config.get("group_membership") != {"7": [1, 2, 3], "8": [4, 5, 6]}:
            raise ValueError("tensor source-basis group membership does not match C1")
        result = C1ResidualPolicy.from_spec.__func__(cls, spec)
        seen = spec.get("supplemental_seen_mask", _INITIAL_SEEN)
        if len(seen) != 6:
            raise ValueError("supplemental seen mask must contain six entries")
        result.seen_mask.copy_(torch.as_tensor(seen, dtype=torch.bool, device=result.seen_mask.device))
        expected_trainable = result._storage == "tensor_learned"
        if bool(config.get("source_basis_trainable")) != expected_trainable:
            raise ValueError("source-basis trainability does not match the declared method")
        if result._storage == "hierarchical_output_mixer":
            if config.get("family_coefficient_shape") != [2, 3, 3]:
                raise ValueError("family coefficient shape must be two by three by three")
            if config.get("family_ids") != [7, 8]:
                raise ValueError("hierarchical tensor family IDs do not match C1")
        elif "family_coefficient_shape" in config or "family_ids" in config:
            raise ValueError("flat source-basis specs cannot declare family coefficients")
        return result

    def storage_report(self) -> dict[str, int | str]:
        report = super().storage_report()
        mask_bytes = int(self.seen_mask.numel() * self.seen_mask.element_size())
        basis_bytes = int(self.source_basis.numel() * self.source_basis.element_size())
        coefficient_bytes = sum(
            int(node.A.numel() * node.A.element_size()) for node in self._nodes.values()
        )
        family_bytes = sum(
            int(node.C.numel() * node.C.element_size()) for node in self._families.values()
        )
        trainable_residuals = [parameter for parameter in self._residual_parameters() if parameter.requires_grad]
        warmed_moment_bytes = sum(
            2 * int(parameter.numel() * parameter.element_size())
            for parameter in trainable_residuals
        )
        # LeafAdam/PyTorch Adam stores a scalar ``step`` tensor per parameter.
        warmed_step_bytes = sum(
            torch.zeros((), dtype=parameter.dtype).element_size()
            for parameter in trainable_residuals
        )
        warmed_residual_and_adam = int(
            report["residual_value_bytes"] + warmed_moment_bytes + warmed_step_bytes
        )
        report.update({
            "storage": self._storage,
            "source_basis_parameters": int(self.source_basis.numel()),
            "source_basis_bytes": basis_bytes,
            "source_basis_trainable": int(self.basis_trainable),
            "band_coefficient_parameters": 6 * 3 * 3 * 3,
            "band_coefficient_bytes": coefficient_bytes,
            "family_coefficient_parameters": 2 * 3 * 3 if self._storage == "hierarchical_output_mixer" else 0,
            "family_coefficient_bytes": family_bytes,
            "fully_warmed_adam_moment_bytes": warmed_moment_bytes,
            "fully_warmed_adam_step_bytes": warmed_step_bytes,
            "fully_warmed_trainable_residual_parameter_count": len(trainable_residuals),
            "coordinate_index_bytes": mask_bytes,
            "index_bytes": mask_bytes,
            "fully_warmed_residual_and_adam_bytes": warmed_residual_and_adam,
            "fully_warmed_adaptive_tensor_bytes": warmed_residual_and_adam + mask_bytes,
            "seen_endpoint_count": int(self.seen_mask.sum().item()),
        })
        return report


__all__ = ["TensorSourceBasisPolicy"]
