"""Active shared correction policy for the C1 source-basis controller.

Each route keeps its own three band-specific local matrices. A route may also
select one shared 3 by 3 correction, broadcast across all three bands. Shared
matrices are separate parameters so inactive groups keep ``grad=None`` and
remain idle in ``LeafAdam``.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .c1_tensor_controls import TensorSourceBasisPolicy
from .c1_policy import _ENDPOINTS, _mark_residual
from .policy import TreePolicy


class _SharedCoefficients(nn.Module):
    """One independently trainable shared 3 by 3 mixer."""

    def __init__(self, *, like: torch.Tensor) -> None:
        super().__init__()
        self.C = _mark_residual(
            nn.Parameter(torch.zeros((3, 3), device=like.device, dtype=like.dtype))
        )


class SharedWritePolicy(TensorSourceBasisPolicy):
    """Frozen W0 source basis with local and live shared route corrections."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__(temperature=temperature)
        self.shared_groups = 2
        self._shared_frozen = False
        self._shared_coefficients = nn.ModuleList()
        self.register_buffer("slot_groups", torch.full((6,), -1, dtype=torch.long), persistent=True)

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        shared_groups: int = 2,
        exploration_epsilon: float = 0.05,
        temperature: float = 2.0,
        train_critic: bool = True,
    ) -> "SharedWritePolicy":
        """Copy the reference once and create an output-mixer shared policy."""
        group_count = cls._validate_group_count(shared_groups)
        result = super().from_reference(
            reference,
            method="output_mixer",
            leaf_rank=1,
            dictionary_rank=3,
            exploration_epsilon=exploration_epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )
        result._configure_shared_groups(group_count)
        result._set_trainable_flags(train_critic=train_critic)
        return result

    @staticmethod
    def _validate_group_count(shared_groups: int) -> int:
        if isinstance(shared_groups, bool):
            raise ValueError("shared_groups must be a nonnegative integer")
        try:
            value = int(shared_groups)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("shared_groups must be a nonnegative integer") from exc
        if value != shared_groups or value < 0:
            raise ValueError("shared_groups must be a nonnegative integer")
        return value

    @staticmethod
    def _validate_slot(slot: int) -> int:
        if isinstance(slot, bool):
            raise ValueError("slot must be an integer from zero through five")
        try:
            value = int(slot)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("slot must be an integer from zero through five") from exc
        if value != slot or value < 0 or value >= len(_ENDPOINTS):
            raise ValueError("slot must be an integer from zero through five")
        return value

    def _make_residual_layout(self, *, like: torch.Tensor) -> None:
        super()._make_residual_layout(like=like)
        self._shared_coefficients = nn.ModuleList(
            _SharedCoefficients(like=like) for _ in range(self.shared_groups)
        )

    def _configure_shared_groups(self, count: int) -> None:
        """Resize initially constructed groups to the factory's requested count."""
        if count == self.shared_groups:
            return
        like = self.source_basis
        self._shared_coefficients = nn.ModuleList(
            _SharedCoefficients(like=like) for _ in range(count)
        )
        self.shared_groups = count
        with torch.no_grad():
            self.slot_groups.fill_(-1)

    def _set_trainable_flags(self, *, train_critic: bool) -> None:
        super()._set_trainable_flags(train_critic=train_critic)
        for parameter in self.shared_parameters():
            parameter.requires_grad_(not self._shared_frozen)

    def _correction(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        """Apply local and selected shared matrices only on routes one through six."""
        obs = self._as_batch(obs)
        ids = obs[:, -1].round().long().clamp(0, 8)
        base_phi, _ = self._feature_banks(obs, latent_pi)
        bands = torch.floor(obs[:, 0].clamp(0.0, 1.0) * 3.0).long().clamp(0, 2)
        result = latent_pi.new_zeros((latent_pi.shape[0], 4))
        contrast_basis = self._contrast_basis.to(device=latent_pi.device, dtype=latent_pi.dtype)

        for route in torch.unique(ids, sorted=True).tolist():
            route = int(route)
            if route not in _ENDPOINTS:
                continue
            rows = torch.nonzero(ids == route, as_tuple=False).reshape(-1)
            phi = base_phi.index_select(0, rows)
            active_bands = bands.index_select(0, rows)
            psi = F.linear(phi, self.source_basis)
            coefficients = self._nodes[str(route)].A.index_select(0, active_bands)
            group = int(self.slot_groups[route - 1].item())
            if group >= 0:
                coefficients = coefficients + self._shared_coefficients[group].C.unsqueeze(0)
            contrast = torch.bmm(coefficients, psi.unsqueeze(-1)).squeeze(-1)
            result = result.index_add(0, rows, contrast @ contrast_basis.T)
        return result

    def group_for_slot(self, slot: int) -> int:
        """Return the shared group assigned to zero-based slot ``slot``, or -1."""
        slot = self._validate_slot(slot)
        return int(self.slot_groups[slot].item())

    def set_slot_group(self, slot: int, group: int, *, preserve_effective: bool = True) -> None:
        """Assign a slot, optionally compensating its local matrices exactly."""
        slot = self._validate_slot(slot)
        if isinstance(group, bool):
            raise ValueError("group must be -1 or an available shared group index")
        requested_group = group
        try:
            group = int(group)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("group must be -1 or an available shared group index") from exc
        if group != requested_group or group < -1 or group >= self.shared_groups:
            raise ValueError("group must be -1 or an available shared group index")

        old_group = self.group_for_slot(slot)
        if preserve_effective and old_group != group:
            old_shared = (
                self._shared_coefficients[old_group].C
                if old_group >= 0
                else self._nodes[str(slot + 1)].A.new_zeros((3, 3))
            )
            new_shared = (
                self._shared_coefficients[group].C
                if group >= 0
                else self._nodes[str(slot + 1)].A.new_zeros((3, 3))
            )
            with torch.no_grad():
                self._nodes[str(slot + 1)].A.add_(old_shared - new_shared)
        with torch.no_grad():
            self.slot_groups[slot] = group

    def read_local_matrix(self) -> np.ndarray:
        """Return local coefficients as a copied six by twenty-seven array."""
        return np.stack(
            [self._nodes[str(slot + 1)].A.detach().cpu().numpy().reshape(27) for slot in range(6)]
        ).copy()

    def install_local_matrix(self, matrix: np.ndarray) -> None:
        """Install a six by twenty-seven local coefficient array."""
        values = np.asarray(matrix)
        if values.shape != (6, 27):
            raise ValueError("local matrix must have shape (6, 27)")
        with torch.no_grad():
            for slot in range(6):
                target = self._nodes[str(slot + 1)].A
                target.copy_(torch.as_tensor(values[slot], device=target.device, dtype=target.dtype).reshape(3, 3, 3))

    def read_shared_matrix(self) -> np.ndarray:
        """Return shared coefficients as a copied group by nine array."""
        if not self.shared_groups:
            return np.empty((0, 9), dtype=self.source_basis.detach().cpu().numpy().dtype)
        return np.stack(
            [group.C.detach().cpu().numpy().reshape(9) for group in self._shared_coefficients]
        ).copy()

    def install_shared_matrix(self, matrix: np.ndarray) -> None:
        """Install a group by nine shared coefficient array."""
        values = np.asarray(matrix)
        if values.shape != (self.shared_groups, 9):
            raise ValueError(f"shared matrix must have shape ({self.shared_groups}, 9)")
        with torch.no_grad():
            for group, target_module in enumerate(self._shared_coefficients):
                target = target_module.C
                target.copy_(torch.as_tensor(values[group], device=target.device, dtype=target.dtype).reshape(3, 3))

    def reset_slot(self, slot: int) -> None:
        """Clear a slot's local correction and remove its shared assignment."""
        slot = self._validate_slot(slot)
        with torch.no_grad():
            self._nodes[str(slot + 1)].A.zero_()
            self.slot_groups[slot] = -1

    def freeze_shared(self, freeze: bool = True) -> None:
        """Freeze or unfreeze shared group parameters."""
        self._shared_frozen = bool(freeze)
        for parameter in self.shared_parameters():
            parameter.requires_grad_(not self._shared_frozen)

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        """Iterate over independently trainable shared group matrices."""
        return tuple(group.C for group in self._shared_coefficients)


__all__ = ["SharedWritePolicy"]
