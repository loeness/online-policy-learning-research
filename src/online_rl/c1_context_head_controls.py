"""Full shared-policy control with compact task and band residual heads."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .c1_continual_controls import C1FullFineTunePolicy
from .c1_policy import _contrast_basis
from .policy import TreePolicy


_ARCHITECTURE = "C1ContextHeadFullPolicy-v1"
_PARENT_ARCHITECTURE = "C1FullFineTunePolicy-v1"
_ENDPOINTS = tuple(range(1, 7))
_FAMILIES = {7: (1, 2, 3), 8: (4, 5, 6)}
_LOCAL_FEATURES = 3 * 65


class C1ContextHeadFullPolicy(C1FullFineTunePolicy):
    """One trainable actor/critic plus six small route-specific residual heads.

    The shared actor receives the inherited zero-initialized route and band
    biases. Each endpoint also has a zero-initialized ``3 x 195`` output head,
    projected through the fixed four-action contrast basis. Its input is the
    shared 64-value actor latent plus a constant, gated by the observed x band.
    Thus initialization exactly reproduces W0, while later updates can adapt a
    shared actor and concise endpoint-specific functions without storing six
    actor networks.

    Parent routes use the detached mean of every known endpoint head. That
    diagnostic fallback does not backpropagate into endpoint heads; the shared
    actor, critic, route bias, and band bias remain trainable through a parent
    route. A newly exposed endpoint copies the mean of its known siblings once
    and is then included in future parent means.
    """

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        route_conditioned: bool = True,
        band_conditioned: bool = True,
        epsilon: float = 0.05,
        temperature: float = 2.0,
    ) -> "C1ContextHeadFullPolicy":
        if not route_conditioned or not band_conditioned:
            raise ValueError("context-head policy requires route and band conditioning")
        result = super().from_reference(
            reference,
            route_conditioned=True,
            band_conditioned=True,
            epsilon=epsilon,
            temperature=temperature,
        )
        like = next(result.mlp_extractor.policy_net.parameters())
        result.context_heads = nn.ParameterList([
            nn.Parameter(torch.zeros((3, _LOCAL_FEATURES), dtype=like.dtype, device=like.device))
            for _ in _ENDPOINTS
        ])
        result.register_buffer(
            "_context_contrast_basis",
            _contrast_basis(dtype=like.dtype, device=like.device),
            persistent=False,
        )
        return result

    @property
    def parameter_update_scope(self) -> str:
        return "full_shared_actor_critic_route_band_and_six_context_heads"

    def actor_named_parameters(self) -> dict[str, nn.Parameter]:
        result = super().actor_named_parameters()
        result.update({f"context_heads.{index}": parameter
                       for index, parameter in enumerate(self.context_heads)})
        return result

    def _context_head_logits(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        band = torch.floor(obs[:, 0].clamp(0.0, 1.0) * 3.0).long().clamp(0, 2)
        base_phi = torch.cat((latent_pi, torch.ones_like(latent_pi[:, :1])), dim=1)
        band_gate = F.one_hot(band, num_classes=3).to(dtype=latent_pi.dtype)
        local_phi = (band_gate.unsqueeze(-1) * base_phi.unsqueeze(1)).reshape(batch, _LOCAL_FEATURES)
        routes = obs[:, -1].round().long().clamp(0, 8)
        result = latent_pi.new_zeros((batch, 4))

        for endpoint in _ENDPOINTS:
            indices = torch.nonzero(routes == endpoint, as_tuple=False).reshape(-1)
            if indices.numel() == 0:
                continue
            features = local_phi.index_select(0, indices)
            contrast = F.linear(features, self.context_heads[endpoint - 1])
            correction = F.linear(contrast, self._context_contrast_basis)
            result = result.index_add(0, indices, correction)

        for parent, members in _FAMILIES.items():
            indices = torch.nonzero(routes == parent, as_tuple=False).reshape(-1)
            if indices.numel() == 0:
                continue
            known = [member for member in members if bool(self.known_endpoint_mask[member - 1])]
            if not known:
                continue
            with torch.no_grad():
                average_head = torch.stack([self.context_heads[member - 1].detach() for member in known]).mean(0)
            contrast = F.linear(local_phi.index_select(0, indices), average_head)
            correction = F.linear(contrast, self._context_contrast_basis)
            result = result.index_add(0, indices, correction)
        return result

    def _actor_logits_from_features(self, obs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        first_layer = self.mlp_extractor.policy_net[0]
        hidden_pre_activation = F.linear(features[:, :4], first_layer.weight, first_layer.bias)
        hidden_pre_activation = hidden_pre_activation + self._route_shift(obs)
        band = (obs[:, 0] * 8).round().long().clamp(0, 8) // 3
        hidden_pre_activation = hidden_pre_activation + self.band_embedding[band]
        latent_pi = self.mlp_extractor.policy_net[1:](hidden_pre_activation)
        shared_logits = self.action_net(latent_pi)
        return shared_logits + self._context_head_logits(obs, latent_pi)

    def initialize_heldout(self, endpoint: int) -> None:
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, np.integer)) or int(endpoint) not in _ENDPOINTS:
            raise ValueError("endpoint must be an integer ID one through six")
        endpoint = int(endpoint)
        if bool(self.known_endpoint_mask[endpoint - 1]):
            raise ValueError(f"endpoint {endpoint} is already known")
        members = _FAMILIES[7] if endpoint <= 3 else _FAMILIES[8]
        siblings = [member for member in members
                    if member != endpoint and bool(self.known_endpoint_mask[member - 1])]
        if not siblings:
            raise ValueError(f"endpoint {endpoint} has no known sibling for initialization")
        with torch.no_grad():
            self.context_heads[endpoint - 1].copy_(torch.stack(
                [self.context_heads[member - 1] for member in siblings]
            ).mean(0))
        super().initialize_heldout(endpoint)

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        spec["architecture"] = _ARCHITECTURE
        spec["layout"] = "c1_shared_actor_context_heads_v1"
        spec["context_head_count"] = len(_ENDPOINTS)
        spec["context_head_shape"] = [3, _LOCAL_FEATURES]
        spec["context_head_bytes"] = sum(
            parameter.numel() * parameter.element_size() for parameter in self.context_heads
        )
        spec["parent_head_fallback"] = "detached mean over all known family endpoints"
        spec["known_endpoint_mask"] = self.known_endpoint_mask.detach().cpu().tolist()
        return spec

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "C1ContextHeadFullPolicy":
        if int(spec.get("format_version", -1)) != 1 or spec.get("architecture") != _ARCHITECTURE:
            raise ValueError("unsupported C1 context-head policy spec")
        if spec.get("layout") != "c1_shared_actor_context_heads_v1":
            raise ValueError("unsupported C1 context-head layout")
        if int(spec.get("context_head_count", -1)) != 6 or spec.get("context_head_shape") != [3, _LOCAL_FEATURES]:
            raise ValueError("C1 context-head spec must contain six 3 x 195 heads")
        normalized = copy.deepcopy(dict(spec))
        normalized["architecture"] = _PARENT_ARCHITECTURE
        normalized["route_conditioned"] = True
        normalized["band_conditioned"] = True
        result = super().from_spec(normalized)
        known = spec.get("known_endpoint_mask", [True, True, False, True, True, False])
        if len(known) != 6:
            raise ValueError("known endpoint mask must contain six entries")
        result.known_endpoint_mask.copy_(torch.as_tensor(
            known, dtype=torch.bool, device=result.known_endpoint_mask.device
        ))
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError("unsupported C1 context-head payload version")
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

    def storage_report(self, optimizer: torch.optim.Optimizer | None = None) -> dict[str, int | str | bool]:
        report = super().storage_report(optimizer)
        head_params = sum(int(parameter.numel() * parameter.element_size()) for parameter in self.context_heads)
        report["context_head_count"] = len(self.context_heads)
        report["context_head_shape"] = [3, _LOCAL_FEATURES]
        report["context_head_parameter_bytes"] = head_params
        report["known_endpoint_mask_bytes"] = int(
            self.known_endpoint_mask.numel() * self.known_endpoint_mask.element_size()
        )
        return report


__all__ = ["C1ContextHeadFullPolicy"]
