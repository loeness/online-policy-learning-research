from __future__ import annotations

import numpy as np
import torch

from scripts.benchmark_score_features import paired_coefficient_probs


torch.set_num_threads(1)


class _IdentityActor:
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features


class _TinyCore:
    def __init__(self) -> None:
        self.mlp_extractor = _IdentityActor()
        self.source_basis = torch.tensor(
            [
                [0.10, -0.03, 0.20],
                [-0.15, 0.08, 0.04],
                [0.02, 0.12, -0.05],
            ],
            dtype=torch.float32,
        )
        self._contrast_basis = torch.tensor(
            [[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1], [0.3, 0.2, -0.2], [-0.1, -0.3, 0.2]],
            dtype=torch.float32,
        )
        self.action_net = torch.nn.Linear(2, 4, bias=True)
        with torch.no_grad():
            self.action_net.weight.copy_(
                torch.tensor([[0.1, 0.2], [-0.3, 0.1], [0.2, -0.2], [-0.1, 0.3]])
            )
            self.action_net.bias.copy_(torch.tensor([0.01, -0.02, 0.03, -0.01]))
        self.temperature = 0.8
        self.epsilon = 0.08

    @staticmethod
    def extract_features(observations: torch.Tensor) -> torch.Tensor:
        return observations[:, 1:3]


def _manual_probs(core: _TinyCore, observations: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    latent = obs[:, 1:3]
    phi = np.concatenate((latent, np.ones((len(obs), 1), dtype=np.float32)), axis=1)
    psi = phi @ core.source_basis.numpy().T
    base_logits = latent @ core.action_net.weight.detach().numpy().T
    base_logits += core.action_net.bias.detach().numpy()
    coeff = np.asarray(coefficients, dtype=np.float32)
    if coeff.shape == (3, 3):
        contrast = psi @ coeff.T
    else:
        bands = np.floor(np.clip(obs[:, 0], 0.0, 1.0) * 3).astype(np.int64)
        bands = np.clip(bands, 0, 2)
        contrast = np.einsum("nij,nj->ni", coeff[bands], psi)
    logits = base_logits + contrast @ core._contrast_basis.numpy().T
    logits = logits / core.temperature
    logits -= logits.max(axis=1, keepdims=True)
    softmax = np.exp(logits)
    softmax /= softmax.sum(axis=1, keepdims=True)
    return ((1.0 - core.epsilon) * softmax + core.epsilon / 4.0).astype(np.float32)


def test_paired_features_match_manual_probabilities_for_numeric_inputs() -> None:
    core = _TinyCore()
    observations = np.zeros((4, 13), dtype=np.float32)
    observations[:, 0] = [0.1, 0.4, 0.7, 0.95]
    observations[:, 1:3] = [[0.2, -0.1], [-0.4, 0.3], [0.6, 0.2], [-0.2, -0.5]]
    candidate = np.asarray([[0.1, -0.2, 0.0], [0.05, 0.0, 0.1], [-0.1, 0.2, 0.03]], dtype=np.float32)
    baseline = np.asarray([[-0.03, 0.01, 0.04], [0.02, 0.08, -0.05], [0.1, -0.07, 0.0]], dtype=np.float32)

    candidate_probs, baseline_probs = paired_coefficient_probs(
        core, observations, candidate, baseline
    )

    assert np.allclose(candidate_probs, _manual_probs(core, observations, candidate), atol=1e-7, rtol=0)
    assert np.allclose(baseline_probs, _manual_probs(core, observations, baseline), atol=1e-7, rtol=0)
    assert np.allclose(candidate_probs.sum(axis=1, dtype=np.float64), 1.0, atol=1e-7, rtol=0)
    assert not np.allclose(candidate_probs, baseline_probs)


def test_paired_features_support_banded_and_mixed_coefficients() -> None:
    core = _TinyCore()
    observations = np.zeros((4, 13), dtype=np.float32)
    observations[:, 0] = [0.0, 0.34, 0.68, 1.0]
    observations[:, 1:3] = [[0.1, 0.3], [-0.2, 0.4], [0.5, -0.1], [-0.3, -0.2]]
    candidate = np.stack(
        [
            np.eye(3, dtype=np.float32) * 0.05,
            np.asarray([[0.0, 0.1, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.08]], dtype=np.float32),
            np.asarray([[0.03, 0.0, -0.04], [0.0, 0.06, 0.0], [0.05, 0.0, 0.02]], dtype=np.float32),
        ],
        axis=0,
    )
    baseline = np.asarray([[0.02, -0.01, 0.03], [-0.04, 0.01, 0.02], [0.0, 0.05, -0.03]], dtype=np.float32)

    candidate_probs, baseline_probs = paired_coefficient_probs(
        core, observations, candidate, baseline
    )

    assert np.allclose(candidate_probs, _manual_probs(core, observations, candidate), atol=1e-7, rtol=0)
    assert np.allclose(baseline_probs, _manual_probs(core, observations, baseline), atol=1e-7, rtol=0)
    assert np.max(np.abs(candidate_probs[0] - candidate_probs[1])) > 1e-6


def test_paired_features_return_valid_empty_batches() -> None:
    core = _TinyCore()
    candidate_probs, baseline_probs = paired_coefficient_probs(
        core,
        np.empty((0, 13), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.ones((3, 3, 3), dtype=np.float32),
    )

    assert candidate_probs.shape == (0, 4)
    assert baseline_probs.shape == (0, 4)
    assert candidate_probs.dtype == np.float32
    assert baseline_probs.dtype == np.float32
