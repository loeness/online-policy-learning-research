"""Bounded selected-feedback value estimates for storage actions.

The model learns a separate contextual linear value function for each action.
It receives only the outcome for the action that was actually selected; it is
not an environment runner or an off-policy evaluation method.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class Choice(NamedTuple):
    """A sampled action and its marginal behavior propensity."""

    selected: str
    predicted_values: dict[str, float]
    propensity: float
    explored: bool

    @property
    def action(self) -> str:
        """Alias for ``selected`` for call sites that use action terminology."""

        return self.selected


class RetentionValueModel:
    """Discounted ridge value estimates for ``keep``, ``replace``, ``mean``, ``none``.

    The state is fixed-size: one Gram matrix and target vector per action plus
    per-action observation counts. ``default_action`` breaks exact prediction
    ties, including the initial all-zero tie. Exploration uses only the RNG
    passed to :meth:`choose`.
    """

    ACTIONS = ("keep", "replace", "mean", "none")
    FORMAT_VERSION = 1
    __slots__ = (
        "n_features",
        "ridge",
        "discount",
        "lambda_cpu",
        "default_action",
        "A",
        "b",
        "counts",
        "discounted_counts",
    )

    def __init__(
        self,
        n_features: int,
        *,
        ridge: float = 1.0,
        discount: float = 0.97,
        lambda_cpu: float = 0.0,
        default_action: str = "keep",
    ) -> None:
        if isinstance(n_features, (bool, np.bool_)) or not isinstance(
            n_features, (int, np.integer)
        ):
            raise ValueError("n_features must be a positive integer")
        if int(n_features) <= 0:
            raise ValueError("n_features must be a positive integer")
        ridge_value = _finite_float(ridge, "ridge")
        if ridge_value <= 0:
            raise ValueError("ridge must be greater than zero")
        discount_value = _finite_float(discount, "discount")
        if not 0 < discount_value <= 1:
            raise ValueError("discount must be in (0, 1]")
        lambda_value = _finite_float(lambda_cpu, "lambda_cpu")
        if lambda_value < 0:
            raise ValueError("lambda_cpu cannot be negative")
        if default_action not in self.ACTIONS:
            raise ValueError(f"default_action must be one of {self.ACTIONS}")

        self.n_features = int(n_features)
        self.ridge = ridge_value
        self.discount = discount_value
        self.lambda_cpu = lambda_value
        self.default_action = default_action

        action_count = len(self.ACTIONS)
        self.A = np.broadcast_to(
            np.eye(self.n_features, dtype=np.float64) * self.ridge,
            (action_count, self.n_features, self.n_features),
        ).copy()
        self.b = np.zeros((action_count, self.n_features), dtype=np.float64)
        self.counts = np.zeros(action_count, dtype=np.int64)
        self.discounted_counts = np.zeros(action_count, dtype=np.float64)

    def predict(self, features: Any) -> dict[str, float]:
        """Return the estimated mean utility for every storage action."""

        feature_vector = self._features(features)
        return self._prediction_details(feature_vector)[0]

    def predict_with_uncertainty(
        self, features: Any
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return means and ``x.T @ A^-1 @ x`` diagnostics per action.

        The second mapping is a ridge leverage diagnostic. It is not an upper
        confidence bound or a calibrated uncertainty guarantee.
        """

        feature_vector = self._features(features)
        return self._prediction_details(feature_vector)

    def choose(
        self,
        features: Any,
        epsilon: float = 0.0,
        rng: Any | None = None,
    ) -> Choice:
        """Choose greedily or from an epsilon/uniform mixture.

        ``rng`` must be an externally supplied NumPy Generator when epsilon is
        positive, so exploration can be kept independent from environment
        randomness. ``propensity`` is the marginal probability of the chosen
        action under the greedy/uniform mixture. ``explored`` marks selection
        from the uniform branch, even when that branch picks the greedy action.
        """

        epsilon_value = _finite_float(epsilon, "epsilon")
        if not 0 <= epsilon_value <= 1:
            raise ValueError("epsilon must be in [0, 1]")

        predicted = self.predict(features)
        greedy = self._greedy_action(predicted)
        explored = False
        selected = greedy

        if epsilon_value > 0:
            if rng is None or not callable(getattr(rng, "random", None)) or not callable(
                getattr(rng, "integers", None)
            ):
                raise ValueError("positive epsilon requires an external NumPy Generator")
            if float(rng.random()) < epsilon_value:
                selected = self.ACTIONS[int(rng.integers(len(self.ACTIONS)))]
                explored = True

        propensity = epsilon_value / len(self.ACTIONS)
        if selected == greedy:
            propensity += 1.0 - epsilon_value
        return Choice(selected, predicted, float(propensity), explored)

    def update(
        self,
        action: str,
        features: Any,
        reward_rate: float,
        cpu_s: float,
        environment_steps: int,
    ) -> float:
        """Update only the selected action using its measured utility.

        The target is exactly ``reward_rate - lambda_cpu * (cpu_s /
        environment_steps)``. Every action's accumulated evidence is discounted
        once per update, while the ridge prior is retained in each Gram matrix.
        Returns the utility target that was added to the selected action.
        """

        action_index = self._action_index(action)
        feature_vector = self._features(features)
        reward_value = _finite_float(reward_rate, "reward_rate")
        cpu_value = _finite_float(cpu_s, "cpu_s")
        if cpu_value < 0:
            raise ValueError("cpu_s cannot be negative")
        if isinstance(environment_steps, (bool, np.bool_)) or not isinstance(
            environment_steps, (int, np.integer)
        ):
            raise ValueError("environment_steps must be a positive integer")
        if int(environment_steps) <= 0:
            raise ValueError("environment_steps must be a positive integer")

        utility = reward_value - self.lambda_cpu * (cpu_value / int(environment_steps))
        if not np.isfinite(utility):
            raise ValueError("measured utility must be finite")
        gram_increment = np.outer(feature_vector, feature_vector)
        target_increment = utility * feature_vector
        if not np.isfinite(gram_increment).all() or not np.isfinite(target_increment).all():
            raise ValueError("update would produce non-finite sufficient statistics")
        if self.counts[action_index] == np.iinfo(np.int64).max:
            raise OverflowError("per-action count exceeds int64 capacity")

        # Discount all actions together so each has the same evidence age.
        self.A *= self.discount
        self.A += (1.0 - self.discount) * self.ridge * np.eye(
            self.n_features, dtype=np.float64
        )[None, :, :]
        self.b *= self.discount
        self.discounted_counts *= self.discount

        self.A[action_index] += gram_increment
        self.b[action_index] += target_increment
        self.counts[action_index] += 1
        self.discounted_counts[action_index] += 1.0
        return float(utility)

    def state_dict(self) -> dict[str, Any]:
        """Return an independent, exact snapshot of the bounded model state."""

        return {
            "format_version": self.FORMAT_VERSION,
            "actions": self.ACTIONS,
            "n_features": self.n_features,
            "ridge": self.ridge,
            "discount": self.discount,
            "lambda_cpu": self.lambda_cpu,
            "default_action": self.default_action,
            "A": self.A.copy(),
            "b": self.b.copy(),
            "counts": self.counts.copy(),
            "discounted_counts": self.discounted_counts.copy(),
        }

    @classmethod
    def from_state_dict(cls, state: Any) -> RetentionValueModel:
        """Restore a model from :meth:`state_dict`, validating shapes and values."""

        if not isinstance(state, dict):
            raise ValueError("state must be a state_dict mapping")
        if state.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported retention value state format")
        if tuple(state.get("actions", ())) != cls.ACTIONS:
            raise ValueError("state actions do not match this model")

        model = cls(
            state.get("n_features"),
            ridge=state.get("ridge"),
            discount=state.get("discount"),
            lambda_cpu=state.get("lambda_cpu"),
            default_action=state.get("default_action"),
        )
        action_count = len(cls.ACTIONS)
        A = _finite_array(
            state.get("A"),
            "A",
            (action_count, model.n_features, model.n_features),
        )
        b = _finite_array(
            state.get("b"), "b", (action_count, model.n_features)
        )
        counts = _count_array(state.get("counts"), action_count)
        discounted_counts = _finite_array(
            state.get("discounted_counts"), "discounted_counts", (action_count,)
        )
        if np.any(discounted_counts < 0):
            raise ValueError("discounted_counts cannot contain negative values")
        if not np.array_equal(A, A.swapaxes(1, 2)):
            raise ValueError("A must contain symmetric Gram matrices")
        try:
            for matrix in A:
                np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("A must contain positive-definite Gram matrices") from exc

        model.A = A
        model.b = b
        model.counts = counts
        model.discounted_counts = discounted_counts
        return model

    def numeric_nbytes(self) -> int:
        """Return the bytes in the model's fixed-size NumPy state arrays."""

        return int(self.A.nbytes + self.b.nbytes + self.counts.nbytes + self.discounted_counts.nbytes)

    def _features(self, features: Any) -> np.ndarray:
        try:
            raw = np.asarray(features)
        except (TypeError, ValueError) as exc:
            raise ValueError("features must be a finite numeric vector") from exc
        expected_shape = (self.n_features,)
        if raw.shape != expected_shape:
            raise ValueError(f"features must have shape {expected_shape}, got {raw.shape}")
        if raw.dtype.kind not in "fiu":
            raise ValueError("features must contain real numeric values")
        result = np.array(raw, dtype=np.float64, copy=True)
        if not np.isfinite(result).all():
            raise ValueError("features must contain only finite values")
        return result

    def _prediction_details(
        self, features: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float]]:
        values: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        for index, action in enumerate(self.ACTIONS):
            coefficient = np.linalg.solve(self.A[index], self.b[index])
            leverage_vector = np.linalg.solve(self.A[index], features)
            values[action] = float(np.dot(features, coefficient))
            uncertainty[action] = float(max(0.0, np.dot(features, leverage_vector)))
        return values, uncertainty

    def _greedy_action(self, predicted: dict[str, float]) -> str:
        maximum = max(predicted.values())
        tied = [action for action in self.ACTIONS if predicted[action] == maximum]
        return self.default_action if self.default_action in tied else tied[0]

    @classmethod
    def _action_index(cls, action: str) -> int:
        if not isinstance(action, str) or action not in cls.ACTIONS:
            raise ValueError(f"action must be one of {cls.ACTIONS}")
        return cls.ACTIONS.index(action)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real number")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _finite_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _count_array(value: Any, action_count: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must be a nonnegative integer array") from exc
    if raw.shape != (action_count,) or raw.dtype.kind not in "iu":
        raise ValueError(f"counts must be an integer array with shape ({action_count},)")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("counts exceed int64 capacity")
    result = np.array(raw, dtype=np.int64, copy=True)
    if np.any(result < 0):
        raise ValueError("counts cannot contain negative values")
    return result


__all__ = ["Choice", "RetentionValueModel"]
