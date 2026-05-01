from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import EvalSet


class PerChannelLinearRegression:
    """Fit one scaled linear regressor per EEG channel."""

    def __init__(
        self,
        eeg_channels: Sequence[str],
        params: dict[str, Any] | None = None,
    ) -> None:
        self.eeg_channels = list(eeg_channels)
        self.params = dict(params or {})
        self.models: dict[str, Pipeline] = {}

    def fit(
        self, X: NDArray, y: NDArray, eval_set: EvalSet = None
    ) -> "PerChannelLinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.shape[1] != len(self.eeg_channels):
            raise ValueError(
                f"y has {y.shape[1]} channels but {len(self.eeg_channels)} names were provided"
            )

        self.models = {}
        for channel_idx, channel_name in enumerate(self.eeg_channels):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression(**self.params)),
                ]
            )
            pipe.fit(X, y[:, channel_idx])
            self.models[channel_name] = pipe
        return self

    def predict(self, X: NDArray) -> NDArray:
        if not self.models:
            raise RuntimeError("Model must be fitted before predict().")
        X = np.asarray(X, dtype=float)
        columns = [self.models[channel].predict(X) for channel in self.eeg_channels]
        return np.column_stack(columns)
