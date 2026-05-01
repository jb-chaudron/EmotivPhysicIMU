from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .base import EEGRegressor, EvalSet, FeatureBlocks, require_feature_block

ModelBuilder = Callable[[], EEGRegressor]


class TwoStepResidualRegressor:
    """Fit a first model, then a second model on the first model's residual."""

    def __init__(
        self,
        eeg_channels: Sequence[str],
        feature_blocks: FeatureBlocks,
        first_features: str,
        residual_features: str,
        first_model_builder: ModelBuilder,
        residual_model_builder: ModelBuilder | None = None,
    ) -> None:
        self.eeg_channels = list(eeg_channels)
        self.feature_blocks = feature_blocks
        self.first_features = first_features
        self.residual_features = residual_features
        self.first_model_builder = first_model_builder
        self.residual_model_builder = residual_model_builder or first_model_builder
        self.first_model: EEGRegressor | None = None
        self.residual_model: EEGRegressor | None = None

    def fit(
        self, X: NDArray, y: NDArray, eval_set: EvalSet = None
    ) -> "TwoStepResidualRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        first_cols = require_feature_block(self.feature_blocks, self.first_features)
        residual_cols = require_feature_block(
            self.feature_blocks, self.residual_features
        )
        first_eval_set = None
        residual_eval_X = None
        residual_eval_y = None
        if eval_set is not None:
            eval_X, eval_y = eval_set
            eval_X = np.asarray(eval_X, dtype=float)
            eval_y = np.asarray(eval_y, dtype=float)
            first_eval_set = (eval_X[:, first_cols], eval_y)
            residual_eval_X = eval_X[:, residual_cols]
            residual_eval_y = eval_y

        self.first_model = self.first_model_builder()
        self.first_model.fit(X[:, first_cols], y, eval_set=first_eval_set)
        residual = y - self.first_model.predict(X[:, first_cols])

        self.residual_model = self.residual_model_builder()
        residual_eval_set = None
        if residual_eval_X is not None and residual_eval_y is not None:
            residual_eval = residual_eval_y - self.first_model.predict(
                eval_X[:, first_cols]
            )
            residual_eval_set = (residual_eval_X, residual_eval)
        self.residual_model.fit(
            X[:, residual_cols], residual, eval_set=residual_eval_set
        )
        return self

    def predict(self, X: NDArray) -> NDArray:
        if self.first_model is None or self.residual_model is None:
            raise RuntimeError("Model must be fitted before predict().")
        X = np.asarray(X, dtype=float)
        first_cols = require_feature_block(self.feature_blocks, self.first_features)
        residual_cols = require_feature_block(
            self.feature_blocks, self.residual_features
        )
        return self.first_model.predict(X[:, first_cols]) + self.residual_model.predict(
            X[:, residual_cols]
        )
