from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from catboost import CatBoostRegressor as _CatBoostRegressor
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from .base import EvalSet

CatBoostChannelHandling = Literal["multioutput", "channel_feature"]


def _stack_numeric_and_categorical(X_num: NDArray, channel_names: NDArray) -> NDArray:
    out = np.empty((X_num.shape[0], X_num.shape[1] + 1), dtype=object)
    out[:, :-1] = X_num.astype(float, copy=False)
    out[:, -1] = np.asarray(channel_names, dtype=object)
    return out


class CatBoostEEGRegressor:
    """CatBoost backend for multi-channel EEG regression."""

    def __init__(
        self,
        eeg_channels: Sequence[str],
        channel_handling: CatBoostChannelHandling = "multioutput",
        random_state: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if channel_handling not in ("multioutput", "channel_feature"):
            raise ValueError(
                "channel_handling must be 'multioutput' or 'channel_feature'"
            )
        self.eeg_channels = list(eeg_channels)
        self.channel_handling = channel_handling
        self.params = self._default_params(random_state) | dict(params or {})
        self._multioutput_models: list[tuple[StandardScaler, _CatBoostRegressor]] = []
        self._channel_scaler: StandardScaler | None = None
        self._channel_model: _CatBoostRegressor | None = None

    @staticmethod
    def _default_params(random_state: int | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "iterations": 80,
            "verbose": False,
            "allow_writing_files": False,
        }
        if random_state is not None:
            params["random_seed"] = random_state
        return params

    def fit(
        self, X: NDArray, y: NDArray, eval_set: EvalSet = None
    ) -> "CatBoostEEGRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.shape[1] != len(self.eeg_channels):
            raise ValueError(
                f"y has {y.shape[1]} channels but {len(self.eeg_channels)} names were provided"
            )
        if self.channel_handling == "multioutput":
            self._fit_multioutput(X, y, eval_set=eval_set)
        else:
            self._fit_channel_feature(X, y, eval_set=eval_set)
        return self

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self.channel_handling == "multioutput":
            if not self._multioutput_models:
                raise RuntimeError("Model must be fitted before predict().")
            columns = [
                model.predict(scaler.transform(X))
                for scaler, model in self._multioutput_models
            ]
            return np.column_stack(columns)
        return self._predict_channel_feature(X)

    def _fit_multioutput(
        self, X: NDArray, y: NDArray, eval_set: EvalSet = None
    ) -> None:
        self._multioutput_models = []
        for channel_idx in range(y.shape[1]):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = _CatBoostRegressor(**self.params)
            fit_kwargs: dict[str, Any] = {}
            if eval_set is not None:
                X_eval, y_eval = eval_set
                fit_kwargs["eval_set"] = (
                    scaler.transform(np.asarray(X_eval, dtype=float)),
                    np.asarray(y_eval, dtype=float)[:, channel_idx],
                )
            model.fit(X_scaled, y[:, channel_idx], **fit_kwargs)
            self._multioutput_models.append((scaler, model))

    def _expand_channel_feature(
        self, X: NDArray, y: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        n_samples, n_channels = y.shape
        X_rep = np.repeat(X, n_channels, axis=0)
        channel_names = np.tile(np.asarray(self.eeg_channels, dtype=str), n_samples)
        y_flat = y.reshape(-1, order="C")
        return X_rep, channel_names, y_flat

    def _fit_channel_feature(
        self, X: NDArray, y: NDArray, eval_set: EvalSet = None
    ) -> None:
        X_rep, channel_names, y_flat = self._expand_channel_feature(X, y)
        self._channel_scaler = StandardScaler()
        X_num = self._channel_scaler.fit_transform(X_rep)
        cat_idx = X_num.shape[1]
        params = dict(self.params)
        params["cat_features"] = [cat_idx]
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval_rep, eval_channel_names, y_eval_flat = self._expand_channel_feature(
                np.asarray(X_eval, dtype=float),
                np.asarray(y_eval, dtype=float),
            )
            X_eval_num = self._channel_scaler.transform(X_eval_rep)
            fit_kwargs["eval_set"] = (
                _stack_numeric_and_categorical(X_eval_num, eval_channel_names),
                y_eval_flat,
            )
        self._channel_model = _CatBoostRegressor(**params)
        self._channel_model.fit(
            _stack_numeric_and_categorical(X_num, channel_names),
            y_flat,
            **fit_kwargs,
        )

    def _predict_channel_feature(self, X: NDArray) -> NDArray:
        if self._channel_scaler is None or self._channel_model is None:
            raise RuntimeError("Model must be fitted before predict().")
        y_shape = (X.shape[0], len(self.eeg_channels))
        X_rep, channel_names, _ = self._expand_channel_feature(
            X, np.zeros(y_shape, dtype=float)
        )
        X_num = self._channel_scaler.transform(X_rep)
        pred_flat = self._channel_model.predict(
            _stack_numeric_and_categorical(X_num, channel_names)
        )
        return pred_flat.reshape(y_shape, order="C")
