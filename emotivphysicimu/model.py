from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import EMOTIV_CHANNELS
from .metrics import (
    coherence,
    eeg_quality,
    mutual_information_columns,
    pearson,
    prediction_metrics,
    rank_composite,
)

NUMERATORS: NDArray = np.outer(
    [1, 3, 5, 7],
    10.0 ** np.array([-3, -2, -1, 1]),
).ravel()


class IMURegressor:
    """sklearn-style multi-channel EEG artefact regressor.

    Inputs are channels-first: ``X`` has shape ``(n_channels, n_points, n_features)``
    and ``y`` has shape ``(n_channels, n_points)``. The class wraps either a single
    sklearn-compatible estimator (``channel_handling='all_channels'``) or one
    estimator per EEG channel (``channel_handling='per_channel'``) and supports an
    ordered residual chain over named feature blocks.

    A ``conformal=True`` post-hoc step fits a ``MapieRegressor`` (prefit, on a
    calibration tail) and searches a per-channel numerator over
    ``{1,3,5,7} * 10^{-3..1}``; ``clean(y, X)`` returns ``y - coef * predict(X)``
    where ``coef = numerator / mapie_interval_width`` per sample.

    Parameters
    ----------
    eeg_channels : list of str
        Channel names; length must equal ``X.shape[0]`` and ``y.shape[0]``.
    feature_blocks : dict of str -> slice, optional
        Required when ``chain`` is set; slices index the last axis of ``X``.
    base_model : {'linear_regression', 'tweedie', 'catboost'}
    params : dict, optional
        Extra keyword arguments forwarded to the underlying estimator.
    channel_handling : {'per_channel', 'all_channels'}
    use_electrode_feature : bool
        Catboost-only: append channel name as a categorical feature when
        ``channel_handling='all_channels'``.
    chain : list of dict, optional
        Residual chain: ``[{'features': <block_name>, 'model': {'base_model': ..., 'params': ...}}, ...]``.
    conformal : bool
        Enable the MAPIE-driven conformal post-hoc.
    conformal_alpha : float
        Alpha used by MAPIE to compute prediction-interval widths (denominator).
    calibration_fraction : float
        Fraction of training samples (along the time axis) reserved for MAPIE
        calibration and numerator selection.
    sfreq : float, optional
        EEG sampling frequency, required for ``score()`` / ``correlation_score()``
        and the conformal selector.
    random_state : int
    """

    def __init__(
        self,
        eeg_channels: Sequence[str] = EMOTIV_CHANNELS,
        feature_blocks: Mapping[str, slice] | None = None,
        base_model: str = "linear_regression",
        params: Mapping[str, Any] | None = None,
        channel_handling: str = "per_channel",
        use_electrode_feature: bool = False,
        chain: Sequence[Mapping[str, Any]] | None = None,
        conformal: bool = False,
        conformal_alpha: float = 0.1,
        calibration_fraction: float = 0.2,
        sfreq: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.eeg_channels = list(eeg_channels)
        self.feature_blocks = dict(feature_blocks) if feature_blocks is not None else None
        self.base_model = base_model
        self.params = dict(params or {})
        self.channel_handling = channel_handling
        self.use_electrode_feature = use_electrode_feature
        self.chain = [dict(stage) for stage in chain] if chain else None
        self.conformal = conformal
        self.conformal_alpha = conformal_alpha
        self.calibration_fraction = calibration_fraction
        self.sfreq = sfreq
        self.random_state = random_state

        self._validate_init()

        # Fitted attributes
        self._estimators: list = []          # per_channel: one per channel
        self._estimator = None               # all_channels: one estimator
        self._chain_models: list[tuple[str, "IMURegressor"]] = []
        self._mapie_models: list = []        # per_channel conformal
        self._mapie = None                   # all_channels conformal
        self.numerators_: NDArray | None = None

    def _validate_init(self) -> None:
        if self.channel_handling not in ("per_channel", "all_channels"):
            raise ValueError(
                f"channel_handling must be 'per_channel' or 'all_channels', got {self.channel_handling!r}"
            )
        if self.use_electrode_feature:
            if self.channel_handling != "all_channels":
                raise ValueError("use_electrode_feature requires channel_handling='all_channels'")
            if _normalize_name(self.base_model) != "catboost":
                raise ValueError("use_electrode_feature requires base_model='catboost'")
        if self.chain is not None:
            if self.feature_blocks is None:
                raise ValueError("chain requires feature_blocks")
            for stage in self.chain:
                if "features" not in stage:
                    raise ValueError("Each chain stage needs a 'features' key")
                if stage["features"] not in self.feature_blocks:
                    raise ValueError(
                        f"Chain stage references unknown feature block: {stage['features']!r}"
                    )
            if self.conformal:
                raise NotImplementedError("conformal is not yet supported with chain")
        if self.conformal and self.sfreq is None:
            raise ValueError("conformal=True requires sfreq to be set")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")

    def fit(self, X: NDArray, y: NDArray) -> "IMURegressor":
        X = _as_3d(X)
        y = _as_2d(y)
        self._validate_xy(X, y)

        if self.chain is not None:
            self._fit_chain(X, y)
        elif self.conformal:
            self._fit_with_conformal(X, y)
        elif self.channel_handling == "per_channel":
            self._fit_per_channel(X, y)
        else:
            self._fit_all_channels(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        X = _as_3d(X)
        if self.chain is not None:
            return self._predict_chain(X)
        if self.channel_handling == "per_channel":
            return self._predict_per_channel(X)
        return self._predict_all_channels(X)

    def clean(self, y: NDArray, X: NDArray) -> NDArray:
        """Return ``y - coef * predict(X)`` (per-sample coef from MAPIE if conformal)."""
        y = _as_2d(y)
        X = _as_3d(X)
        if not self.conformal:
            return y - self.predict(X)
        y_pred, widths = self._predict_with_widths(X)
        if self.numerators_ is None:
            raise RuntimeError("conformal numerators not fitted; call fit() first")
        eps = 1e-12
        coef = self.numerators_[:, None] / np.maximum(widths, eps)
        return y - coef * y_pred

    def score(self, X: NDArray, y: NDArray) -> dict[str, dict[str, NDArray]]:
        """Per-channel EEG quality before vs. after correction.

        Returns a dict with three sub-dicts: ``"raw"`` (metrics on the input
        ``y``), ``"cleaned"`` (metrics on ``clean(y, X)``), and ``"delta"``
        (cleaned - raw). Each sub-dict has the keys ``kurtosis``,
        ``spectral_slope``, and ``hfp_tp`` with shape ``(n_channels,)``.
        """
        if self.sfreq is None:
            raise ValueError("score() requires sfreq to be set")
        y_raw = _as_2d(y)
        cleaned = self.clean(y, X)
        raw_q = [eeg_quality(y_raw[c], sfreq=self.sfreq) for c in range(y_raw.shape[0])]
        clean_q = [eeg_quality(cleaned[c], sfreq=self.sfreq) for c in range(cleaned.shape[0])]
        keys = ("kurtosis", "spectral_slope", "hfp_tp")
        raw = {k: np.array([m[k] for m in raw_q]) for k in keys}
        cleaned_d = {k: np.array([m[k] for m in clean_q]) for k in keys}
        delta = {k: cleaned_d[k] - raw[k] for k in keys}
        return {"raw": raw, "cleaned": cleaned_d, "delta": delta}

    def prediction_score(self, X: NDArray, y: NDArray) -> dict[str, NDArray]:
        """Per-channel RMSE / MAE / R^2 of ``predict(X)`` vs ``y``."""
        y = _as_2d(y)
        y_pred = self.predict(X)
        rmse = np.zeros(y.shape[0])
        mae = np.zeros(y.shape[0])
        r2 = np.zeros(y.shape[0])
        for c in range(y.shape[0]):
            m = prediction_metrics(y[c], y_pred[c])
            rmse[c] = m["rmse"]
            mae[c] = m["mae"]
            r2[c] = m["r2"]
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def correlation_score(self, X: NDArray) -> dict[str, NDArray]:
        """Per-channel x per-regressor zero-lag association with the predicted time series.

        Returns a dict with keys ``pearson``, ``coherence``, ``mutual_information``;
        each value has shape ``(n_channels, n_features)``.
        """
        if self.sfreq is None:
            raise ValueError("correlation_score() requires sfreq to be set")
        X = _as_3d(X)
        y_pred = self.predict(X)
        n_channels, _, n_features = X.shape
        pear = np.zeros((n_channels, n_features), dtype=float)
        coh = np.zeros((n_channels, n_features), dtype=float)
        mi = np.zeros((n_channels, n_features), dtype=float)
        for c in range(n_channels):
            for f in range(n_features):
                pear[c, f] = pearson(X[c, :, f], y_pred[c])
                coh[c, f] = coherence(X[c, :, f], y_pred[c], sfreq=self.sfreq)
            mi[c] = mutual_information_columns(
                X[c], y_pred[c], random_state=self.random_state
            )
        return {"pearson": pear, "coherence": coh, "mutual_information": mi}

    # --- internal: per-channel / all-channels --------------------------------------

    def _fit_per_channel(self, X: NDArray, y: NDArray) -> None:
        self._estimators = []
        for c in range(y.shape[0]):
            est = _make_2d_estimator(self.base_model, self.params, self.random_state)
            est.fit(X[c], y[c])
            self._estimators.append(est)

    def _predict_per_channel(self, X: NDArray) -> NDArray:
        if not self._estimators:
            raise RuntimeError("Model is not fitted")
        return np.stack(
            [self._estimators[c].predict(X[c]) for c in range(len(self._estimators))],
            axis=0,
        )

    def _fit_all_channels(self, X: NDArray, y: NDArray) -> None:
        X_flat, y_flat = self._flatten_all_channels(X, y)
        cat_idx = X.shape[2] if self.use_electrode_feature else None
        self._estimator = _make_2d_estimator(
            self.base_model, self.params, self.random_state, cat_feature_idx=cat_idx
        )
        self._estimator.fit(X_flat, y_flat)

    def _predict_all_channels(self, X: NDArray) -> NDArray:
        if self._estimator is None:
            raise RuntimeError("Model is not fitted")
        n_channels, n_points, _ = X.shape
        X_flat = self._flatten_X(X)
        return self._estimator.predict(X_flat).reshape(n_channels, n_points)

    def _flatten_X(self, X: NDArray) -> NDArray:
        n_channels, n_points, n_features = X.shape
        X_flat = X.reshape(n_channels * n_points, n_features).astype(float, copy=False)
        if self.use_electrode_feature:
            channel_names = np.repeat(np.asarray(self.eeg_channels, dtype=object), n_points)
            out = np.empty((X_flat.shape[0], n_features + 1), dtype=object)
            out[:, :n_features] = X_flat
            out[:, n_features] = channel_names
            return out
        return X_flat

    def _flatten_all_channels(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        return self._flatten_X(X), y.reshape(-1).astype(float, copy=False)

    # --- internal: residual chain --------------------------------------------------

    def _fit_chain(self, X: NDArray, y: NDArray) -> None:
        self._chain_models = []
        residual = y.copy()
        for stage in self.chain or []:
            block = self.feature_blocks[stage["features"]]  # type: ignore[index]
            stage_model = stage.get("model") or {}
            sub = IMURegressor(
                eeg_channels=self.eeg_channels,
                feature_blocks=None,
                base_model=stage_model.get("base_model", self.base_model),
                params=stage_model.get("params", self.params),
                channel_handling=self.channel_handling,
                use_electrode_feature=self.use_electrode_feature,
                chain=None,
                conformal=False,
                sfreq=self.sfreq,
                random_state=self.random_state,
            )
            sub.fit(X[:, :, block], residual)
            residual = residual - sub.predict(X[:, :, block])
            self._chain_models.append((stage["features"], sub))

    def _predict_chain(self, X: NDArray) -> NDArray:
        if not self._chain_models:
            raise RuntimeError("Model is not fitted")
        n_channels, n_points, _ = X.shape
        out = np.zeros((n_channels, n_points), dtype=float)
        for features_name, sub in self._chain_models:
            block = self.feature_blocks[features_name]  # type: ignore[index]
            out = out + sub.predict(X[:, :, block])
        return out

    # --- internal: MAPIE conformal -------------------------------------------------

    def _fit_with_conformal(self, X: NDArray, y: NDArray) -> None:
        n_points = X.shape[1]
        cal_size = max(1, int(round(n_points * self.calibration_fraction)))
        if cal_size >= n_points:
            raise ValueError("calibration_fraction too large for the dataset size")
        train_end = n_points - cal_size

        X_train = X[:, :train_end, :]
        y_train = y[:, :train_end]
        X_cal = X[:, train_end:, :]
        y_cal = y[:, train_end:]

        confidence_level = float(np.clip(1.0 - self.conformal_alpha, 1e-3, 1 - 1e-3))
        if self.channel_handling == "per_channel":
            self._fit_per_channel(X_train, y_train)
            self._mapie_models = []
            for c in range(y.shape[0]):
                mapie = _build_mapie(self._estimators[c], confidence_level)
                mapie.conformalize(X_cal[c], y_cal[c])
                self._mapie_models.append(mapie)
            self.numerators_ = self._search_numerators_per_channel(X_cal, y_cal)
        else:
            self._fit_all_channels(X_train, y_train)
            X_cal_flat = self._flatten_X(X_cal)
            y_cal_flat = y_cal.reshape(-1).astype(float, copy=False)
            self._mapie = _build_mapie(self._estimator, confidence_level)
            self._mapie.conformalize(X_cal_flat, y_cal_flat)
            self.numerators_ = self._search_numerators_all_channels(X_cal, y_cal)

    def _predict_with_widths(self, X: NDArray) -> tuple[NDArray, NDArray]:
        """Return (y_pred, widths) shaped (C, N) using MAPIE intervals."""
        if self.channel_handling == "per_channel":
            if not self._mapie_models:
                raise RuntimeError("Conformal not fitted")
            preds = []
            widths = []
            for c, mapie in enumerate(self._mapie_models):
                p, w = _mapie_point_and_width(mapie, X[c])
                preds.append(p)
                widths.append(w)
            return np.stack(preds, axis=0), np.stack(widths, axis=0)

        if self._mapie is None:
            raise RuntimeError("Conformal not fitted")
        n_channels, n_points, _ = X.shape
        X_flat = self._flatten_X(X)
        y_pred_flat, widths_flat = _mapie_point_and_width(self._mapie, X_flat)
        return (
            y_pred_flat.reshape(n_channels, n_points),
            widths_flat.reshape(n_channels, n_points),
        )

    def _search_numerators_per_channel(self, X_cal: NDArray, y_cal: NDArray) -> NDArray:
        if self.sfreq is None:
            raise ValueError("conformal selector requires sfreq")
        n_channels = y_cal.shape[0]
        numerators = np.zeros(n_channels, dtype=float)
        for c in range(n_channels):
            y_pred_c, widths = _mapie_point_and_width(self._mapie_models[c], X_cal[c])
            numerators[c] = self._best_numerator(y_cal[c], y_pred_c, widths)
        return numerators

    def _search_numerators_all_channels(self, X_cal: NDArray, y_cal: NDArray) -> NDArray:
        if self.sfreq is None:
            raise ValueError("conformal selector requires sfreq")
        n_channels, n_points, _ = X_cal.shape
        X_cal_flat = self._flatten_X(X_cal)
        y_pred_flat, widths_flat = _mapie_point_and_width(self._mapie, X_cal_flat)
        y_pred = y_pred_flat.reshape(n_channels, n_points)
        widths = widths_flat.reshape(n_channels, n_points)
        numerators = np.zeros(n_channels, dtype=float)
        for c in range(n_channels):
            numerators[c] = self._best_numerator(y_cal[c], y_pred[c], widths[c])
        return numerators

    def _best_numerator(self, y_true: NDArray, y_pred: NDArray, widths: NDArray) -> float:
        eps = 1e-12
        candidates = []
        for numerator in NUMERATORS:
            coef = numerator / np.maximum(widths, eps)
            cleaned = y_true - coef * y_pred
            candidates.append(eeg_quality(cleaned, sfreq=self.sfreq))
        best = rank_composite(candidates)
        return float(NUMERATORS[best])

    # --- validation ---------------------------------------------------------------

    def _validate_xy(self, X: NDArray, y: NDArray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X / y channel mismatch: X {X.shape[0]} vs y {y.shape[0]}")
        if X.shape[1] != y.shape[1]:
            raise ValueError(f"X / y time mismatch: X {X.shape[1]} vs y {y.shape[1]}")
        if X.shape[0] != len(self.eeg_channels):
            raise ValueError(
                f"X has {X.shape[0]} channels but {len(self.eeg_channels)} channel names were provided"
            )


# --- helpers ------------------------------------------------------------------------


def _normalize_name(name: str | None) -> str:
    if name is None:
        return "linear_regression"
    normalized = name.replace("-", "_").lower()
    aliases = {
        "linearregression": "linear_regression",
        "linear_regression": "linear_regression",
        "lr": "linear_regression",
        "tweedieregressor": "tweedie",
        "tweedie": "tweedie",
        "catboostregressor": "catboost",
        "catboost": "catboost",
    }
    return aliases.get(normalized, normalized)


def _make_2d_estimator(
    name: str,
    params: Mapping[str, Any] | None,
    random_state: int | None,
    cat_feature_idx: int | None = None,
):
    """Return a sklearn-compatible 2D regressor."""
    name = _normalize_name(name)
    params = dict(params or {})
    if name == "linear_regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression(**params))])
    if name == "tweedie":
        defaults: dict[str, Any] = {"power": 1.5, "alpha": 0.5, "max_iter": 200}
        merged = {**defaults, **params}
        return Pipeline([("scaler", StandardScaler()), ("model", TweedieRegressor(**merged))])
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "CatBoost support is optional. Install it with `pip install emotivphysicimu[catboost]`."
            ) from exc
        defaults = {"iterations": 80, "verbose": False, "allow_writing_files": False}
        if random_state is not None:
            defaults["random_seed"] = random_state
        merged = {**defaults, **params}
        if cat_feature_idx is not None:
            merged["cat_features"] = [cat_feature_idx]
        return CatBoostRegressor(**merged)
    raise ValueError(f"Unknown base model: {name!r}")


def _build_mapie(estimator, confidence_level: float):
    """Wrap a fitted sklearn estimator with MAPIE's prefit split-conformal regressor."""
    try:
        from mapie.regression import SplitConformalRegressor
    except ImportError as exc:
        raise ImportError(
            "conformal=True requires the `mapie` package; install with `pip install mapie`."
        ) from exc
    return SplitConformalRegressor(
        estimator=estimator,
        confidence_level=confidence_level,
        prefit=True,
    )


def _mapie_point_and_width(mapie, X) -> tuple[NDArray, NDArray]:
    """Return (point_pred, interval_width) per sample from a conformalized MAPIE model."""
    point_pred, intervals = mapie.predict_interval(X)
    point_pred = np.asarray(point_pred).ravel()
    intervals = np.asarray(intervals)
    widths = intervals[:, 1, 0] - intervals[:, 0, 0]
    return point_pred, widths


def _as_3d(X: NDArray) -> NDArray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (n_channels, n_points, n_features), got {X.shape}")
    return X


def _as_2d(y: NDArray) -> NDArray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (n_channels, n_points), got {y.shape}")
    return y
