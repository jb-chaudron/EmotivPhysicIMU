from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Mapping, Tuple

import numpy as np
from matplotlib.figure import Figure

from numpy.typing import NDArray
from sklearn.model_selection import LeaveOneGroupOut
from emotivphysicimu.evaluation import evaluate_predictions, plot_predictions

try:
    from .models import EEGRegressor, EvalSet, FeatureBlocks, build_model
except ImportError:  # Allows running run_cleaning.py directly from this folder.
    from models import EEGRegressor, EvalSet, FeatureBlocks, build_model


class MLPipeline:
    """src: (N_samples, N_features), tgt: (N_samples, N_channels)."""

    def __init__(
        self,
        src_features: NDArray,
        tgt_features: NDArray,
        EEG_channels: list[str],
        model_config: Mapping[str, Any] | Any | None = None,
        feature_blocks: FeatureBlocks | None = None,
    ) -> None:
        if tgt_features.shape[1] != len(EEG_channels):
            raise ValueError(
                f"tgt_features has {tgt_features.shape[1]} channels but "
                f"len(EEG_channels)={len(EEG_channels)}"
            )
        self.src_features = np.asarray(src_features, dtype=float)
        self.tgt_features = np.asarray(tgt_features, dtype=float)
        self.EEG_channels = list(EEG_channels)
        self.model_config = model_config or {"name": "linear_regression"}
        self.feature_blocks = dict(feature_blocks or {})
        self._model: EEGRegressor | None = None
        self._n_features: int = self.src_features.shape[1]

    def split_data(
        self, groups: NDArray
    ) -> Generator[Tuple[NDArray, NDArray, NDArray, NDArray], None, None]:
        """Leave-one-group-out on ``groups`` (one row = one subject id)."""
        cv = LeaveOneGroupOut()
        for tr, te in cv.split(self.src_features, self.tgt_features, groups=groups):
            yield (
                self.src_features[tr],
                self.src_features[te],
                self.tgt_features[tr],
                self.tgt_features[te],
            )

    def _maybe_shuffle_train(
        self, X: NDArray, y: NDArray, shuffle: bool, random_state: int
    ) -> Tuple[NDArray, NDArray]:
        if not shuffle:
            return X, y
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        return X[idx], y[idx]

    def _new_model(self) -> EEGRegressor:
        return build_model(
            self.model_config,
            eeg_channels=self.EEG_channels,
            feature_blocks=self.feature_blocks,
        )

    def _fit_model(
        self, X: NDArray, y: NDArray, eval_set: EvalSet = None
    ) -> EEGRegressor:
        model = self._new_model()
        model.fit(X, y, eval_set=eval_set)
        return model

    @staticmethod
    def _metrics(y_true: NDArray, y_pred: NDArray) -> dict[str, float]:
        return evaluate_predictions(y_true, y_pred)

    def train(
        self,
        groups: NDArray,
        model_config: Mapping[str, Any] | Any | None = None,
        shuffle_train: bool = False,
        random_state: int = 42,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Leave-one-subject-out CV, then refit on full data. ``predict`` keeps row order of ``X``."""
        if model_config is not None:
            self.model_config = model_config

        fold_rows: list[dict[str, Any]] = []
        for fold_i, (X_tr, X_te, y_tr, y_te) in enumerate(self.split_data(groups)):
            X_trf, y_trf = self._maybe_shuffle_train(
                X_tr, y_tr, shuffle_train, random_state + fold_i
            )
            fold_model = self._fit_model(X_trf, y_trf, eval_set=(X_te, y_te))

            y_tr_hat = fold_model.predict(X_tr)
            y_te_hat = fold_model.predict(X_te)
            fold_rows.append(
                {
                    "fold": fold_i,
                    "train": self._metrics(y_tr, y_tr_hat),
                    "test": self._metrics(y_te, y_te_hat),
                }
            )

        report = self._aggregate_report(fold_rows)
        if verbose:
            self._print_report(report)

        # Refit on all data (in time order; optional shuffle of training rows only)
        X_all, y_all = self.src_features, self.tgt_features
        X_fit, y_fit = self._maybe_shuffle_train(
            X_all, y_all, shuffle_train, random_state
        )
        self._model = self._fit_model(X_fit, y_fit)

        return report

    @staticmethod
    def _aggregate_report(folds: list[dict[str, Any]]) -> dict[str, Any]:
        def mean_std(split: str, key: str) -> tuple[float, float]:
            vals = np.array([f[split][key] for f in folds], dtype=float)
            return float(vals.mean()), float(vals.std())

        out: dict[str, Any] = {"n_folds": len(folds), "folds": folds}
        for split in ("train", "test"):
            for m in ("rmse", "mae", "r2"):
                mu, sd = mean_std(split, m)
                out[f"{split}_{m}_mean"] = mu
                out[f"{split}_{m}_std"] = sd
        return out

    @staticmethod
    def _print_report(report: dict[str, Any]) -> None:
        print(
            f"CV report ({report['n_folds']} folds, leave-one-group-out)\n"
            f"  Train — RMSE: {report['train_rmse_mean']:.4f} ± {report['train_rmse_std']:.4f} | "
            f"MAE: {report['train_mae_mean']:.4f} ± {report['train_mae_std']:.4f} | "
            f"R²: {report['train_r2_mean']:.4f} ± {report['train_r2_std']:.4f}\n"
            f"  Test  — RMSE: {report['test_rmse_mean']:.4f} ± {report['test_rmse_std']:.4f} | "
            f"MAE: {report['test_mae_mean']:.4f} ± {report['test_mae_std']:.4f} | "
            f"R²: {report['test_r2_mean']:.4f} ± {report['test_r2_std']:.4f}"
        )

    def predict(self, X: NDArray) -> NDArray:
        """Predict in the same row order as ``X`` (time order preserved)."""
        if self._model is None:
            raise RuntimeError("MLPipeline must be trained before predict().")
        X = np.asarray(X, dtype=float)
        return self._model.predict(X)

    def visualize_results(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        n_show: int = 1_000,
        channel: int = 0,
        path: Path | None = None,
    ) -> Figure:
        """First ``n_show`` time points: scatter, traces, residual."""
        return plot_predictions(
            y_true,
            y_pred,
            channel_names=self.EEG_channels,
            n_show=n_show,
            channel=channel,
            path=path,
        )


if __name__ == "__main__":
    pass

