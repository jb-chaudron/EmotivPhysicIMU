from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HFP_LOW_CUTOFF_HZ = 30.0


def prediction_metrics(y_true: NDArray, y_pred: NDArray) -> dict[str, float]:
    """Flat RMSE / MAE / R2 across all samples and channels."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def eeg_quality(y: NDArray, sfreq: float) -> dict[str, float]:
    """EEG-quality metrics on a 1D signal: kurtosis, spectral slope, HFP/TP."""
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 4:
        return {"kurtosis": 0.0, "spectral_slope": 0.0, "hfp_tp": 0.0}

    kurt = float(sp_stats.kurtosis(y, fisher=False))

    nperseg = min(256, y.size)
    freqs, psd = sp_signal.welch(y, fs=sfreq, nperseg=nperseg)
    valid = (freqs > 0) & (psd > 0)
    if valid.sum() >= 2:
        slope = float(np.polyfit(np.log10(freqs[valid]), np.log10(psd[valid]), 1)[0])
    else:
        slope = 0.0

    total_power = float(psd.sum())
    hfp = float(psd[freqs >= HFP_LOW_CUTOFF_HZ].sum())
    hfp_tp = hfp / total_power if total_power > 0 else 0.0

    return {"kurtosis": kurt, "spectral_slope": slope, "hfp_tp": hfp_tp}


def rank_composite(metrics_per_candidate: Sequence[dict[str, float]]) -> int:
    """Pick the candidate with the highest mean rank across EEG-quality metrics.

    Higher rank = better. Ranks are computed so that:
      - `hfp_tp` ascending (higher = better),
      - `|kurtosis - 3|` descending (closer to 3 = better).
    """
    if len(metrics_per_candidate) == 0:
        raise ValueError("Need at least one candidate")

    hfp = np.array([m.get("hfp_tp", 0.0) for m in metrics_per_candidate])
    kurt_dist = np.array([abs(m.get("kurtosis", 0.0) - 3.0) for m in metrics_per_candidate])

    rank_components = [sp_stats.rankdata(hfp), sp_stats.rankdata(-kurt_dist)]
    avg = np.mean(np.stack(rank_components, axis=0), axis=0)
    return int(np.argmax(avg))


def pearson(a: NDArray, b: NDArray) -> float:
    """Zero-lag Pearson correlation between two 1D signals."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = min(a.size, b.size)
    if m < 2:
        return 0.0
    a = a[:m] - a[:m].mean()
    b = b[:m] - b[:m].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def coherence(a: NDArray, b: NDArray, sfreq: float) -> float:
    """Mean magnitude-squared coherence between two 1D signals over Welch bins."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = min(a.size, b.size)
    if m < 8:
        return 0.0
    nperseg = min(256, m)
    _, cxy = sp_signal.coherence(a[:m], b[:m], fs=sfreq, nperseg=nperseg)
    if cxy.size == 0:
        return 0.0
    return float(np.nanmean(cxy))


def mutual_information_columns(
    X_2d: NDArray,
    y_1d: NDArray,
    *,
    random_state: int = 0,
) -> NDArray:
    """Mutual information between each column of `X_2d` and `y_1d`.

    Returns one MI value per column of X (shape: (n_features,)).
    """
    X_2d = np.asarray(X_2d, dtype=float)
    y_1d = np.asarray(y_1d, dtype=float).ravel()
    if X_2d.ndim != 2:
        raise ValueError(f"X_2d must be 2D, got {X_2d.shape}")
    if y_1d.size < 2 or X_2d.shape[0] < 2:
        return np.zeros(X_2d.shape[1], dtype=float)
    return mutual_info_regression(X_2d, y_1d, random_state=random_state)
