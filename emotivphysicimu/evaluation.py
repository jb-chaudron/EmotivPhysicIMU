from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .metrics import prediction_metrics


def evaluate_predictions(y_true: NDArray, y_pred: NDArray) -> dict[str, float]:
    return prediction_metrics(y_true, y_pred)


def plot_predictions(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    channel_names: Sequence[str],
    n_show: int = 1_000,
    channel: int = 0,
    path: Path | str | None = None,
) -> Figure:
    """First n_show time points: scatter, traces, residual."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    channel_names = list(channel_names)

    if y_true.ndim == 1:
        yt = y_true[:n_show]
        yp = y_pred[:n_show]
        name = channel_names[0] if channel_names else "ch"
    else:
        yt = y_true[:n_show, channel]
        yp = y_pred[:n_show, channel]
        name = channel_names[channel]

    t = np.arange(len(yt))
    resid = yt - yp

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), constrained_layout=True)
    ax[0].scatter(yt, yp, s=12, alpha=0.7)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax[0].plot(lims, lims, "k--", alpha=0.4, lw=1)
    ax[0].set_xlabel("True")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title(f"{name}: true vs predicted ({len(yt)} points)")

    ax[1].plot(t, yt, label="True", alpha=0.85)
    ax[1].plot(t, yp, label="Predicted", alpha=0.85)
    ax[1].set_xlabel("Time index")
    ax[1].set_ylabel("Value")
    ax[1].legend()
    ax[1].set_title(f"{name}: time series")

    ax[2].plot(t, resid, color="C2")
    ax[2].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax[2].set_xlabel("Time index")
    ax[2].set_ylabel("True - pred")
    ax[2].set_title(f"{name}: residual")

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    else:
        plt.show()
    return fig
