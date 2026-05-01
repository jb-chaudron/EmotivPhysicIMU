"""Self-contained HTML report for ``IMURegressor`` results."""
from __future__ import annotations

import base64
import html as _html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .metrics import prediction_metrics
from .model import IMURegressor

DEFAULT_PLOT_POINTS = 1000


@dataclass(frozen=True)
class _Split:
    name: str
    X: NDArray
    y: NDArray


_STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1180px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }
  h2 { margin-top: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.2em; }
  h3 { margin-top: 1.2em; color: #333; }
  table { border-collapse: collapse; margin: 0.5em 0 1em 0; font-size: 0.9em; }
  th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f5f5f7; }
  figure { margin: 1em 0; text-align: center; }
  img { max-width: 100%; height: auto; }
  figcaption { color: #666; font-size: 0.9em; margin-top: 0.4em; }
  details { margin: 0.5em 0 1em 0; }
  summary { cursor: pointer; font-weight: 600; }
  section { margin-bottom: 2em; }
  .all-row td { background: #fafafa; font-weight: 600; }
</style>
"""


class IMUReport:
    """Render a self-contained HTML report for a fitted ``IMURegressor``.

    The report includes:

    1. Model hyperparameters.
    2. Signal plots (residual, true vs predicted, cleaned) for ``n_plot_points``
       contiguous samples from a randomly picked EEG channel.
    3. Per-split metrics tables (prediction RMSE/MAE/R^2, EEG quality before
       vs. after cleaning, mean correlation between IMU regressors and the
       prediction).
    4. ICA topomaps and source time series before vs. after cleaning. Test
       data is preferred when provided; otherwise training data is used.

    Parameters
    ----------
    model : IMURegressor
        Already fitted regressor; ``model.sfreq`` must be set.
    X_train, y_train : array
        Training tensors with shapes ``(C, N, F)`` and ``(C, N)``.
    X_test, y_test : array, optional
        Test tensors. Must be provided together (both or neither).
    random_state : int, optional
        Seed for channel and window selection. Defaults to ``model.random_state``.
    ica_n_components : int, optional
        Number of ICA components. Defaults to ``len(model.eeg_channels)``.
    ica_max_iter : int
        Max iterations for ICA fitting (default 200).
    n_plot_points : int
        Window length used in section 2 and the ICA source plots (default 1000).
    """

    def __init__(
        self,
        model: IMURegressor,
        X_train: NDArray,
        y_train: NDArray,
        X_test: NDArray | None = None,
        y_test: NDArray | None = None,
        *,
        random_state: int | None = None,
        ica_n_components: int | None = None,
        ica_max_iter: int = 200,
        n_plot_points: int = DEFAULT_PLOT_POINTS,
    ) -> None:
        if model.sfreq is None:
            raise ValueError("model.sfreq must be set to generate a report")
        if (X_test is None) ^ (y_test is None):
            raise ValueError("X_test and y_test must be both provided or both None")

        self.model = model
        self.train = _Split("train", np.asarray(X_train), np.asarray(y_train))
        self.test = (
            _Split("test", np.asarray(X_test), np.asarray(y_test))
            if X_test is not None and y_test is not None
            else None
        )
        self.random_state = (
            int(random_state) if random_state is not None else int(model.random_state)
        )
        self.ica_n_components = ica_n_components
        self.ica_max_iter = int(ica_max_iter)
        self.n_plot_points = int(n_plot_points)

    def generate(self, output_path: str | Path) -> Path:
        """Render the report to ``output_path`` and return the resolved path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(
            [
                self._section_hyperparameters(),
                self._section_signal_plots(),
                self._section_metrics(),
                self._section_ica(),
            ]
        )
        html = _wrap_html(body)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _splits(self) -> list[_Split]:
        return [self.train] + ([self.test] if self.test is not None else [])

    def _section_hyperparameters(self) -> str:
        m = self.model
        attrs = {
            "eeg_channels": m.eeg_channels,
            "base_model": m.base_model,
            "params": m.params,
            "channel_handling": m.channel_handling,
            "use_electrode_feature": m.use_electrode_feature,
            "chain": m.chain,
            "conformal": m.conformal,
            "conformal_alpha": m.conformal_alpha,
            "calibration_fraction": m.calibration_fraction,
            "sfreq": m.sfreq,
            "random_state": m.random_state,
        }
        rows = "".join(
            f"<tr><th>{_html.escape(k)}</th>"
            f"<td>{_html.escape(repr(v))}</td></tr>"
            for k, v in attrs.items()
        )
        if m.conformal and getattr(m, "numerators_", None) is not None:
            rows += (
                "<tr><th>numerators_</th>"
                f"<td>{_html.escape(np.array2string(m.numerators_, precision=4))}</td></tr>"
            )
        return (
            '<section id="section-hyperparameters">'
            "<h2>1. Model hyperparameters</h2>"
            f"<table>{rows}</table>"
            "</section>"
        )

    def _section_signal_plots(self) -> str:
        rng = np.random.default_rng(self.random_state)
        n_channels = self.train.y.shape[0]
        channel = int(rng.integers(0, n_channels))
        channel_name = (
            self.model.eeg_channels[channel]
            if channel < len(self.model.eeg_channels)
            else f"ch{channel}"
        )

        body = []
        for split in self._splits():
            y_pred = self.model.predict(split.X)
            cleaned = self.model.clean(split.y, split.X)
            n_points = split.y.shape[1]
            n_show = min(n_points, self.n_plot_points)
            start = int(rng.integers(0, max(1, n_points - n_show + 1)))
            stop = start + n_show

            fig = _make_signal_figure(
                t=np.arange(start, stop),
                y_true=split.y[channel, start:stop],
                y_pred=y_pred[channel, start:stop],
                cleaned=cleaned[channel, start:stop],
                title=f"{split.name} - {channel_name} (samples {start}-{stop})",
            )
            img = _fig_to_base64(fig)
            plt.close(fig)
            body.append(
                f'<figure><img src="{img}" alt="{split.name} signals"/>'
                f"<figcaption>{split.name}</figcaption></figure>"
            )
        return (
            '<section id="section-signal-plots">'
            f"<h2>2. Signal plots - channel {_html.escape(channel_name)}</h2>"
            + "".join(body)
            + "</section>"
        )

    def _section_metrics(self) -> str:
        body = []
        for split in self._splits():
            y_pred = self.model.predict(split.X)
            pred = self.model.prediction_score(split.X, split.y)
            qual = self.model.score(split.X, split.y)
            corr = self.model.correlation_score(split.X)
            channels = self.model.eeg_channels

            agg = prediction_metrics(np.asarray(split.y).ravel(), np.asarray(y_pred).ravel())
            pred_table = _prediction_quality_table(channels, pred, qual, agg)
            corr_summary = _correlation_summary_table(channels, corr)
            corr_full = _correlation_full_tables(channels, corr)

            body.append(
                f"<h3>{_html.escape(split.name)}</h3>"
                "<h4>Prediction &amp; EEG quality (raw vs cleaned)</h4>"
                f"{pred_table}"
                "<h4>Mean correlation across IMU regressors</h4>"
                f"{corr_summary}"
                "<details><summary>Per-feature correlation matrices</summary>"
                f"{corr_full}</details>"
            )
        return (
            '<section id="section-metrics">'
            "<h2>3. Summary metrics</h2>"
            + "".join(body)
            + "</section>"
        )

    def _section_ica(self) -> str:
        split = self.test if self.test is not None else self.train
        cleaned = self.model.clean(split.y, split.X)

        raw_before = _make_raw(split.y, self.model.eeg_channels, self.model.sfreq)
        raw_after = _make_raw(cleaned, self.model.eeg_channels, self.model.sfreq)

        n_components = self.ica_n_components or len(self.model.eeg_channels)

        figs_before = _ica_figures(
            raw_before,
            n_components=n_components,
            max_iter=self.ica_max_iter,
            random_state=self.random_state,
            n_plot_points=self.n_plot_points,
        )
        figs_after = _ica_figures(
            raw_after,
            n_components=n_components,
            max_iter=self.ica_max_iter,
            random_state=self.random_state,
            n_plot_points=self.n_plot_points,
        )

        return (
            '<section id="section-ica">'
            f"<h2>4. ICA before vs after cleaning ({_html.escape(split.name)} data)</h2>"
            + _ica_block("Before cleaning", figs_before)
            + _ica_block("After cleaning", figs_after)
            + "</section>"
        )


def _wrap_html(body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>IMURegressor report</title>"
        + _STYLE
        + "</head><body><h1>IMURegressor report</h1>"
        + body
        + "</body></html>"
    )


def _fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _make_raw(y_2d: NDArray, channels: Iterable[str], sfreq: float) -> mne.io.RawArray:
    info = mne.create_info(ch_names=list(channels), sfreq=float(sfreq), ch_types="eeg")
    raw = mne.io.RawArray(np.asarray(y_2d, dtype=float), info, verbose=False)
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
    except Exception:
        pass
    return raw


def _make_signal_figure(
    *,
    t: NDArray,
    y_true: NDArray,
    y_pred: NDArray,
    cleaned: NDArray,
    title: str,
) -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    axes[0].plot(t, y_true - y_pred, color="C2")
    axes[0].axhline(0.0, color="k", lw=0.6, alpha=0.4)
    axes[0].set_title(f"{title}\nResidual (y - y_pred)")
    axes[0].set_ylabel("residual")

    axes[1].plot(t, y_true, label="True", alpha=0.85)
    axes[1].plot(t, y_pred, label="Predicted", alpha=0.85)
    axes[1].legend(loc="upper right")
    axes[1].set_title("True vs Predicted")
    axes[1].set_ylabel("value")

    axes[2].plot(t, cleaned, color="C3")
    axes[2].axhline(0.0, color="k", lw=0.6, alpha=0.4)
    axes[2].set_title("Cleaned (y - coef * y_pred)")
    axes[2].set_xlabel("time index")
    axes[2].set_ylabel("cleaned")
    return fig


def _prediction_quality_table(
    channels: list[str],
    pred: dict[str, NDArray],
    qual: dict[str, dict[str, NDArray]],
    agg: dict[str, float],
) -> str:
    head = (
        "<tr>"
        "<th>channel</th>"
        "<th>RMSE</th><th>MAE</th><th>R<sup>2</sup></th>"
        "<th>kurt raw</th><th>kurt clean</th><th>&Delta;kurt</th>"
        "<th>slope raw</th><th>slope clean</th><th>&Delta;slope</th>"
        "<th>HFP/TP raw</th><th>HFP/TP clean</th><th>&Delta;HFP/TP</th>"
        "</tr>"
    )
    rows: list[str] = []
    for c, name in enumerate(channels):
        rows.append(
            "<tr>"
            f"<td>{_html.escape(name)}</td>"
            f"<td>{pred['rmse'][c]:.4f}</td>"
            f"<td>{pred['mae'][c]:.4f}</td>"
            f"<td>{pred['r2'][c]:.4f}</td>"
            f"<td>{qual['raw']['kurtosis'][c]:.3f}</td>"
            f"<td>{qual['cleaned']['kurtosis'][c]:.3f}</td>"
            f"<td>{qual['delta']['kurtosis'][c]:+.3f}</td>"
            f"<td>{qual['raw']['spectral_slope'][c]:.3f}</td>"
            f"<td>{qual['cleaned']['spectral_slope'][c]:.3f}</td>"
            f"<td>{qual['delta']['spectral_slope'][c]:+.3f}</td>"
            f"<td>{qual['raw']['hfp_tp'][c]:.3f}</td>"
            f"<td>{qual['cleaned']['hfp_tp'][c]:.3f}</td>"
            f"<td>{qual['delta']['hfp_tp'][c]:+.3f}</td>"
            "</tr>"
        )
    rows.append(
        "<tr class='all-row'>"
        "<td>all (flattened)</td>"
        f"<td>{agg['rmse']:.4f}</td>"
        f"<td>{agg['mae']:.4f}</td>"
        f"<td>{agg['r2']:.4f}</td>"
        + "<td>-</td>" * 9
        + "</tr>"
    )
    return f"<table>{head}{''.join(rows)}</table>"


def _correlation_summary_table(channels: list[str], corr: dict[str, NDArray]) -> str:
    head = (
        "<tr><th>channel</th>"
        "<th>mean Pearson</th><th>mean coherence</th><th>mean MI</th></tr>"
    )
    rows = []
    for c, name in enumerate(channels):
        rows.append(
            "<tr>"
            f"<td>{_html.escape(name)}</td>"
            f"<td>{float(np.mean(corr['pearson'][c])):+.3f}</td>"
            f"<td>{float(np.mean(corr['coherence'][c])):.3f}</td>"
            f"<td>{float(np.mean(corr['mutual_information'][c])):.3f}</td>"
            "</tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _correlation_full_tables(channels: list[str], corr: dict[str, NDArray]) -> str:
    n_features = corr["pearson"].shape[1]
    parts: list[str] = []
    for metric_name in ("pearson", "coherence", "mutual_information"):
        mat = corr[metric_name]
        head = (
            "<tr><th>channel</th>"
            + "".join(f"<th>f{j}</th>" for j in range(n_features))
            + "</tr>"
        )
        body = []
        for c, name in enumerate(channels):
            body.append(
                "<tr>"
                f"<td>{_html.escape(name)}</td>"
                + "".join(f"<td>{mat[c, j]:+.3f}</td>" for j in range(n_features))
                + "</tr>"
            )
        parts.append(
            f"<h5>{_html.escape(metric_name)}</h5>"
            f"<table>{head}{''.join(body)}</table>"
        )
    return "".join(parts)


def _ica_figures(
    raw: mne.io.RawArray,
    *,
    n_components: int,
    max_iter: int,
    random_state: int,
    n_plot_points: int,
) -> dict[str, Figure]:
    n_components = max(1, min(n_components, len(raw.ch_names)))
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        method="fastica",
        verbose=False,
    )
    ica.fit(raw, verbose=False)

    topo = ica.plot_components(picks=range(n_components), show=False)
    if isinstance(topo, list):
        topo_fig = topo[0] if topo else plt.figure()
    else:
        topo_fig = topo

    sources = ica.get_sources(raw).get_data()
    n_show = min(sources.shape[1], n_plot_points)
    sources = sources[:, :n_show]

    src_fig, ax = plt.subplots(figsize=(10, max(3.0, 0.6 * n_components)))
    spread = float(np.nanmax(np.abs(sources))) if sources.size else 1.0
    spacing = max(spread * 4.0, 1e-6)
    offsets = np.arange(n_components) * spacing
    for i in range(n_components):
        ax.plot(sources[i] + offsets[i], lw=0.7)
    ax.set_yticks(offsets)
    ax.set_yticklabels([f"IC{i}" for i in range(n_components)])
    ax.set_xlabel("time index")
    ax.set_title(f"ICA sources (first {n_show} samples)")
    src_fig.tight_layout()
    return {"topomaps": topo_fig, "sources": src_fig}


def _ica_block(label: str, figs: dict[str, Figure]) -> str:
    parts = [f"<h3>{_html.escape(label)}</h3>"]
    for kind, fig in figs.items():
        img = _fig_to_base64(fig)
        plt.close(fig)
        parts.append(
            f'<figure><img src="{img}" alt="ICA {kind} {label}"/>'
            f"<figcaption>{kind}</figcaption></figure>"
        )
    return "".join(parts)
