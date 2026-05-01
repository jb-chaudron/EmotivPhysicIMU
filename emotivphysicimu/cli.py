from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import mne
import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .constants import EMOTIV_CHANNELS
from .evaluation import plot_predictions
from .features import FeatureSet, extract_features
from .model import IMURegressor


class ChainStage(BaseModel):
    features: str
    model: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    base_model: str = "linear_regression"
    channel_handling: str = "per_channel"
    use_electrode_feature: bool = False
    conformal: bool = False
    conformal_alpha: float = 0.1
    calibration_fraction: float = 0.2
    random_state: int = 42
    params: dict[str, Any] = Field(default_factory=dict)
    chain: list[ChainStage] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    experiment_name: str
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, str]
    model: ModelConfig = Field(default_factory=ModelConfig)
    report: dict[str, str] = Field(default_factory=dict)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Emotiv IMU cleaning pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    args = parser.parse_args(argv)

    config = PipelineConfig.model_validate(yaml.safe_load(args.config.read_text()))
    X, y, feature_blocks, sfreq = load_feature_matrix(config)

    chain_cfg = [stage.model_dump() for stage in config.model.chain] or None
    regressor = IMURegressor(
        eeg_channels=EMOTIV_CHANNELS,
        feature_blocks=feature_blocks,
        base_model=config.model.base_model,
        params=config.model.params,
        channel_handling=config.model.channel_handling,
        use_electrode_feature=config.model.use_electrode_feature,
        chain=chain_cfg,
        conformal=config.model.conformal,
        conformal_alpha=config.model.conformal_alpha,
        calibration_fraction=config.model.calibration_fraction,
        sfreq=sfreq,
        random_state=config.model.random_state,
    )
    regressor.fit(X, y)

    pred_metrics = regressor.prediction_score(X, y)
    print(
        f"Prediction (mean over channels) - "
        f"RMSE: {pred_metrics['rmse'].mean():.4f} | "
        f"MAE: {pred_metrics['mae'].mean():.4f} | "
        f"R2: {pred_metrics['r2'].mean():.4f}"
    )
    eeg_metrics = regressor.score(X, y)
    print(
        f"EEG quality (cleaned, mean) - "
        f"kurtosis: {eeg_metrics['kurtosis'].mean():.4f} | "
        f"slope: {eeg_metrics['spectral_slope'].mean():.4f} | "
        f"HFP/TP: {eeg_metrics['hfp_tp'].mean():.4f}"
    )

    y_pred = regressor.predict(X)
    plot_predictions(y.T, y_pred.T, channel_names=EMOTIV_CHANNELS)


def load_feature_matrix(
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, slice], float]:
    """Stack per-subject features along the time axis to feed IMURegressor.

    Returns
    -------
    X : (n_channels, sum_of_n_points, n_features * n_windows)
    y : (n_channels, sum_of_n_points)
    feature_blocks : dict[str, slice]
    sfreq : float
    """
    eeg_path = _env_path(config.data["eeg_path"])
    artefact_path = _env_path(config.data["artefact_path"])

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    feature_blocks: dict[str, slice] | None = None
    sfreq: float | None = None

    eeg_paths = sorted(path for path in eeg_path.iterdir() if path.suffix == ".edf")
    for raw_path in eeg_paths:
        name = raw_path.name.removesuffix(".edf")
        motion_path = artefact_path / f"{name}.md.edf"
        features = extract_features(
            raw_eeg=mne.io.read_raw_edf(str(raw_path)),
            raw_imu=mne.io.read_raw_edf(str(motion_path)),
            eeg_channels=EMOTIV_CHANNELS,
            n_windows=config.preprocessing.get("n_windows", 1),
            target_ratio=config.preprocessing.get("target_ratio", 4),
            physics=config.preprocessing.get("physics_informed_artefacts", False),
        )
        feature_blocks = _consistent_feature_blocks(feature_blocks, features, raw_path)
        sfreq = sfreq if sfreq is not None else features.sfreq
        X_parts.append(features.X)
        y_parts.append(features.y)

    if not X_parts:
        raise ValueError(f"No EDF files found in {eeg_path}")
    return (
        np.concatenate(X_parts, axis=1),
        np.concatenate(y_parts, axis=1),
        feature_blocks or {},
        float(sfreq) if sfreq is not None else 0.0,
    )


def _consistent_feature_blocks(
    current: dict[str, slice] | None,
    features: FeatureSet,
    raw_path: Path,
) -> dict[str, slice]:
    if current is None:
        return features.feature_blocks
    if not _same_feature_blocks(current, features.feature_blocks):
        raise ValueError(
            f"Feature blocks changed for {raw_path}: {features.feature_blocks} != {current}"
        )
    return current


def _same_feature_blocks(left: dict[str, slice], right: dict[str, slice]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        (left[key].start, left[key].stop, left[key].step)
        == (right[key].start, right[key].stop, right[key].step)
        for key in left
    )


def _env_path(name: str) -> Path:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name!r} is not set")
    return Path(value)


if __name__ == "__main__":
    main()
