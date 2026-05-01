from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from .constants import EMOTIV_CHANNELS, IMU_CHANNELS, IMU_CHANNELS_UNIQUE
from .physics import (
    compute_electrode_motion_features,
    compute_lorentzian_features,
    get_emotiv_coords,
)

LORENTZ_FEATURE_NAMES = ["lorentz_trans", "lorentz_rot"]
MOTION_FEATURE_NAMES = [
    "pressure_trans",
    "pressure_alpha",
    "pressure_omega2",
    "pressure_total",
    "shear_trans",
    "shear_alpha",
    "shear_omega2",
    "shear_total",
    "speed",
    "jerk_speed",
]


@dataclass(frozen=True)
class FeatureSet:
    """Channels-first feature container.

    X has shape (n_channels, n_points, n_features * n_windows). For each EEG step
    `t`, the last axis stacks IMU features at the previous `n_windows` steps
    (oldest first). y has shape (n_channels, n_points) with `n_points = n_raw - n_windows`.
    """

    X: NDArray
    y: NDArray
    feature_blocks: dict[str, slice]
    eeg_channels: list[str]
    feature_names: list[str]
    sfreq: float

    def __iter__(self):
        # Backwards-compatible unpacking: X, y = extract_features(...)
        yield self.X
        yield self.y


def extract_features(
    raw_eeg,
    raw_imu=None,
    *,
    eeg_channels: Sequence[str] = EMOTIV_CHANNELS,
    imu_channels: Sequence[str] = IMU_CHANNELS,
    n_windows: int = 1,
    target_ratio: int | None = None,
    physics: bool = False,
) -> FeatureSet:
    """Extract IMU-derived features paired with averaged EEG targets.

    Parameters
    ----------
    raw_eeg : mne.io.Raw
        Raw with EEG channels. If `raw_imu` is None, IMU channels must also be
        present in `raw_eeg`.
    raw_imu : mne.io.Raw or None
        Raw with IMU channels. If None, IMU channels are read from `raw_eeg`.
    n_windows : int
        Number of past IMU steps stacked as features for the current EEG step
        (>= 1). The first `n_windows` EEG steps are dropped to align with the
        windowed features.
    target_ratio : int or None
        EEG samples averaged per IMU sample. If None (default), derived from
        `raw_eeg.n_times // imu_source.n_times`. For a single-Raw input, this
        evaluates to 1 (no averaging) since EEG and IMU share a sample rate.
    physics : bool
        If True, build physics-informed per-electrode features. Otherwise stack
        raw IMU values + yaw/pitch/roll, tiled across channels.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")

    eeg_channels = list(eeg_channels)
    imu_source = raw_imu if raw_imu is not None else raw_eeg

    _validate_channels(raw_eeg, eeg_channels, raw_name="raw_eeg")
    _validate_channels(imu_source, imu_channels, raw_name="raw_imu")

    if target_ratio is None:
        eeg_n = int(raw_eeg.n_times)
        imu_n = int(imu_source.n_times)
        if imu_n == 0 or eeg_n % imu_n != 0:
            raise ValueError(
                "Cannot auto-derive target_ratio: raw_eeg.n_times="
                f"{eeg_n} is not a positive multiple of imu_source.n_times={imu_n}. "
                "Pass target_ratio explicitly."
            )
        target_ratio = eeg_n // imu_n
    if target_ratio <= 0:
        raise ValueError("target_ratio must be > 0")

    if physics:
        feats_cnf, feature_blocks, feature_names = _physics_features(imu_source, eeg_channels, imu_channels)
    else:
        feats_cnf, feature_blocks, feature_names = _movement_features(imu_source, len(eeg_channels), imu_channels)

    X = _stack_past_windows(feats_cnf, n_windows)
    feature_blocks = {name: slice(sl.start * n_windows, sl.stop * n_windows) for name, sl in feature_blocks.items()}
    feature_names = _expand_names_for_windows(feature_names, n_windows)

    y = _extract_target(raw_eeg, eeg_channels, target_ratio, n_windows)
    if X.shape[1] != y.shape[1]:
        raise ValueError(f"X / y time mismatch: X {X.shape[1]} vs y {y.shape[1]}")

    sfreq = float(raw_eeg.info["sfreq"]) / target_ratio
    return FeatureSet(
        X=X,
        y=y,
        feature_blocks=feature_blocks,
        eeg_channels=eeg_channels,
        feature_names=feature_names,
        sfreq=sfreq,
    )


def _stack_past_windows(feats_cnf: NDArray, n_windows: int) -> NDArray:
    """(C, N_raw, F) -> (C, N_raw - n_windows, F * n_windows). Past-only window.

    For step `t`, features are taken from indices `[t - n_windows, t - 1]`, ordered
    oldest first inside the stacked vector. Valid `t` ranges over `[n_windows, N_raw)`.
    """
    if feats_cnf.ndim != 3:
        raise ValueError(f"Expected 3D features (C, N, F), got {feats_cnf.shape}")
    n_channels, n_raw, n_features = feats_cnf.shape
    if n_raw <= n_windows:
        raise ValueError(f"Need n_raw > n_windows, got {n_raw} and {n_windows}")

    windowed = sliding_window_view(feats_cnf, window_shape=n_windows, axis=1)
    # windowed: (C, n_raw - n_windows + 1, F, n_windows). The window starting at i
    # covers samples [i, i + n_windows - 1] and pairs with EEG step i + n_windows.
    # Drop the last window since there is no EEG step at index n_raw.
    windowed = windowed[:, : n_raw - n_windows, :, :]
    n_points = windowed.shape[1]
    return windowed.reshape(n_channels, n_points, n_features * n_windows)


def _expand_names_for_windows(names: list[str], n_windows: int) -> list[str]:
    expanded: list[str] = []
    for name in names:
        for w in range(n_windows):
            lag = n_windows - w
            expanded.append(f"{name}@t-{lag}")
    return expanded


def _extract_target(
    raw_eeg,
    eeg_channels: Sequence[str],
    target_ratio: int,
    n_windows: int,
) -> NDArray:
    target = raw_eeg.get_data(picks=list(eeg_channels))
    n_channels, n_eeg_raw = target.shape
    if n_eeg_raw % target_ratio != 0:
        raise ValueError(
            f"raw_eeg has {n_eeg_raw} samples, which is not divisible by target_ratio={target_ratio}"
        )
    target = target.reshape(n_channels, -1, target_ratio).mean(axis=2)
    if target.shape[1] <= n_windows:
        raise ValueError(
            f"Not enough target samples after averaging: {target.shape[1]} <= n_windows={n_windows}"
        )
    return target[:, n_windows:]


def _movement_features(
    raw_imu,
    n_channels: int,
    channels: list[str] = IMU_CHANNELS,
) -> tuple[NDArray, dict[str, slice], list[str]]:
    motion = raw_imu.get_data(picks=channels)
    yaw, pitch, roll = _quat_to_ypr(motion[0], motion[1], motion[2], motion[3])
    feats = np.vstack([yaw[None, :], pitch[None, :], roll[None, :], motion])
    n_features, n_samples = feats.shape

    feats_cnf = np.broadcast_to(
        feats.T[np.newaxis, :, :],
        (n_channels, n_samples, n_features),
    ).copy()

    feature_names = ["yaw", "pitch", "roll"] + list(channels)
    feature_blocks = {"motion_history": slice(0, n_features)}
    return feats_cnf, feature_blocks, feature_names


def _physics_features(
    raw_imu,
    eeg_channels: Sequence[str],
    channels: list[str] = IMU_CHANNELS,
) -> tuple[NDArray, dict[str, slice], list[str]]:
    coords, coord_channels = get_emotiv_coords(list(eeg_channels))
    if coord_channels != list(eeg_channels):
        raise ValueError(
            f"Only found coordinates for {coord_channels}, expected {list(eeg_channels)}"
        )
    if "Q0" in channels:
        quats = raw_imu.get_data(picks=["Q0", "Q1", "Q2", "Q3"], units="uV").T
        accels = raw_imu.get_data(picks=["ACCX", "ACCY", "ACCZ"], units="uV").T
        magnetic = raw_imu.get_data(picks=["MAGX", "MAGY", "MAGZ"], units="uV").T
    else:
        quats = raw_imu.get_data(picks=["MOT.Q0", "MOT.Q1", "MOT.Q2", "MOT.Q3"], units="uV").T
        accels = raw_imu.get_data(picks=["MOT.AccX", "MOT.AccY", "MOT.AccZ"], units="uV").T
        magnetic = raw_imu.get_data(picks=["MOT.MagX", "MOT.MagY", "MOT.MagZ"], units="uV").T

    lorentz = compute_lorentzian_features(coords, quats, accels, magnetic)
    motion = compute_electrode_motion_features(coords, quats, accels)

    feats_cnf = np.concatenate([lorentz, motion], axis=2)
    feature_names = list(LORENTZ_FEATURE_NAMES) + list(MOTION_FEATURE_NAMES)
    feature_blocks = {
        "lorentz": slice(0, len(LORENTZ_FEATURE_NAMES)),
        "motion": slice(len(LORENTZ_FEATURE_NAMES), len(feature_names)),
    }
    return feats_cnf, feature_blocks, feature_names


def _quat_to_ypr(q0: NDArray, q1: NDArray, q2: NDArray, q3: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    norm = np.where(norm > 0, norm, 1.0)
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return yaw, pitch, roll


def _validate_channels(raw, channels: Sequence[str], *, raw_name: str) -> None:
    missing = [channel for channel in channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{raw_name} is missing channels: {missing}")
