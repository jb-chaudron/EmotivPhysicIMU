from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_electrode_motion_features(
    coords: NDArray,        # (n_elec, 3)
    quaternions: NDArray,   # (n_samples, 4)
    accelerations: NDArray, # (n_samples, 3), translational acceleration in head frame
    dt: float = 1 / 32.0,
) -> NDArray:
    """
    Compute per-electrode motion features from rigid-body kinematics.

    Returns an array of shape (n_elec, n_samples, n_features) with the features:
        0. pressure_trans   = a_trans · n_hat
        1. pressure_alpha   = (alpha x r) · n_hat
        2. pressure_omega2  = (omega x (omega x r)) · n_hat
        3. pressure_total   = a_e · n_hat
        4. shear_trans      = ||a_trans - proj_n(a_trans)||
        5. shear_alpha      = ||alpha x r - proj_n(alpha x r)||
        6. shear_omega2     = ||omega x (omega x r) - proj_n(omega x (omega x r))||
        7. shear_total      = ||a_e - proj_n(a_e)||
        8. speed            = ||omega x r||
        9. jerk_speed       = d(speed) / dt
    """
    coords = np.asarray(coords, dtype=float)
    quaternions = np.asarray(quaternions, dtype=float)
    accelerations = np.asarray(accelerations, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_elec, 3)")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (n_samples, 4)")
    if accelerations.ndim != 2 or accelerations.shape[1] != 3:
        raise ValueError("accelerations must have shape (n_samples, 3)")
    if quaternions.shape[0] != accelerations.shape[0]:
        raise ValueError("quaternions and accelerations must have the same number of samples")

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    n_hat = coords / np.where(norms > 0, norms, 1.0)          # (n_elec, 3)
    n_hat_ = n_hat[np.newaxis, :, :]                          # (1, n_elec, 3)

    omega = _angular_velocity(quaternions, dt=dt)             # (n_samples, 3)
    alpha = np.gradient(omega, dt, axis=0)                    # (n_samples, 3)

    r = coords[np.newaxis, :, :]                              # (1, n_elec, 3)
    a_trans = accelerations[:, np.newaxis, :]                 # (n_samples, 1, 3)

    alpha_cross_r = np.cross(alpha[:, np.newaxis, :], r)      # (n_samples, n_elec, 3)
    omega_cross_r = np.cross(omega[:, np.newaxis, :], r)      # (n_samples, n_elec, 3)
    omega_cross_omega_cross_r = np.cross(
        omega[:, np.newaxis, :], omega_cross_r
    )                                                         # (n_samples, n_elec, 3)

    a_e = a_trans + alpha_cross_r + omega_cross_omega_cross_r

    pressure_trans = np.sum(a_trans * n_hat_, axis=2).T
    pressure_alpha = np.sum(alpha_cross_r * n_hat_, axis=2).T
    pressure_omega2 = np.sum(omega_cross_omega_cross_r * n_hat_, axis=2).T
    pressure_total = np.sum(a_e * n_hat_, axis=2).T

    a_trans_normal = pressure_trans.T[:, :, np.newaxis] * n_hat_
    alpha_normal = pressure_alpha.T[:, :, np.newaxis] * n_hat_
    omega2_normal = pressure_omega2.T[:, :, np.newaxis] * n_hat_
    a_total_normal = pressure_total.T[:, :, np.newaxis] * n_hat_

    shear_trans_vec = a_trans - a_trans_normal
    shear_alpha_vec = alpha_cross_r - alpha_normal
    shear_omega2_vec = omega_cross_omega_cross_r - omega2_normal
    shear_total_vec = a_e - a_total_normal

    shear_trans = np.linalg.norm(shear_trans_vec, axis=2).T
    shear_alpha = np.linalg.norm(shear_alpha_vec, axis=2).T
    shear_omega2 = np.linalg.norm(shear_omega2_vec, axis=2).T
    shear_total = np.linalg.norm(shear_total_vec, axis=2).T

    speed = np.linalg.norm(omega_cross_r, axis=2).T
    jerk_speed = np.gradient(speed, dt, axis=1)

    return np.stack(
        [
            pressure_trans,
            pressure_alpha,
            pressure_omega2,
            pressure_total,
            shear_trans,
            shear_alpha,
            shear_omega2,
            shear_total,
            speed,
            jerk_speed,
        ],
        axis=2,
    )


def compute_lorentzian_features(
    coords: NDArray,
    quats: NDArray,
    accels: NDArray,
    magnetic_field: NDArray,
    dt: float = 1 / 32.0,
) -> NDArray:
    """Lorentz-inspired per-electrode features from IMU velocity, rotation and B field."""
    v_imu = np.cumsum(accels * dt, axis=0)
    v_imu -= v_imu.mean(axis=0)
    omega = _angular_velocity(quats, dt=dt)

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    n_hat = coords / np.where(norms > 0, norms, 1.0)

    vx_b_trans = np.cross(v_imu, magnetic_field)
    f_trans = (vx_b_trans @ n_hat.T).T

    omega_cross_r = np.cross(omega[:, np.newaxis, :], coords[np.newaxis, :, :])
    rot_cross_b = np.cross(omega_cross_r, magnetic_field[:, np.newaxis, :])
    f_rot = np.einsum("tne,ne->nt", rot_cross_b, n_hat)

    return np.stack([f_trans, f_rot], axis=2)


def get_emotiv_coords(ch_names: list[str] | None = None) -> tuple[NDArray, list[str]]:
    """Return Emotiv electrode positions from MNE's standard 10-20 montage."""
    try:
        import mne
    except ImportError as exc:
        raise ImportError("MNE-Python required: pip install mne") from exc

    emotiv_channels = [
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
    ]
    if ch_names is not None:
        emotiv_channels = [channel for channel in emotiv_channels if channel in ch_names]

    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]
    coords = np.array([pos_dict[channel] for channel in emotiv_channels if channel in pos_dict])
    coord_channels = [channel for channel in emotiv_channels if channel in pos_dict]
    return coords, coord_channels


def _angular_velocity(quaternions: NDArray, dt: float) -> NDArray:
    dq = np.gradient(quaternions, dt, axis=0)
    q_conj = quaternions * np.array([1.0, -1.0, -1.0, -1.0])

    qw, qx, qy, qz = q_conj[:, 0], q_conj[:, 1], q_conj[:, 2], q_conj[:, 3]
    dw, dx, dy, dz = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]

    ox = qw * dx + qx * dw + qy * dz - qz * dy
    oy = qw * dy - qx * dz + qy * dw + qz * dx
    oz = qw * dz + qx * dy - qy * dx + qz * dw
    return 2.0 * np.stack([ox, oy, oz], axis=1)


# Backward-compatible spelling for existing callers.
compute_Lorentzian_features = compute_lorentzian_features
