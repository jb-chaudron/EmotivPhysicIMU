import numpy as np
import mne

# –– Physics Informed Artefacts ––––––––––––––––––
def compute_electrode_motion_features(
    coords: np.ndarray,
    quaternions: np.ndarray,
    accelerations: np.ndarray,
    dt: float = 1 / 32.0,
) -> np.ndarray:
    """
    Per-electrode motion features from IMU data.

    Parameters
    ----------
    coords : (N, 3)
        Electrode positions in 3D head-centered space (meters).
    quaternions : (T, 4)
        Orientation quaternions [w, x, y, z].
    accelerations : (T, 3)
        Linear accelerations [ax, ay, az] (m/s²).
    dt : float
        Sampling interval in seconds (default 1/128 s).

    Returns
    -------
    features : (N, T, 4)
        Per-electrode, per-timestep features:
            [0] f_n      — normal acceleration (pressure on skin-electrode)
            [1] f_t      — tangential acceleration (shear)
            [2] f_omega  — rotational friction magnitude (||ω × r_e||)
            [3] jerk_e   — time derivative of rotational friction
    """
    N = coords.shape[0]
    T = quaternions.shape[0]

    # Unit normal vectors (outward from sphere center to electrode)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    n_hat = coords / np.where(norms > 0, norms, 1.0)  # (N, 3)

    # --- Angular velocity from quaternion derivative ---
    # ω = 2 * q_conj ⊗ dq/dt  →  take vector part
    dq = np.gradient(quaternions, dt, axis=0)  # (T, 4), central diff
    q_conj = quaternions * np.array([1.0, -1.0, -1.0, -1.0])  # (T, 4)

    qw, qx, qy, qz = q_conj[:, 0], q_conj[:, 1], q_conj[:, 2], q_conj[:, 3]
    dw, dx, dy, dz = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]

    # Quaternion product vector part (q_conj ⊗ dq)[1:4]
    ox = qw*dx + qx*dw + qy*dz - qz*dy
    oy = qw*dy - qx*dz + qy*dw + qz*dx
    oz = qw*dz + qx*dy - qy*dx + qz*dw
    omega = 2.0 * np.stack([ox, oy, oz], axis=1)  # (T, 3)

    # --- f_n : normal component of acceleration per electrode ---
    # a @ n_hat.T  →  (T, N)
    f_n = (accelerations @ n_hat.T).T  # (N, T)

    # --- f_t : tangential (shear) component ---
    # a_normal = f_n * n_hat per timestep  →  (T, N, 3)
    a_normal = f_n.T[:, :, np.newaxis] * n_hat[np.newaxis, :, :]
    a_tang = accelerations[:, np.newaxis, :] - a_normal          # (T, N, 3)
    f_t = np.linalg.norm(a_tang, axis=2).T                       # (N, T)

    # --- f_omega : rotational friction  ||ω × r_e|| ---
    # np.cross(omega, coords)  →  (T, N, 3)
    omega_cross_r = np.cross(
        omega[:, np.newaxis, :],    # (T, 1, 3)
        coords[np.newaxis, :, :],   # (1, N, 3)
    )                               # (T, N, 3)
    f_omega = np.linalg.norm(omega_cross_r, axis=2).T  # (N, T)

    # --- jerk_e : d/dt(f_omega) ---
    jerk_e = np.gradient(f_omega, dt, axis=1)  # (N, T)
    return np.stack([f_n, f_t, f_omega, jerk_e], axis=2)  # (N, T, 4)

def compute_Lorentzian_features(
    coords: np.ndarray,
    quats: np.ndarray,
    accels: np.ndarray,
    B: np.ndarray,
    dt: float = 1 / 32.0,
) -> np.ndarray:
    """
    Lorentzian features from IMU data.
    """
    # v_IMU : (T, 3) — vitesse translation depuis cumsum accélero
    v_imu = np.cumsum(accels * dt, axis=0)
    v_imu -= v_imu.mean(axis=0)  # dérive DC

    # omega : (T, 3) — déjà calculé depuis quaternion
    dq = np.gradient(quats, dt, axis=0)  # (T, 4), central diff
    q_conj = quats * np.array([1.0, -1.0, -1.0, -1.0])  # (T, 4)

    qw, qx, qy, qz = q_conj[:, 0], q_conj[:, 1], q_conj[:, 2], q_conj[:, 3]
    dw, dx, dy, dz = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]

    # Quaternion product vector part (q_conj ⊗ dq)[1:4]
    ox = qw*dx + qx*dw + qy*dz - qz*dy
    oy = qw*dy - qx*dz + qy*dw + qz*dx
    oz = qw*dz + qx*dy - qy*dx + qz*dw
    omega = 2.0 * np.stack([ox, oy, oz], axis=1)  # (T, 3)
    # coords : (N, 3), n_hat : (N, 3), B : (T, 3)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    n_hat = coords / np.where(norms > 0, norms, 1.0)  # (N, 3)
    # Terme 1 : translation, même pour tous les canaux modulé par n_hat
    # v_imu × B → (T, 3)
    vxB_trans = np.cross(v_imu, B)           # (T, 3)
    f_trans = (vxB_trans @ n_hat.T).T        # (N, T)

    # Terme 2 : rotation, différent par électrode
    # (omega × r_e) × B → per electrode
    omega_cross_r = np.cross(
        omega[:, np.newaxis, :],             # (T, 1, 3)
        coords[np.newaxis, :, :]             # (1, N, 3)
    )                                        # (T, N, 3)

    # Pour chaque électrode : (omega × r_e) × B
    # omega_cross_r : (T, N, 3), B : (T, 3)
    rot_cross_B = np.cross(
        omega_cross_r,                       # (T, N, 3)
        B[:, np.newaxis, :]                  # (T, 1, 3)
    )                                        # (T, N, 3)

    # Projection sur n_hat_e
    f_rot = np.einsum('tne,ne->nt', rot_cross_B, n_hat)  # (N, T)

    # Feature Lorentz complet par canal : 2 scalaires
    features_lorentz = np.stack([f_trans, f_rot], axis=2)  # (N, T, 2)

    return features_lorentz


# ── Helper : get Emotiv electrode coords from MNE ────────────────────────────
def get_emotiv_coords(ch_names=None):
    """
    Returns (N, 3) electrode positions (meters, head-centered) for Emotiv EPOC+.
    Uses MNE standard_1020 montage, filtered to Emotiv channels.
    """
    try:
        import mne
        emotiv_channels = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4',
        ]
        if ch_names is not None:
            emotiv_channels = [c for c in emotiv_channels if c in ch_names]
        montage = mne.channels.make_standard_montage('standard_1020')
        pos_dict = montage.get_positions()['ch_pos']
        coords = np.array([pos_dict[ch] for ch in emotiv_channels if ch in pos_dict])
        return coords, [ch for ch in emotiv_channels if ch in pos_dict]
    except ImportError:
        raise ImportError("MNE-Python required: pip install mne")