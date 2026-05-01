import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from emotivphysicimu import IMURegressor, IMUReport, extract_features, plot_predictions
from emotivphysicimu.constants import EMOTIV_CHANNELS, IMU_CHANNELS
from emotivphysicimu.model import NUMERATORS


def _raw(data, names, sfreq=128):
    info = mne.create_info(ch_names=names, sfreq=sfreq, ch_types="misc")
    return mne.io.RawArray(np.asarray(data, dtype=float), info, verbose=False)


def _toy(n_channels=2, n_points=60, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_channels, n_points, n_features))
    weights = rng.normal(size=(n_channels, n_features))
    y = np.einsum("cnf,cf->cn", X, weights) + 0.05 * rng.normal(size=(n_channels, n_points))
    return X, y


def test_extract_features_channels_first_past_only_window():
    imu = np.zeros((len(IMU_CHANNELS), 6))
    imu[0] = 1.0
    raw_imu = _raw(imu, IMU_CHANNELS, sfreq=32)
    raw_eeg = _raw(np.arange(24).reshape(1, -1), ["AF3"], sfreq=128)

    features = extract_features(
        raw_eeg,
        raw_imu,
        eeg_channels=["AF3"],
        n_windows=3,
        target_ratio=4,
    )

    n_features = 3 + len(IMU_CHANNELS)
    assert features.X.shape == (1, 6 - 3, n_features * 3)
    assert features.y.shape == (1, 6 - 3)
    np.testing.assert_allclose(features.y.ravel(), [13.5, 17.5, 21.5])
    assert features.feature_blocks == {"motion_history": slice(0, n_features * 3)}
    assert features.sfreq == pytest.approx(32.0)


def test_extract_features_accepts_single_raw_with_imu():
    n_samples = 8
    eeg_channel = "AF3"
    combined_names = [eeg_channel] + list(IMU_CHANNELS)
    combined_data = np.zeros((len(combined_names), n_samples), dtype=float)
    combined_data[0] = np.arange(n_samples, dtype=float)
    combined_data[1] = 1.0

    raw = _raw(combined_data, combined_names, sfreq=32)
    features = extract_features(
        raw,
        eeg_channels=[eeg_channel],
        n_windows=2,
        target_ratio=1,
    )
    assert features.X.shape[0] == 1
    assert features.X.shape[1] == features.y.shape[1] == n_samples - 2


def test_imuregressor_per_channel_predicts_shape():
    X, y = _toy()
    regressor = IMURegressor(
        eeg_channels=["AF3", "F7"],
        sfreq=32.0,
    ).fit(X, y)
    y_pred = regressor.predict(X)
    assert y_pred.shape == y.shape


def test_imuregressor_all_channels_predicts_shape():
    X, y = _toy(n_channels=3)
    regressor = IMURegressor(
        eeg_channels=["AF3", "F7", "F3"],
        channel_handling="all_channels",
        sfreq=32.0,
    ).fit(X, y)
    assert regressor.predict(X).shape == y.shape


def test_imuregressor_residual_chain_three_stages_predicts_shape():
    rng = np.random.default_rng(0)
    n_channels, n_points = 2, 60
    motion = rng.normal(size=(n_channels, n_points, 3))
    lorentz = rng.normal(size=(n_channels, n_points, 2))
    raw_block = rng.normal(size=(n_channels, n_points, 4))
    X = np.concatenate([motion, lorentz, raw_block], axis=2)
    y = (
        0.5 * motion[:, :, 0]
        + 0.3 * lorentz[:, :, 0]
        + 0.1 * raw_block[:, :, 0]
        + 0.05 * rng.normal(size=(n_channels, n_points))
    )
    feature_blocks = {
        "motion": slice(0, 3),
        "lorentz": slice(3, 5),
        "raw": slice(5, 9),
    }
    regressor = IMURegressor(
        eeg_channels=["AF3", "F7"],
        feature_blocks=feature_blocks,
        chain=[
            {"features": "motion", "model": {"base_model": "linear_regression"}},
            {"features": "lorentz", "model": {"base_model": "linear_regression"}},
            {"features": "raw", "model": {"base_model": "linear_regression"}},
        ],
        sfreq=32.0,
    ).fit(X, y)
    assert regressor.predict(X).shape == y.shape


def test_imuregressor_score_returns_raw_cleaned_and_delta():
    X, y = _toy(n_channels=2, n_points=400)
    regressor = IMURegressor(eeg_channels=["AF3", "F7"], sfreq=128.0).fit(X, y)
    score = regressor.score(X, y)
    assert set(score) == {"raw", "cleaned", "delta"}
    for section in score.values():
        assert set(section) == {"kurtosis", "spectral_slope", "hfp_tp"}
        for arr in section.values():
            assert arr.shape == (2,)
    for k in ("kurtosis", "spectral_slope", "hfp_tp"):
        np.testing.assert_allclose(
            score["delta"][k], score["cleaned"][k] - score["raw"][k]
        )


def test_imuregressor_prediction_score_keys():
    X, y = _toy()
    regressor = IMURegressor(eeg_channels=["AF3", "F7"], sfreq=32.0).fit(X, y)
    metrics = regressor.prediction_score(X, y)
    assert set(metrics) == {"rmse", "mae", "r2"}
    for arr in metrics.values():
        assert arr.shape == (2,)


def test_imuregressor_correlation_score_returns_pearson_coherence_mi():
    X, y = _toy(n_channels=2, n_points=200, n_features=3)
    regressor = IMURegressor(eeg_channels=["AF3", "F7"], sfreq=128.0).fit(X, y)
    out = regressor.correlation_score(X)
    assert set(out) == {"pearson", "coherence", "mutual_information"}
    for arr in out.values():
        assert arr.shape == (2, 3)


def test_imuregressor_conformal_picks_numerator_per_channel():
    rng = np.random.default_rng(0)
    n_channels, n_points, n_features = 2, 600, 3
    X = rng.normal(size=(n_channels, n_points, n_features))
    weights = rng.normal(size=(n_channels, n_features))
    y = np.einsum("cnf,cf->cn", X, weights) + 0.1 * rng.normal(size=(n_channels, n_points))

    regressor = IMURegressor(
        eeg_channels=["AF3", "F7"],
        sfreq=128.0,
        conformal=True,
        conformal_alpha=0.1,
    ).fit(X, y)

    assert regressor.numerators_ is not None
    assert regressor.numerators_.shape == (n_channels,)
    assert all(num in NUMERATORS for num in regressor.numerators_.tolist())

    cleaned = regressor.clean(y, X)
    assert cleaned.shape == y.shape


def test_imuregressor_conformal_with_chain_raises():
    feature_blocks = {"all": slice(0, 4)}
    with pytest.raises(NotImplementedError):
        IMURegressor(
            eeg_channels=["AF3", "F7"],
            feature_blocks=feature_blocks,
            chain=[{"features": "all", "model": {"base_model": "linear_regression"}}],
            conformal=True,
            sfreq=128.0,
        )


def test_imuregressor_use_electrode_feature_requires_all_channels():
    with pytest.raises(ValueError, match="all_channels"):
        IMURegressor(
            eeg_channels=["AF3", "F7"],
            base_model="catboost",
            channel_handling="per_channel",
            use_electrode_feature=True,
        )


def test_imureport_generates_html(tmp_path):
    rng = np.random.default_rng(0)
    n_channels = len(EMOTIV_CHANNELS)
    n_points, n_features = 500, 4
    X = rng.normal(size=(n_channels, n_points, n_features))
    weights = rng.normal(size=(n_channels, n_features))
    y = (
        np.einsum("cnf,cf->cn", X, weights)
        + 0.1 * rng.normal(size=(n_channels, n_points))
    )

    regressor = IMURegressor(eeg_channels=EMOTIV_CHANNELS, sfreq=128.0).fit(X, y)
    report = IMUReport(
        regressor,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        ica_n_components=4,
        ica_max_iter=100,
        n_plot_points=200,
    )
    out = report.generate(tmp_path / "report.html")

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert len(text) > 1000
    for anchor in (
        'id="section-hyperparameters"',
        'id="section-signal-plots"',
        'id="section-metrics"',
        'id="section-ica"',
    ):
        assert anchor in text
    assert "data:image/png;base64," in text


def test_plot_predictions_writes_file(tmp_path):
    y_true = np.arange(20, dtype=float).reshape(10, 2)
    y_pred = y_true + 0.5
    path = tmp_path / "predictions.png"

    fig = plot_predictions(
        y_true,
        y_pred,
        channel_names=["AF3", "F7"],
        path=path,
    )
    assert path.exists()
    assert fig is not None
