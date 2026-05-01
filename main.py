def main():
    print("Hello from emotivphysicimu!")


if __name__ == "__main__":
    from emotivphysicimu.features import extract_features
    from emotivphysicimu.constants import IMU_CHANNELS_UNIQUE, EMOTIV_CHANNELS
    import mne
    path = ""
    raw = mne.io.read_raw(path)
    raw.load_data()
    raw = raw.filter(1, 40)
    fs = extract_features(raw, imu_channels=IMU_CHANNELS_UNIQUE)
    print("X:", fs.X.shape, "y:", fs.y.shape, "sfreq:", fs.sfreq)

    from emotivphysicimu.model import IMURegressor
    model = IMURegressor(sfreq=fs.sfreq, eeg_channels=EMOTIV_CHANNELS, channel_handling="all_channels",conformal=True)
    model.fit(fs.X, fs.y)
    print("Model fitted")

    y_pred = model.predict(fs.X)
    print("y_pred:", y_pred.shape)

    print(model.score(fs.X, fs.y))

    from emotivphysicimu import IMUReport

    IMUReport(model, X_train=fs.X, y_train=fs.y).generate("report.html")