from typing import List

import mne

from emotivphysicimu.features import FeatureSet, extract_features

FeatureExtractionResult = FeatureSet


class MovementFeatureExtractor:
    def __init__(self,
                 motion_data: mne.io.Raw,
                 raw_data: mne.io.Raw,
                 eeg_channels: List[str]) -> None:
        
        self.raw_data = raw_data
        self.motion_data = motion_data
        self.eeg_channels = eeg_channels

    def compute_src_tgt_features(self,
                                 n_points: int = 10,
                                 physics_informed_artefacts: bool = False,
                                 target_ratio: int = 4) -> FeatureExtractionResult:
        """Compute the independant and dependant variables for the signal regression

        ARGS:
        -----
            - n_points: int, the number of last points used to extract the features

        RETURNS:
        ----------
            - src_features: NDArray, the independant variables for the signal regression
            - tgt_features: NDArray, the dependant variables for the signal regression
        """
        return extract_features(
            raw_imu=self.motion_data,
            raw_eeg=self.raw_data,
            eeg_channels=self.eeg_channels,
            n_points=n_points,
            target_ratio=target_ratio,
            physics=physics_informed_artefacts,
        )
