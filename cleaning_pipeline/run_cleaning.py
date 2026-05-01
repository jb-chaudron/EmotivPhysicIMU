from emotivphysicimu.cli import ModelConfig, PipelineConfig, load_feature_matrix, main
from emotivphysicimu.constants import EMOTIV_CHANNELS as EEG_CH_NAMES

__all__ = ["EEG_CH_NAMES", "ModelConfig", "PipelineConfig", "load_feature_matrix", "main"]


if __name__ == "__main__":
    main()
