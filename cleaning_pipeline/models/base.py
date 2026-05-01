from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from numpy.typing import NDArray

FeatureBlocks = Mapping[str, slice]
EvalSet = tuple[NDArray, NDArray] | None


class EEGRegressor(Protocol):
    """Common interface for models predicting all EEG channels."""

    def fit(
        self, X: NDArray, y: NDArray, eval_set: EvalSet = None
    ) -> "EEGRegressor":
        ...

    def predict(self, X: NDArray) -> NDArray:
        ...


def as_config_dict(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    return dict(config)


def normalize_model_name(name: str | None) -> str:
    if name is None:
        return "linear_regression"
    normalized = name.replace("-", "_").lower()
    aliases = {
        "linearregression": "linear_regression",
        "linear_regression": "linear_regression",
        "lr": "linear_regression",
        "catboostregressor": "catboost",
        "catboost": "catboost",
        "lr_2steps": "two_step_residual",
        "two_steps": "two_step_residual",
        "two_step": "two_step_residual",
        "two_step_residual": "two_step_residual",
    }
    return aliases.get(normalized, normalized)


def require_feature_block(feature_blocks: FeatureBlocks, name: str) -> slice:
    try:
        return feature_blocks[name]
    except KeyError as exc:
        available = ", ".join(sorted(feature_blocks)) or "<none>"
        raise ValueError(
            f"Feature block {name!r} is not available. Available blocks: {available}"
        ) from exc


def validate_prediction_shape(
    prediction: NDArray, n_samples: int, eeg_channels: Sequence[str]
) -> None:
    expected = (n_samples, len(eeg_channels))
    if prediction.shape != expected:
        raise ValueError(
            f"Model predicted shape {prediction.shape}, expected {expected}"
        )
