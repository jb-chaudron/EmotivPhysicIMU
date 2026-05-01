from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping, Sequence

from .base import (
    EEGRegressor,
    EvalSet,
    FeatureBlocks,
    as_config_dict,
    normalize_model_name,
)
from .Catboosts import CatBoostEEGRegressor
from .LinearRegressions import PerChannelLinearRegression
from .Residuals import TwoStepResidualRegressor


def build_model(
    model_config: Mapping[str, Any] | Any | None,
    eeg_channels: Sequence[str],
    feature_blocks: FeatureBlocks | None = None,
) -> EEGRegressor:
    config = as_config_dict(model_config)
    name = normalize_model_name(config.get("type") or config.get("name"))
    random_state = config.get("random_state")
    params = dict(config.get("params") or {})

    if name == "linear_regression":
        return PerChannelLinearRegression(eeg_channels=eeg_channels, params=params)

    if name == "catboost":
        channel_handling = config.get("channel_handling", "multioutput")
        return CatBoostEEGRegressor(
            eeg_channels=eeg_channels,
            channel_handling=channel_handling,
            random_state=random_state,
            params=params,
        )

    if name == "two_step_residual":
        if feature_blocks is None:
            raise ValueError("two_step_residual requires named feature_blocks")
        first_features = config.get("first_features", "lorentz")
        residual_features = config.get("residual_features", "motion")
        first_model_config = dict(config.get("first_model_config") or {})
        residual_model_config = dict(config.get("residual_model_config") or {})

        base_model_name = config.get("base_model", "linear_regression")
        first_model_config.setdefault("name", base_model_name)
        residual_model_config.setdefault("name", base_model_name)
        if random_state is not None:
            first_model_config.setdefault("random_state", random_state)
            residual_model_config.setdefault("random_state", random_state)

        def make_builder(stage_config: dict[str, Any]) -> Callable[[], EEGRegressor]:
            def builder() -> EEGRegressor:
                return build_model(
                    stage_config,
                    eeg_channels=eeg_channels,
                    feature_blocks=feature_blocks,
                )

            return builder

        return TwoStepResidualRegressor(
            eeg_channels=eeg_channels,
            feature_blocks=feature_blocks,
            first_features=first_features,
            residual_features=residual_features,
            first_model_builder=make_builder(first_model_config),
            residual_model_builder=make_builder(residual_model_config),
        )

    raise ValueError(f"Unknown model name {config.get('name')!r}")


__all__ = [
    "CatBoostEEGRegressor",
    "EEGRegressor",
    "EvalSet",
    "FeatureBlocks",
    "PerChannelLinearRegression",
    "TwoStepResidualRegressor",
    "build_model",
]
