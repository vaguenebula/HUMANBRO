"""Load a trained model + metadata and turn feature rows into velocities or timing quantiles.

A model is two files side by side:
    model.json        the XGBoost booster (already truncated to the best iteration)
    model.meta.json   feature columns (in order), target mode, the beat/feature/target
                      pipeline config used to build its training data, and metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import Config, pipeline_config_from_dict
from midi_features import residual_baseline
from timing_sampling import CopulaParams, sort_quantiles
from utils import load_json


def meta_path_for(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_name(p.stem + ".meta.json")


@dataclass
class VelocityModel:
    booster: xgb.Booster
    meta: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, model_path: str | Path, device: str = "cpu") -> VelocityModel:
        model_path = Path(model_path)
        meta_file = meta_path_for(model_path)
        if not meta_file.exists():
            raise FileNotFoundError(f"model metadata {meta_file} not found next to {model_path}")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        booster.set_param({"device": device})
        return cls(booster, load_json(meta_file), model_path)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.meta["feature_columns"])

    @property
    def target_mode(self) -> str:
        return self.meta["target_mode"]

    @property
    def experiment(self) -> str:
        return self.meta.get("experiment", "primary")

    def pipeline_config(self) -> Config:
        return pipeline_config_from_dict(self.meta["pipeline_config"])

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Model output: velocity (absolute mode) or velocity delta (residual mode)."""
        cols = self.feature_columns
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} model features missing from the input (e.g. {missing[:5]}); "
                "were the features built with the same pipeline config?"
            )
        return self.booster.inplace_predict(df[cols].astype(np.float32)).astype(np.float64)

    def to_velocity(self, raw: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
        if self.target_mode == "absolute":
            return raw
        if baseline is None:
            raise ValueError("residual model needs a baseline velocity per note")
        return np.asarray(baseline, dtype=np.float64) + raw

    def inference_baseline(
        self, input_velocity: np.ndarray, group_id: np.ndarray, source: str = "input", constant: float = 64.0
    ) -> np.ndarray:
        """Baseline for residual models at inference time (canonical order).

        "input":    rolling median of the *input file's* velocities, i.e. the macro
                    dynamics already drawn in by the user (a flat file gives a flat baseline);
        "constant": a fixed level.
        """
        if source == "constant":
            return np.full(len(group_id), float(constant))
        if source != "input":
            raise ValueError(f"unknown baseline source '{source}'")
        tcfg = self.pipeline_config().target
        return residual_baseline(input_velocity, group_id, tcfg.residual_window_notes, tcfg.residual_stat)


@dataclass
class TimingModel:
    """Multi-quantile XGBoost model of onset offsets (ms) plus the fitted sampling copula."""

    booster: xgb.Booster
    meta: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, model_path: str | Path, device: str = "cpu") -> TimingModel:
        model_path = Path(model_path)
        meta_file = meta_path_for(model_path)
        if not meta_file.exists():
            raise FileNotFoundError(f"model metadata {meta_file} not found next to {model_path}")
        meta = load_json(meta_file)
        if meta.get("task") != "timing":
            raise ValueError(f"{model_path} is not a timing model (train it with train_timing.py)")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        booster.set_param({"device": device})
        return cls(booster, meta, model_path)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.meta["feature_columns"])

    @property
    def quantiles(self) -> np.ndarray:
        return np.asarray(self.meta["quantiles"], dtype=np.float64)

    @property
    def copula(self) -> CopulaParams:
        return CopulaParams.from_dict(self.meta.get("copula"))

    def pipeline_config(self) -> Config:
        return pipeline_config_from_dict(self.meta["pipeline_config"])

    def predict_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        """(notes, quantiles) offsets in ms, sorted per row (no quantile crossing)."""
        cols = self.feature_columns
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{len(missing)} model features missing from the input (e.g. {missing[:5]})")
        q = self.booster.inplace_predict(df[cols].astype(np.float32))
        return sort_quantiles(np.asarray(q, dtype=np.float64).reshape(len(df), -1))

    def median_index(self) -> int:
        return int(np.argmin(np.abs(self.quantiles - 0.5)))
