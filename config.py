"""Central configuration for the velocity-humanization pipeline.

Every tunable lives in a dataclass below. Values can come from three places,
later ones winning:

1. the dataclass defaults in this file,
2. a YAML file (``--config config.yaml``) with the same nested structure,
3. ``--set section.key=value`` command-line overrides (values parsed as YAML,
   so ``--set features.quantize=true`` or ``--set features.time_windows_s=[0.5,1]``).

The *pipeline* sections (``beat``, ``features``, ``target``) are frozen into the
feature-cache metadata at build time and copied into the model metadata at
train time, so ``humanize_midi.py`` always re-creates exactly the features the
model was trained on.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    maestro_dir: str = "data/maestro-v3.0.0"
    features_path: str = "data/features.parquet"
    # Channel 10 (index 9) is percussion in General MIDI; it is never piano.
    include_drums: bool = False


@dataclass
class BeatConfig:
    # Where beat/bar positions come from:
    #   "tracked": estimate beats from note onsets (never from velocity). Required
    #              for MAESTRO, whose MIDI files carry a dummy 120 BPM / 4/4 map.
    #   "midi":    trust the file's tempo + time-signature map (DAW / score MIDI).
    #   "auto":    use "midi" when onsets sit on the file's tick grid, else "tracked".
    source: str = "tracked"
    frame_rate: float = 50.0  # onset-envelope frames per second for the tracker
    tempo_prior_bpm: float = 100.0  # centre of the log-normal tempo prior
    tempo_prior_octaves: float = 1.0  # std-dev of that prior, in octaves
    min_bpm: float = 40.0
    max_bpm: float = 220.0
    local_tempo_window_s: float = 6.0
    local_tempo_hop_s: float = 1.0
    local_tempo_max_ratio: float = 1.5  # local tempo stays within [g/r, g*r] of global
    tightness: float = 60.0  # DP penalty on deviating from the local beat period
    snap_to_onset_frac: float = 0.15  # snap beats to onsets within this fraction of a beat
    meter_candidates: list[int] = field(default_factory=lambda: [3, 4])
    fixed_meter: int | None = None  # force beats-per-bar when tracking (phase still estimated)
    meter_segment_beats: int = 32  # downbeat phase is re-estimated per segment of this many beats
    meter_phase_change_penalty: float = 2.0
    auto_quantized_fraction: float = 0.8  # "auto": fraction of onsets on a 1/16 or 1/12 grid


@dataclass
class FeatureConfig:
    # "bidirectional" (offline humanization, future context allowed) or "causal"
    # (only past/present context; for real-time use). Causal mode drops every
    # feature that needs look-ahead.
    context: str = "bidirectional"
    chord_tolerance_s: float = 0.035  # notes starting within this of a group's first onset form a chord
    time_windows_s: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 3.0])
    beat_windows: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    nearest_n: list[int] = field(default_factory=lambda: [8, 32])
    strong_beat_tolerance: float = 0.12  # in beats
    rest_min_s: float = 0.2  # silence (no key held) at least this long counts as a rest
    phrase_gap_beats: float = 1.5  # inter-onset gap (beats) that marks a phrase-like boundary
    phrase_gap_min_s: float = 0.35  # ...and it must also be at least this many seconds
    include_piece_features: bool = True
    # Second experiment: snap onsets/offsets to the beat grid before extracting
    # features, to approximate genuinely score-like (deadpan) input.
    quantize: bool = False
    quantize_subdivisions: list[int] = field(default_factory=lambda: [4])  # [4, 3] adds triplets


@dataclass
class TargetConfig:
    # "absolute": predict performed velocity.
    # "residual": predict velocity - local baseline (rolling statistic of the
    #             *surrounding* notes' velocities, own chord excluded).
    mode: str = "absolute"
    residual_window_notes: int = 24  # notes on each side used for the baseline
    residual_stat: str = "median"  # "median" or "mean"


@dataclass
class XGBConfig:
    objective: str = "reg:squarederror"
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 10.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    max_bin: int = 256
    early_stopping_rounds: int = 100
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    n_jobs: int = -1


@dataclass
class TrainConfig:
    seed: int = 42
    feature_set: str = "D"  # A | B | C | D (see midi_features.FEATURE_SETS)
    # "primary": score-to-performance model (no velocity information in inputs).
    # "performance_conditioned": adds neighbouring/chord-mate velocities. Analysis only.
    experiment: str = "primary"
    max_files_per_split: int | None = None  # subsample whole files (never notes) for quick runs
    verbose_every: int = 100


@dataclass
class TimingConfig:
    """Timing pipeline: how a performance is split into a score and micro-timing offsets."""

    # Onset groups (chords) are snapped as a whole to the tracked grid, one subdivision
    # per beat. 3 = triplets: DAW/score MIDI contains them, so the training data must too.
    quantize_subdivisions: list[int] = field(default_factory=lambda: [4, 3])
    # A beat leaves the first subdivision only if another lowers its total squared
    # snapping error by more than this (beats^2). Resolves e.g. uneven eighths vs triplets.
    subdivision_switch_cost: float = 0.01
    # Half-width (beats) of the local linear fit that turns tracked beats into the
    # reference beat. Deviations from it are the target ("timing"); slower changes
    # are tempo and are not predicted.
    reference_smoothing_beats: int = 2
    # Half-width (beats) of the smoothing that gives the score timeline the features
    # are computed on: section-level tempo only, like a DAW tempo map.
    score_smoothing_beats: int = 8


@dataclass
class TimingModelConfig:
    """Quantile XGBoost model for micro-timing offsets (train_timing.py)."""

    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    feature_set: str = "C"  # ablation: D (structure + piece-level) adds nothing over C
    # Notes further than this from the reference beat get zero training weight
    # (nearly always beat-tracking errors). They are still evaluated.
    max_abs_offset_ms: float = 200.0
    n_estimators: int = 1500
    learning_rate: float = 0.08
    max_depth: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 20.0
    reg_lambda: float = 1.0
    max_bin: int = 256
    early_stopping_rounds: int = 50


@dataclass
class SmoothingConfig:
    enabled: bool = False
    alpha: float = 0.35  # EMA coefficient over successive onset groups
    strength: float = 0.5  # 0 = no smoothing, 1 = replace by the smoothed trend
    accent_threshold: float = 8.0  # groups louder than the trend by this much are left alone


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    beat: BeatConfig = field(default_factory=BeatConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    timing_model: TimingModelConfig = field(default_factory=TimingModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def pipeline_dict(self) -> dict[str, Any]:
        """The sections that determine how MIDI turns into features/targets."""
        d = self.to_dict()
        return {k: d[k] for k in ("beat", "features", "target")}

    def timing_pipeline_dict(self) -> dict[str, Any]:
        """The sections that determine how MIDI turns into timing features/targets."""
        d = self.to_dict()
        return {k: d[k] for k in ("beat", "features", "timing")}


def _section_classes() -> dict[str, type]:
    return {f.name: type(getattr(Config(), f.name)) for f in dataclasses.fields(Config)}


def config_from_dict(d: dict[str, Any] | None, base: Config | None = None) -> Config:
    """Merge a (possibly partial) nested dict into a Config. Unknown keys are errors."""
    cfg = dataclasses.replace(base) if base is not None else Config()
    classes = _section_classes()
    for section, values in (d or {}).items():
        if section not in classes:
            raise KeyError(f"unknown config section '{section}' (valid: {sorted(classes)})")
        if values is None:
            continue
        current = getattr(cfg, section)
        valid = {f.name for f in dataclasses.fields(current)}
        unknown = set(values) - valid
        if unknown:
            raise KeyError(f"unknown keys in [{section}]: {sorted(unknown)}")
        setattr(cfg, section, dataclasses.replace(current, **values))
    return cfg


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply ``section.key=value`` strings; values are parsed as YAML scalars/lists."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        path, raw = item.split("=", 1)
        parts = path.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"override '{item}' must be section.key=value")
        cfg = config_from_dict({parts[0]: {parts[1]: yaml.safe_load(raw)}}, base=cfg)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = config_from_dict(yaml.safe_load(fh) or {}, base=cfg)
    return apply_overrides(cfg, overrides)


def pipeline_config_from_dict(d: dict[str, Any]) -> Config:
    """Rebuild a Config whose beat/features/target sections come from saved metadata."""
    return config_from_dict({k: d[k] for k in ("beat", "features", "target", "timing") if k in d})


def add_config_args(parser: Any) -> None:
    parser.add_argument("--config", default=None, help="YAML config file (see config.yaml)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set xgb.max_depth=10 (repeatable)",
    )
