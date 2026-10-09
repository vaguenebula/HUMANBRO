# HUMANBRO: MIDI velocity and timing humanization with XGBoost on MAESTRO

This project trains a gradient-boosted regressor that predicts the note-on velocity of each note in a
piano MIDI file from its musical context: pitch, metrical position, melodic and chord context, local
texture, and phrase position. It learns from the human performances in
[MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro), then writes the predicted velocities
into a deadpan MIDI file so you can listen to the result.

A second model handles **timing**: for quantized MIDI it predicts, per note, a distribution of onset
offsets (XGBoost quantile regression) and samples correlated micro-timing from it. See
[section 8](#8-timing-model-micro-timing-with-quantile-regression).

What it prioritises: no target leakage, honest evaluation (whole performances per split, baselines and
oracles side by side), and MIDI files you can actually audition. It does not aim for maximum model
complexity.

---

## 1. Setup

Requires Python 3.11 or newer. A CUDA GPU is optional; XGBoost uses one automatically if it can.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

### Getting MAESTRO

Use the MIDI-only release, `maestro-v3.0.0-midi.zip` (about 58 MB). Any one of these works:

```bash
python download_or_locate_maestro.py --zip maestro-v3.0.0-midi.zip --extract_to data   # zip you already have
python download_or_locate_maestro.py --download --extract_to data                      # fetch from Google storage
python download_or_locate_maestro.py --maestro_dir data/maestro-v3.0.0                 # just verify
```

`build_dataset.py --maestro_dir` also accepts the zip directly.

---

## 2. Exact commands

```bash
# 0. Self-checks (synthetic MIDI; no dataset needed)
python -m pytest tests -q

# 1. Features -> Parquet cache. ~20 s for all 1,276 performances with 16 workers; ~1.4 GB on disk
python build_dataset.py --maestro_dir data/maestro-v3.0.0 --output data/features.parquet

# 2. Train the primary model (feature set D, absolute velocity) plus the A/B/C/D ablation
python train_xgboost.py --data data/features.parquet --output_dir models/velocity_xgb --ablation

# 3. Evaluate on the test split: metrics, baselines, plots, and listening-test MIDI files
python evaluate.py --model models/velocity_xgb/model.json --data data/features.parquet --n_midi 6 --smoothing

# 4. Humanize any MIDI file
python humanize_midi.py --input input_flat.mid --model models/velocity_xgb/model.json --output output_humanized.mid
```

For **quantized DAW or score MIDI**, train the quantized model (see below) and humanize with
`models/velocity_xgb_quantized/model.json`. It is much better on grid-aligned input (see Results), and
`humanize_midi.py` warns you if the model and the input don't match.

### C++ runtime

```bash
python export_cpp_model.py --model models/velocity_xgb_quantized/model.json     # -> model.hbm
python export_cpp_model.py --model models/timing_xgb/model.json                 # timing model -> model.hbm
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 && cmake --build cpp/build --config Release
cpp/build/Release/humanbro.exe --model models/velocity_xgb_quantized/model.hbm \
    --timing-model models/timing_xgb/model.hbm --input in.mid --output out.mid  # velocity and/or timing
python tests/test_cpp_parity.py                                                 # velocity: bit-exact vs Python
python tests/test_timing_cpp_parity.py                                          # timing: same quantiles, takes, MIDI
```

See `cpp/README.md` for the library API (`Humanizer`, `TimingHumanizer`) and notes on plugin/DAW
integration. A timing take's `--seed` gives the same take in Python and C++.

### Timing model (micro-timing with quantile regression)

```bash
# Features of the reconstructed score + onset offsets. ~20 s with 16 workers
python build_dataset.py --task timing --maestro_dir data/maestro-v3.0.0 --output data/timing.parquet

# Multi-quantile XGBoost, early stopping on validation pinball loss, then the sampler fit. ~15 min on the GPU
python train_timing.py --data data/timing.parquet --output_dir models/timing_xgb

# Calibration, sharpness, realism of sampled timing, plots, listening-test MIDI
python evaluate_timing.py --model models/timing_xgb/model.json --n_midi 5

# Humanize the timing of any MIDI file (velocities untouched; combine with humanize_midi.py)
python humanize_timing.py --input clip.mid --model models/timing_xgb/model.json --output clip_timed.mid
```

| `humanize_timing.py` option | Effect |
|---|---|
| `--mode sample` (default) / `median` | A correlated random take from each note's predicted distribution, or the deterministic median |
| `--temperature 1.0` | Spread of the take: 0 = middle of each distribution, 1 = as loose as the training pianists, >1 looser |
| `--scale 1.0` | Multiply every final offset (0.5 = half as loose) |
| `--seed N` | Reproducible take, the same in the C++ runtime; omit it for a new take every run (the seed used is printed) |
| `--chord_coupling 0..1`, `--correlation_beats B` | Override the fitted sampler: how much chord notes move together, how long timing drifts persist |
| `--max_offset_ms 150` | Clamp |
| `--beat_source auto\|midi\|tracked`, `--snap` | Grid source; snap nearly-quantized input to the grid first |

### Other experiments

```bash
# Residual target: predict velocity minus the local baseline (rolling median of neighbouring notes)
python train_xgboost.py --data data/features.parquet --output_dir models/velocity_xgb_residual --target_mode residual

# Score-like input: snap onsets and durations to the tracked beat grid before extracting features
python build_dataset.py --maestro_dir data/maestro-v3.0.0 --output data/features_quantized.parquet --quantize
python train_xgboost.py --data data/features_quantized.parquet --output_dir models/velocity_xgb_quantized

# Performance-conditioned upper bound (inputs include neighbour/chord-mate velocities). Analysis only.
python train_xgboost.py --data data/features.parquet --output_dir models/velocity_xgb_pc --experiment performance_conditioned

# Causal (no look-ahead) features, for real-time use. Use beat.source=midi: the beat tracker looks ahead.
python build_dataset.py --output data/features_causal.parquet --set features.context=causal
```

### Useful options

| Script | Option | Effect |
|---|---|---|
| all | `--config my.yaml`, `--set section.key=value` | Configuration. For example, `--set xgb.max_depth=10` or `--set beat.fixed_meter=3` |
| build | `--limit_files N` | N random performances per split (quick experiments) |
| build | `--overwrite` | Rebuild parts that already exist. Without it, the build resumes |
| train | `--max_files_per_split N` | Subsample whole performances, never individual notes |
| train | `--device cpu\|cuda\|auto` | `auto` checks that CUDA really works before using it |
| evaluate | `--split validation`, `--max_files N`, `--no_plots`, `--shap_samples N` | |
| humanize | `--smoothing`, `--dynamics_scale 1.2`, `--offset -5`, `--mix 0.7` | Post-processing and blending with the input velocities |
| humanize | `--beat_source auto\|midi\|tracked\|model`, `--fixed_meter 3` | Where bar and beat positions come from |
| humanize | `--baseline input\|constant --base_velocity 64` | Residual models only |

---

## 3. Folder structure

```
HUMANBRO/
├── config.py                      dataclass config + YAML/--set loading
├── config.yaml                    defaults (identical to config.py; a test checks this)
├── download_or_locate_maestro.py  find / extract / download MAESTRO, read the split CSV
├── midi_io.py                     mido-based parsing (tempo map, time signatures, notes), velocity writing
├── beat_tracking.py               beat grid from a MIDI tempo map OR tracked from onsets (+ meter/downbeats)
├── midi_features.py               per-note features, targets, leakage guard, quantization, shared pipeline
├── build_dataset.py               MAESTRO -> partitioned Parquet feature cache (parallel, resumable)
├── train_xgboost.py               XGBRegressor training, early stopping, ablation, importance
├── evaluate.py                    metrics, baselines, oracles, plots, SHAP, listening-test MIDI
├── humanize_midi.py               apply a model to any MIDI file
├── model_io.py                    model + metadata loading, prediction, residual baselines
├── postprocess.py                 chord-aware smoothing, dynamics scaling
├── utils.py                       logging, seeds, GPU detection, metrics, Parquet loaders
├── export_cpp_model.py            model.json -> compact .hbm for the C++ runtime (verified vs XGBoost)
├── cpp/                           dependency-free C++17 runtime + CLI, velocity and timing (see cpp/README.md)
├── tests/test_pipeline.py         synthetic-MIDI tests (tempo maps, beat tracking, leakage, I/O)
├── tests/test_cpp_parity.py       C++ vs Python: identical features, predictions, velocities
├── timing_features.py             timing pipeline: score reconstruction, reference beat, offsets, features
├── timing_sampling.py             quantiles -> quantile function; Gaussian-copula sampler + recalibration
├── timing_metrics.py              pinball, calibration, intervals, event vs asynchrony, realism
├── train_timing.py                multi-quantile XGBoost training, sampler fit (--refit_sampler)
├── evaluate_timing.py             timing metrics, baselines, plots, listening-test MIDI
├── humanize_timing.py             apply the timing model to any MIDI file
├── tests/test_timing.py           synthetic tests: offset recovery, no micro-timing leakage, re-timing I/O, sampler
├── tests/test_timing_cpp_parity.py  C++ vs Python timing: identical quantiles, seeded takes and MIDI
├── requirements.txt
├── data/
│   ├── maestro-v3.0.0/            extracted dataset (+ maestro-v3.0.0.csv with the official split)
│   ├── features.parquet/          _meta.json + split=train|validation|test/part-<file_id>.parquet
│   └── timing.parquet/            the same layout for the timing task (features + offset_ms)
├── models/velocity_xgb/
│   ├── model.json                 booster, truncated to the best iteration
│   ├── model.meta.json            feature columns, target mode, pipeline config, train stats, metrics
│   ├── feature_importance.csv, metrics.json, ablation.csv, ablation.md
│   ├── ablation/{A,B,C}/          the ablation models
│   └── eval_test/                 summary.md, metrics.json, per_file_metrics.csv, plots/, midi/
├── models/timing_xgb/             model.json (7 quantiles), model.meta.json (+ sampler), eval_test/
└── logs/
```

---

## 4. How it works

### 4.1 Data and split

* Each note of each performance becomes one training example. That gives 7.04 M notes from 1,276
  performances.
* The split is MAESTRO's official train/validation/test assignment (962 / 137 / 177 performances). It
  keeps every performance, and every composition, inside a single split. The training script checks
  that no `file_id` appears in two splits. Notes are never split at random.
* Early stopping uses only the validation split. The test split is used only for reporting.

### 4.2 Timing, tempo and the beat grid (important for MAESTRO)

MAESTRO files are Disklavier recordings, and their tempo map is a placeholder: one 120 BPM event and 4/4
in every file (verified on the data). Bar and beat positions read from that map would just be
`seconds × 2`, so beat features computed from it would be silently meaningless. The pipeline therefore
supports two beat sources behind one `BeatGrid` interface:

* **`midi`**: the file's own tempo and time-signature map. Every tempo change and every time-signature
  change is used; none is discarded. Use this for DAW or score MIDI.
* **`tracked`**: beats are estimated from the notes. The estimate uses an onset envelope, a global and
  local tempo from autocorrelation, and Ellis-style dynamic programming with a time-varying period, so
  the beat follows rubato. Beats are then snapped to nearby chords. Meter (3 or 4) and downbeat phase
  come from per-beat salience: onset count, long notes, bass notes, and harmonic change. The phase is
  re-estimated per 32-beat segment with a Viterbi path. **The tracker never sees velocity.** On
  synthetic pieces with known beats it gets beat F≈0.96 and identifies the meter correctly. On extreme
  rubato it degrades, and it can lock onto half or double the notated tempo (see Limitations).
* **`auto`** (default in `humanize_midi.py`) picks `midi` when ≥80 % of onsets sit on a 1/16 or 1/12
  tick grid, which happens for less than 10 % of onsets by chance in a performance. Otherwise it picks
  `tracked`. Training on MAESTRO uses `tracked`.

Timing is kept both in **seconds** (true tempo-mapped time) and in **beats** (via the grid).

**Sustain and soft pedals** are never applied to note durations. Durations are key-down durations,
which are the closest a performance gets to written note lengths. Pedals are not used as features, and
they pass through unchanged when a file is rewritten.

### 4.3 Features (127 columns, all derivable from a deadpan score)

Notes are grouped into onset groups (chords: onsets within 35 ms of the group's first onset). Rows are
then ordered by (group, pitch). Chord-mates share one onset time and one metrical position.

| Group | Features | Why |
|---|---|---|
| **core** (ablation A) | pitch, pitch class, octave, beat in measure (raw and normalised), beat fraction, metrical level (downbeat / half-bar / beat / 8th / 16th-triplet / other), on-beat, strong-beat, downbeat | Register and metrical accent are the classic first-order drivers of dynamics |
| **note** | duration in seconds and in beats, onset beat, beat index, local tempo (BPM), time-signature numerator/denominator | Long notes and slow tempi are usually played louder. These are articulation and tempo context |
| **melodic** | previous/next interval and their absolute values (voice-leading neighbour = the nearest pitch in the previous/next chord), IOI to previous/next onset in seconds and beats, previous/next duration, duration ratio, melodic direction, local peak/trough, top-line (skyline) trend, time since/until the same pitch, pitch relative to the local mean/median of the ±8/±32 nearest notes | Melodic peaks, leaps and contour shape phrasing. Repeated notes are played differently |
| **local** | in ±0.25/0.5/1/3 s and ±1/2/4 beats: note density, onset density, mean pitch, pitch relative to mean, pitch range, mean duration. Notes in the previous/next beat. Pitch percentile, range and time-per-note among the nearest ±N notes | Texture: dense fast passages and wide registers differ systematically in loudness |
| **poly** (C adds) | chord size, notes within onset tolerance, keys held at onset, polyphony, chord highest/lowest/span, normalised position in chord, distance from chord top/bottom, rank from top/bottom, is top/bottom, local mean polyphony per window | Voicing: melody on top of a chord is louder and inner voices are softer. These are the strongest single cues |
| **structure** (D adds) | measure number, relative position in the piece, time since/until the last/next rest (silence ≥ 0.2 s), phrase-like segments (rests or gaps ≥ 1.5 beats): position, length, notes since/until the boundary | Phrase arcs, endings, and openings after rests |
| **piece** (D adds) | duration, note rate, pitch mean/sd, mean polyphony, chord size, median tempo, pitch relative to the piece mean | Whole-piece character (a dense virtuoso étude is louder than a nocturne) |

Missing values (no previous note, a single-note "chord", and so on) are left as NaN. XGBoost learns a
default direction for each split.

Excluded on purpose: velocity in any form, note-off velocity, pedals, and micro-timing deviations from
the beat. All of these are performance data, not score data.

### 4.4 Targets

* `absolute` (default): performed velocity, 1–127.
* `residual`: `velocity − baseline`. The baseline is the rolling median of the ±24 surrounding notes'
  velocities, **excluding the note's own chord**. The baseline belongs to the target transform and is
  never an input feature. At inference it comes from the input file's own velocities by default, so
  dynamics you drew in are kept and the model adds note-level shaping. You can also set it to a
  constant. Residual metrics are reported in delta space and as absolute velocity reconstructed with
  the oracle baseline, and they are labelled as such. They are not comparable with absolute-mode
  numbers.

### 4.5 How leakage is prevented

1. `extract_features()` has no velocity parameter, and the beat tracker has none either. Velocity
   cannot reach the features through the code path.
2. `assert_no_leakage()` rejects any input column containing `velocity`, prefixed `pc_`, or naming a
   target or meta column. The only exception is the explicitly labelled `performance_conditioned`
   experiment.
3. Tests feed the same notes with constant and with random velocities and require **bit-identical
   feature tables**.
4. `evaluate.py` re-runs this check on every listening-test file. It predicts from the original
   performance and from the flat-64 copy, and requires identical model outputs (`leakage_check` column).
5. The split is by performance, and the code checks it.

The performance-conditioned experiment adds previous/next chord mean velocity, chord-mates' mean
velocity, and the local baseline. It is an upper bound for analysis only. `humanize_midi.py` refuses to
run with such a model.

### 4.6 Model

`xgboost.XGBRegressor`: `hist` trees, `reg:squarederror`, up to 2000 trees, learning rate 0.03, depth 8,
subsample and colsample 0.8, `min_child_weight` 10, L1/L2 configurable. Early stopping uses validation
RMSE with 100 rounds of patience. The saved booster is truncated to the best iteration. The GPU is used
when `device=auto` and a CUDA test fit succeeds; otherwise training runs on the CPU. Seeds are fixed.

Training data is streamed from the Parquet parts into one float32 matrix per split. Only the needed
columns are loaded, which comes to about 3 GB for the full training split.

### 4.7 Evaluation

* MAE, RMSE, R², bias, Pearson, Spearman, and **within-performance Pearson** (the mean of per-file
  correlations). The last one ignores the overall loudness of each recording and measures whether the
  dynamic shape is right.
* Baselines (fit on train): global mean, global median, per-pitch mean.
* Oracles (they peek at the target and are shown to calibrate expectations): performance mean
  velocity, the local neighbour-median baseline, and the model re-centred on each performance's true
  mean.
* Plots: predicted vs actual (hexbin), residual histogram, error vs pitch, error and bias vs target
  velocity, true vs predicted distributions, top-30 gain importance, and SHAP. SHAP uses
  `shap.summary_plot` if `shap` is installed, otherwise XGBoost's built-in TreeSHAP shown as a mean
  |SHAP| bar chart.
* Listening test: for N test performances it writes the original, a flat-64 copy, the model's
  humanization of the flat copy, and optionally a smoothed version. Residual models also get a version
  built on the performance's own baseline.

### 4.8 Smoothing (optional, off by default)

`postprocess.py` smooths the mean velocity of successive onset groups with a zero-phase EMA. Each note
keeps its offset from its chord's mean, so voicing survives. Groups that sit more than
`accent_threshold` above the trend are treated as accents and left alone. It is controlled by
`strength` (0 to 1) and `alpha`.

---

## 5. Results (MAESTRO v3, official test split: 177 performances, 741,410 notes)

These come from one run on an RTX 4070 SUPER with the default config. Each full model trains in about
5–6 minutes on the GPU. The full logs are in `logs/` and the evaluation outputs in
`models/*/eval_test/`.

### Primary model vs baselines (absolute velocity, feature set D)

| model | MAE | RMSE | R² | Pearson | Spearman | within-performance r |
|---|---|---|---|---|---|---|
| **XGBoost, full features** | **10.15** | **13.24** | **0.503** | **0.709** | **0.701** | **0.676** |
| + smoothing (strength 0.5) | 10.28 | 13.38 | 0.493 | 0.704 | 0.700 | 0.672 |
| baseline: global mean (train) | 15.30 | 18.79 | 0.00 | – | – | – |
| baseline: global median (train) | 15.25 | 18.80 | 0.00 | – | – | – |
| baseline: per-pitch mean (train) | 14.34 | 17.83 | 0.099 | 0.317 | 0.308 | 0.363 |
| *oracle*: true performance mean | 14.43 | 17.90 | 0.092 | 0.303 | 0.299 | – |
| *oracle*: local neighbour median (±24 notes) | 10.50 | 14.30 | 0.420 | 0.652 | 0.670 | 0.560 |
| *oracle*: model re-centred on true performance mean | 9.73 | 12.76 | 0.538 | 0.734 | 0.728 | 0.676 |

The model, which sees no velocities at all, beats even the oracle that copies the median velocity of
the surrounding notes. Knowing each recording's true overall level would gain only another 0.4 MAE.

### Ablation (same hyperparameters; validation / test)

| feature set | # | val MAE | val r | val piece r | test MAE | test r | test piece r |
|---|---|---|---|---|---|---|---|
| A: pitch + beat position | 10 | 14.35 | 0.339 | 0.365 | 14.05 | 0.363 | 0.384 |
| B: + note, melodic, local context | 88 | 10.35 | 0.699 | 0.661 | 10.36 | 0.694 | 0.654 |
| C: + chord / polyphony | 109 | **10.08** | **0.718** | 0.680 | **10.09** | **0.712** | 0.673 |
| D: + structure, piece-level (full) | 127 | 10.23 | 0.709 | **0.685** | 10.15 | 0.709 | **0.676** |

* Pitch plus (tracked) beat position alone barely beats the per-pitch mean. Texture and melodic context
  carry most of the signal, and chord position adds a consistent gain.
* **D is not better than C on MAE.** The piece-level constants (duration, note rate, and so on) let
  trees partly memorise individual training performances. That adds a little within-piece shape but
  generalises no better. `--feature_set C`, or building with `--set features.include_piece_features=false`,
  is a reasonable default.

Top features by gain: `dist_from_chord_bottom`, `win3000ms_note_density`, `win3000ms_pitch_rel_mean`,
`win1000ms_note_density`, `pitch`, `octave`, `is_chord_top`, `n32_pitch_rel_mean`. By mean |SHAP|:
`dist_from_chord_bottom`, `win3000ms_note_density`, `ioi_prev_s`, `win3000ms_pitch_range`, `dur_s`.
The full lists are in `models/velocity_xgb/feature_importance.csv` and
`eval_test/plots/shap_mean_abs.csv`.

### Experiments

| experiment (test split) | MAE | Pearson | within-performance r | notes |
|---|---|---|---|---|
| primary (performance timing) | 10.15 | 0.709 | 0.676 | the table above |
| **quantized** timing (`--quantize`, its own model) | 11.28 | 0.622 | 0.592 | score-like input |
| primary model fed quantized input | 13.63 | 0.455 | 0.435 | train/inference mismatch |
| residual target, delta space | 8.14 (vs 10.50 for zero delta) | 0.633 | – | delta R² 0.40; absolute reconstruction needs a baseline |
| performance-conditioned (neighbour velocities) | 6.67 | 0.869 | 0.845 | upper bound, not usable for humanization |

**The most important practical finding:** part of the primary model's skill comes from expressive timing
in MAESTRO. Agogic and articulation cues such as `ioi_prev_s` and `dur_s` rank high. When the input
really is deadpan or quantized, the primary model degrades sharply (MAE 13.6). The model trained on
quantized features does much better on that input (MAE 11.3). **Use `models/velocity_xgb_quantized`
for quantized DAW/score MIDI and `models/velocity_xgb` for MIDI that was played in.**
`humanize_midi.py` warns when the two are mismatched.

### Listening test (`models/velocity_xgb/eval_test/midi/`)

Six test performances: Scarlatti K. 54, Chopin Op. 10/12, Bach WTC I G♯ minor, Beethoven Op. 31/1,
Schumann Op. 4, and Mozart K. 280. For each there is the original, a flat-64 copy, and the model's
humanization of the flat copy, with and without smoothing. The leakage check passed on all six:
predictions from the flat copy and from the original were identical. Humanized MAE against the real
performance is 6.9–12.2, compared with 10.2–17.1 for the flat file, with per-piece r of 0.65–0.79.
Predicted dynamics are **compressed**: overall sd is 13.3 vs 18.8 for the performances (see
`velocity_distributions.png`). Try `--dynamics_scale 1.3` when rendering.

---

## 6. Assumptions and limitations

* **Beat tracking is heuristic.** The tracked tactus can be half or double the notated beat, and meter
  and downbeat estimates are noisy on rubato-heavy Romantic repertoire. Bar-level features are
  therefore weaker on MAESTRO than they would be with a real score. With `--beat_source auto`, DAW or
  score MIDI gets its true grid at inference time. That input is cleaner than the training data, which
  is generally harmless for trees but is a distribution shift.
* **"Score-like" is approximate.** Onset timing and key-down durations in MAESTRO are themselves
  expressive. Timing that correlates with loudness, such as melody lead and agogic accents, can carry
  some information. The quantized experiment measures how much the model relies on it.
* **Per-recording loudness is unknowable from a score.** Different years and sessions, piano
  calibration, and the player all shift the whole performance up or down. The "performance mean"
  oracle and the within-piece correlation separate that from shape errors. In practice, set the level
  with `--offset` or with a residual model.
* All non-drum notes are treated as one piano texture. Multi-instrument files are not separated by
  track.
* Regression to the mean: squared-error trees under-predict extremes, so predicted dynamics are
  compressed (see `velocity_distributions.png`). `--dynamics_scale` and quantile sampling (below) address
  this.

---

## 7. Recommendations and extensions

Improvements to the baseline, roughly in order of expected payoff:

1. **Fix dynamic-range compression.** Calibrate the spread per piece (match the predicted sd to typical
   training sd for similar textures), or sample from quantile predictions (item 1 of the extensions).
   This is the most audible weakness.
2. **Better metrical information.** Use score-aligned data, such as the ASAP dataset (MAESTRO
   performances aligned to scores with real beats and downbeats), or a learned beat/downbeat tracker
   (madmom, BeatNet) in place of the heuristic tracker. Bar-level accent features would become reliable.
3. **Voice separation.** Skyline and nearest-pitch voice leading are crude. A proper voice/stream
   separation would make melody and accompaniment features much sharper.
4. **Piece-level normalisation.** Train on velocity minus the performance median (the residual target
   with a whole-piece window) and let the user choose the level. This removes noise the model cannot
   explain.
5. **Hyperparameters.** Early stopping never triggered: the best iteration was about 2000 of 2000, and
   validation RMSE was still falling by about 0.05 per 500 trees. Try `--set xgb.n_estimators=4000` or a
   higher learning rate. Use Optuna over depth, `min_child_weight`, `colsample`, and learning rate on
   validation within-piece correlation, not just RMSE.
6. **Drop the piece-level features** (`--feature_set C`). They did not generalise better than C in the
   ablation.
7. **Weighting.** Down-weight extremely dense passages (thousands of near-identical notes) so phrasing
   in sparse passages counts more.

Suggested extensions:

1. **XGBoost quantile regression** (`objective="reg:quantileerror"`, `quantile_alpha=[0.1, 0.5, 0.9]`):
   sample a velocity per note from the predicted interval for stochastic and less compressed
   humanization, keeping the samples correlated within a phrase.
2. **LightGBM / CatBoost comparison**: same Parquet cache and same splits. CatBoost's ordered boosting
   and native categorical handling (pitch class, metrical level) are worth a test.
3. **Velocity-delta target**: already implemented (`--target_mode residual`). Also try a whole-piece
   window and a two-stage model (absolute model → smoothed baseline → residual model).
4. **Separate melody and accompaniment models** once voices are inferred, or a single model with a voice
   role feature and interactions.
5. **Sequence-aware residual correction**: a small GRU/TCN or a CRF over the XGBoost residuals along the
   note sequence, to capture crescendo and phrase continuity that independent per-note trees miss.
6. **Transformer comparison**: a score-to-performance Transformer over note tokens (as in VirtuosoNet or
   ScorePerformer-style models), trained on the same splits, with the XGBoost model as the baseline it
   must beat on within-piece correlation and listening tests.
7. **Listening protocol**: blind A/B tests of flat vs humanized vs original excerpts. The numbers above
   only approximate what matters.

---

## 8. Timing model: micro-timing with quantile regression

The velocity model's sibling for **onset timing**. Input: quantized MIDI (a DAW clip or a score). Output:
the same notes, each moved by a few milliseconds the way a pianist would move it, keeping the host's
tempo map. It is trained on the same MAESTRO split and cache layout, and it never changes velocities.

### 8.1 What counts as "timing": the target

MAESTRO has no scores, so each performance is split into a score and its timing
(`timing_features.py`):

1. **Raw grid.** Beats are tracked from the performance with the same tracker as the velocity
   pipeline (section 4.2). It follows rubato.
2. **Score position.** Each onset group (a chord, 35 ms tolerance) is snapped *as a whole* to the raw
   grid. All notes of a chord therefore share one score position, and chord asynchrony (melody lead,
   rolled chords) becomes part of the target.
3. **Reference beat.** The raw beat times are smoothed by a local linear fit over ±2 beats. The target
   is

   `offset_ms = performed onset − reference-beat time of the score position`

   Tempo changes slower than that window count as tempo, not timing. In a DAW the host tempo map owns
   them. With ±8 beats, phrase-level rubato would leak into the target (offsets of ±250 ms). With
   ±2 beats, 98 % of offsets fall between −103 and +102 ms (sd 39 ms).
4. **Score timeline for features.** Score positions are mapped through a ±8-beat smoothed grid. It
   keeps section-level tempo, like a DAW tempo map, but none of the beat-to-beat timing the target is
   made of. Features are the same 109 score features as the velocity model's set C, computed on this
   timeline. **Performed onset times never reach the features.** A test checks that adding random
   micro-timing to a score leaves every feature bit-identical while the targets change.

**One grid per beat (an important fix).** DAW MIDI contains triplets, so training must too.
Snapping every note to the union of the 1/16 and triplet grids, however, makes the snapping cells around
the 2nd/4th 16th and the triplet positions lopsided. On MAESTRO that produced fake systematic timing:
2nd-16th notes averaged −20 ms and 4th-16th notes +20 ms, which a model would happily learn. Each beat
is now quantized to one subdivision, 16ths or triplets, chosen from the chords in it. A beat switches to
triplets only if that lowers its squared snapping error by more than `timing.subdivision_switch_cost`
(0.01 beats²). The fake ±20 ms shrank to ±3–4 ms, and that remainder (off-beats pulled slightly
towards mid-beat) also appears in the model's predictions, as it would for a real effect.

### 8.2 Model

One XGBoost booster with `objective="reg:quantileerror"` predicts seven quantiles of the offset per note
(5, 10, 25, 50, 75, 90, 95 %). It uses hist trees, depth 8, learning rate 0.08 and `min_child_weight` 20,
with early stopping on validation mean pinball loss (best round 666, 15 min on the GPU). Notes more than
200 ms from the reference beat get zero training weight: 0.05 % of notes, nearly all beat-tracking
errors. Crossing quantiles are fixed by sorting each row.

Feature-set ablation (200 files per split, test mean pinball, lower is better): A (pitch + beat position)
8.77, B 8.00, **C 7.93**, D 7.94, against 8.86 for the unconditional baseline. Beat position alone barely
helps (a good sign after the quantization fix). Structure and piece-level features add nothing over C, so
C is the default (`timing_model.feature_set`).

Top features by gain: `beat_frac`, `on_beat`, `is_chord_top`, `same_pitch_next_dt`, `metrical_level`,
`same_pitch_prev_dt`, `dur_beats`, `ioi_prev_beats`, `chord_pos_norm`, `ioi_next_beats`.

### 8.3 Sampling micro-timing from the quantiles

`timing_sampling.py` turns each note's seven quantiles into a quantile function, interpolated linearly on
the normal-score scale and extended linearly past the 5 and 95 % knots. A take is `Q(z)` for a
standard-normal `z` per note. Drawing `z` independently would be white jitter: chords smeared at random
and no continuity. Real deviations from the model's expectation are correlated, so `z` comes from a
**Gaussian copula fitted on the validation split's normal scores** (the true offsets pushed through
their predicted distributions):

* `z = √ρ · a(chord) + √(1−ρ) · e(note)`. The shared part `a` is an AR(1) process over successive onset
  groups whose correlation decays as `exp(−Δbeats / ℓ)`. Fitted: **ρ = 0.62** (chord-mates' residuals
  correlate at 0.62), **ℓ = 0.67 beats**.
* **Marginal recalibration.** The raw quantiles were slightly too narrow (90 % interval → 88.5 %
  coverage). The empirical quantiles of the validation normal scores are stored, and every draw is
  mapped through them. On test this gives calibration within ±0.01 at every level.
* `--temperature` scales `z` (0 = the middle of each note's distribution), and `--scale` scales the
  final offsets.

Every marginal is still the note's own (recalibrated) predicted distribution, so the takes stay loose
where pianists are loose and tight where they are tight. The predicted 10–90 % width ranges from about
25 to 180 ms by note, against 97 ms for everything in the unconditional baseline.

### 8.4 Results (official test split: 177 performances, 741,410 notes)

Distributions (offsets in ms; pinball = mean over the seven levels, lower is better):

| predicted distribution | mean pinball | 90 % interval coverage | width | 50 % coverage | width |
|---|---|---|---|---|---|
| **model, recalibrated on validation** (what the sampler uses) | **7.80** | 0.907 | 117 | 0.513 | 46 |
| model, raw quantiles | 7.80 | 0.885 | 109 | 0.492 | 44 |
| model median + one fixed residual spread (fit on validation) | 8.25 | 0.900 | 121 | 0.519 | 42 |
| unconditional: training quantiles for every note ("random humanize") | 8.86 | 0.902 | 131 | 0.511 | 44 |

* The model beats the unconditional distribution by 12 % in pinball loss.
* Shifting the centre gives a bit over half of that gain. Per-note spread gives the rest: narrower
  intervals at the same coverage. Without the quantiles (the fixed-spread row) that gain is lost.

Point prediction (median):

| prediction | MAE | RMSE | r | within-piece r | chord-level MAE / r | asynchrony MAE / r |
|---|---|---|---|---|---|---|
| model median | **26.95** | **36.85** | 0.356 | 0.355 | 22.26 / 0.281 | 13.74 / 0.493 |
| zero offset (deadpan) | 29.07 | 39.44 | – | – | 23.32 / – | 15.75 / – |

* Within-chord asynchrony is the most predictable part (r 0.49). The top voice leads its chord by
  2.26 ms in the performances and by 2.34 ms in the model. The second voice is about 2 ms late, also
  matched (`eval_test/plots/chord_asynchrony.png`).
* Chord-level timing relative to the local beat is much less predictable from a score (r 0.28). Most of
  it is the performer's moment-to-moment choice, which is exactly why sampling from a distribution suits
  this target better than adding one "correct" offset.

Realism of generated timing (one take, temperature 1, vs the real performances):

| offsets | sd | chord-level sd | chord-to-chord continuity (lag-1 r) | within-chord sd |
|---|---|---|---|---|
| performed | 39.4 | 31.9 | 0.48 | 24.3 |
| model median only | 14.0 | 9.4 | −0.01 | 11.2 |
| **model, sampled** | **40.8** | **34.5** | **0.38** | **23.0** |
| random iid (unconditional) | 38.8 | 30.6 | 0.00 | 32.0 |

* The median alone is three times too tight.
* Random jitter of the right size has no continuity and smears chords about 30 % too much.
* The sampled takes match the performances on all four statistics, a little short on continuity (see
  8.5).

Listening test (`models/timing_xgb/eval_test/midi/`): Scarlatti K. 54, Chopin Op. 10/12 (two
performances), Bach WTC I G♯ minor prelude and fugue, and Beethoven Op. 31/3. Each comes as the
original, the deadpan reconstructed score, the median, a sampled take, an iid random take, and the
performer's true offsets on the score timeline (the ceiling for this target). Velocities, durations and
pedalling are the performer's in every file, so only onset timing differs.

### 8.5 Limitations and next steps

* **The score is reconstructed, not given.** Quantization errors from the beat tracker (half/double
  tempo, skipped beats, polyrhythms within one beat) leak into the targets. The proper fix is
  score-aligned data: ASAP / (n)ASAP note alignments for about 1,000 MAESTRO performances would give
  true score positions and replace steps 1–2 of 8.1.
* **How loose a performance is depends on the performer, not the score.** The Bach fugue performance
  has an offset sd of 15 ms, but the model, which cannot know that this pianist is unusually steady,
  samples about 31 ms. Treat `--temperature` / `--scale` as the "tightness" knob.
* **Continuity is slightly underestimated.** The empirical correlation of residuals is 0.49 at a 16th
  apart, 0.39 at an 8th, 0.25 at a dotted 8th, and turns slightly negative at about 2 beats. The
  ±2-beat reference beat absorbs slower drift, which pulls back. A single exponential gets 0.42, 0.29
  and 0.20. A two-parameter kernel (for example a damped cosine) would fit better.
* **Beat unit.** DAW grids use the time-signature denominator as the beat, so 6/8 or 2/2 input has
  beats unlike the tracked tactus the model was trained on (also true for the velocity model). A
  per-meter beat-unit mapping is a small improvement.
* **Velocity is not an input.** Melody lead is partly a "velocity artifact": louder notes sound earlier.
  Conditioning the timing model on velocities (the input file's, or the velocity model's output) is a
  natural next step, trained on performed velocities with a check against predicted ones.
* **No duration or articulation model yet.** Key-down durations are kept as written.
* **C++ covers quantized input only.** The C++ runtime (`TimingHumanizer`, `humanbro --timing-model`)
  reproduces the quantized-input path exactly: same quantiles, same seeded takes, same MIDI. Re-timing a
  *played* performance needs the beat tracker and score reconstruction, which are Python-only.
