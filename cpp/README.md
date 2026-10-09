# HUMANBRO C++ runtime

This is a dependency-free C++17 port of the inference path for both models. A MIDI file or a DAW clip
goes in, features are computed, the tree ensemble runs, and humanized velocities and/or micro-timing come
out.

* **Velocity model.** The output is identical to the Python pipeline: the 127 features match bit for bit,
  and the written velocities are identical.
* **Timing model** (quantile regression + correlated sampling). The 7 predicted quantiles and the median
  offsets are bit-identical to Python. A seeded random take matches Python's take for the same seed to
  within 3e-13 ms, and the written MIDI files are identical event for event.

`tests/test_cpp_parity.py` and `tests/test_timing_cpp_parity.py` check all of this.

There is no XGBoost, Python or other runtime dependency. Models are compact `.hbm` files evaluated by a
small built-in tree walker: about 16 MB for the velocity model and 30 MB for the timing model (7 quantiles
× 667 rounds = 4,669 trees).

## 1. Export the models (Python side)

```bash
python export_cpp_model.py --model models/velocity_xgb_quantized/model.json   # -> model.hbm next to it
python export_cpp_model.py --model models/timing_xgb/model.json               # timing model
```

Each export checks itself against XGBoost on 20,000 rows and refuses to write a file that doesn't
reproduce it (both current models reproduce it exactly). The `.hbm` file holds the trees, the feature
order and the feature-pipeline settings the model was trained with. A timing model also holds its quantile
levels, the sampler (within-chord correlation, correlation length, recalibration table) and the grid-snap
settings. Format version 2 records the output of each tree. Version-1 velocity files from earlier
exports still load.

## 2. Build

```bash
# Windows (Visual Studio 2022)
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Release          # -> cpp/build/Release/humanbro.exe

# Windows with Ninja (run from a "x64 Native Tools" prompt, or call vcvars64.bat first)
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build                           # -> cpp/build/humanbro.exe

# Linux / macOS
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build                           # -> cpp/build/humanbro
```

This produces `humanbro` (the static library) and the `humanbro` CLI. Keep the strict floating-point
flags set in `CMakeLists.txt` (`/fp:precise`, `-ffp-contract=off`, no `-ffast-math`). Fast-math would
change feature values in the last bits and break exact parity.

## 3. Command line

```bash
# velocities only (as humanize_midi.py)
humanbro --model models/velocity_xgb_quantized/model.hbm --input clip.mid --output clip_v.mid --dynamics-scale 1.3

# micro-timing only (as humanize_timing.py --beat_source midi)
humanbro --timing-model models/timing_xgb/model.hbm --input clip.mid --output clip_t.mid
humanbro --timing-model models/timing_xgb/model.hbm --input clip.mid --output clip_t.mid --seed 7 --temperature 0.8

# both in one pass: both models read the original clip, the file is written once
humanbro --model models/velocity_xgb_quantized/model.hbm --timing-model models/timing_xgb/model.hbm \
         --input clip.mid --output clip_human.mid
```

The timing options mirror `humanize_timing.py`: `--timing-mode sample|median`, `--temperature`,
`--timing-scale`, `--seed`, `--max-offset-ms`, `--chord-coupling`, `--correlation-beats` and `--snap`.
Without `--seed` every run is a new take, and the CLI prints the seed it used, so a take you like can be
reproduced in C++ or in Python. `--dump-features`, `--dump-raw` (velocity) and `--dump-timing` (quantiles
and offsets) write CSVs for debugging and parity checks.

When re-timing, every note keeps its duration (the release moves with the onset). Notes of the same key
keep their order, and a release never cuts the next strike of that key short. All other events (tempo,
time signatures, pedals, controllers, drums) keep their ticks and bytes.

## 4. Library API

```cpp
#include <humanbro/humanbro.hpp>

humanbro::Humanizer velocity("velocity.hbm");     // load once, reuse; const methods are thread-safe
humanbro::TimingHumanizer timing("timing.hbm");   // ~130 ms to load

humanbro::Score score;
score.ticks_per_quarter = 960;
score.tempo_changes   = {{0, 500000.0}};          // microseconds per quarter note (120 BPM)
score.time_signatures = {{0, 3, 4}};
score.notes.push_back({/*onset*/ 0, /*offset*/ 900, /*pitch*/ 60, /*velocity*/ 64});
// ... every non-drum note of the clip

humanbro::Options opt;
opt.dynamics_scale = 1.3;
std::vector<int> velocities = velocity.humanize(score, opt);    // velocities[i] belongs to score.notes[i]

humanbro::TimingOptions topt;
topt.temperature = 0.8;                           // a little tighter than the training pianists
topt.seed = 1234;                                 // omit for a new take every call
humanbro::TimingResult t = timing.humanize(score, topt);
// t.onset_tick[i], t.offset_tick[i]: new note-on/off ticks of score.notes[i]
// t.offset_ms[i]: the shift applied; t.seed: the seed used (store it to recreate the take)
```

Lower-level calls work as for velocities. `TimingHumanizer::features(score)` returns the `FeatureTable`,
and `predict_quantiles(table)` returns rows × levels of offsets in ms, sorted per row, in canonical order.
`quantile_levels()`, `chord_coupling()` and `correlation_beats()` report the model's settings.

`TimingOptions`: `sample` (false = deterministic median offsets), `temperature` (0 = middle of each
note's distribution, 1 = as loose as the training data), `scale`, `max_offset_ms`, `seed`,
`chord_coupling` and `correlation_beats` (optional overrides of the fitted sampler), `snap`.

### Using it in a plugin or DAW (JUCE, VST3, CLAP, ...)

* **It works on whole clips, not as a real-time stream.** The features use future context (next note,
  time until the next rest, phrase length, piece-level statistics). Run it as an offline "humanize
  selection/clip" action, not inside `processBlock`. Velocity takes about 40 ms for 200 notes and about
  170 ms for 11,000 notes. Timing takes about 30 ms for 150 notes and about 0.55 s for 17,000 notes on
  one desktop CPU. Prediction uses all cores for large clips.
* **Run it off the audio thread.** It allocates memory, and a large clip can take a noticeable fraction
  of a second.
* **Fill `Score` from the host.** Use clip notes in ticks with key-down note-offs (don't extend them by
  sustain pedal), the host tempo map, and time signatures. Leave out drum tracks. The grid comes from
  this tempo map, which is right for DAW material and is what both quantized models expect.
* **Timing: apply the new ticks to the clip.** Move each note to `onset_tick[i]` / `offset_tick[i]`.
  If the clip mixes MIDI channels, set `Note::channel` so same-key notes on different channels are kept
  apart. Keep the seed with the clip if users should be able to get the same take back. Tighter or looser
  is a user taste (`temperature`, `scale`): the model can't know how strict a given player would be.
* **Order with velocity.** Both models expect the quantized clip. Compute velocities and timing from the
  same input, as the CLI does, rather than feeding re-timed notes to the velocity model.
* Untested JUCE sketch:

```cpp
humanbro::Score toScore(const juce::MidiMessageSequence& seq, int ppq, double bpm, int num, int den) {
    humanbro::Score s;
    s.ticks_per_quarter = ppq;
    s.tempo_changes = {{0, 60'000'000.0 / bpm}};
    s.time_signatures = {{0, num, den}};
    for (int i = 0; i < seq.getNumEvents(); ++i) {
        auto* e = seq.getEventPointer(i);
        if (!e->message.isNoteOn() || e->message.getChannel() == 10) continue;
        const auto off = e->noteOffObject ? e->noteOffObject->message.getTimeStamp() : e->message.getTimeStamp();
        humanbro::Note n{(int64_t) e->message.getTimeStamp(), (int64_t) off,
                         e->message.getNoteNumber(), e->message.getVelocity()};
        n.channel = e->message.getChannel() - 1;
        s.notes.push_back(n);
    }
    return s;   // call seq.updateMatchedPairs() first; timestamps in ticks
}
```

## 5. What matches Python, and what doesn't

| Python | C++ |
|---|---|
| `midi_io.parse_midi` (mido semantics: running status, re-strikes, unreleased notes, drums skipped) | `MidiFile::load` |
| tempo map, `grid_from_midi`, `BeatGrid.extended`, `quantize_notes` | `pipeline.cpp` |
| `extract_features` (bidirectional, all 127 features) | `extract_features` |
| residual baseline, smoothing, `shape_dynamics`, `mix`, clipping | `Humanizer::humanize` |
| XGBoost `inplace_predict` (single and multi-output) | `TreeModel::predict` |
| `timing_features.featurize_quantized`, `snap_groups` / `snap_beats` | `timing.cpp` |
| `timing_sampling`: `PortableRng`, copula, recalibration, quantile function | `timing.cpp` |
| `humanize_timing.retime`, `midi_io.with_timing` | `TimingHumanizer::humanize`, `MidiFile::save_with_timing` |

Not ported:

* **The onset-based beat tracker** (`beat_source=tracked`). C++ always uses the file's or host's tempo
  map. That matches Python for grid-aligned or DAW input, which is what the quantized velocity model and
  the timing model are for. Free-time recordings without a meaningful tempo map, such as MAESTRO-style
  performances, need the tracker and the score reconstruction (`timing_features.featurize_performance`).
  Use Python for those files, or port `beat_tracking.py`.
* **Causal-context models** and **performance-conditioned models**. `.hbm` loading rejects both.

Numeric details:

* NumPy 2 sums floats in a SIMD order that can't be reproduced portably. The only feature that sums
  non-integers is `piece_pitch_std`, so C++ computes it with an exact integer formula. Both versions round
  to the same float32 value, and the parity test confirms it on every file tried.
* Random numbers come from SplitMix64 + Box–Muller, implemented the same way in both languages
  (`timing_sampling.PortableRng`). The only differences are last-bit differences between the C runtime's
  and NumPy's `log`, `cos`, `sin`, `exp` and `erfc`, which is why sampled offsets agree to about 1e-13 ms
  rather than exactly.

## 6. Verify

```bash
python tests/test_cpp_parity.py --maestro_files 6          # velocity
python tests/test_timing_cpp_parity.py --maestro_files 3   # timing
```

The velocity test compares 3 synthetic scores (tempo and time-signature changes) plus MAESTRO test files
under both velocity models. Every feature value is identical, raw predictions agree to float32 precision,
there are 0 differing velocities with and without smoothing, and all non-velocity events are preserved.

The timing test runs four option sets on 3 synthetic scores (tempo and time-signature changes, triplets,
chords, re-strikes, drums, pedal) and on quantized versions of MAESTRO test pieces. The option sets are:
median; seed 7; seed 42 with temperature 0.7, scale 1.3 and both sampler overrides; and `--snap` with
seed 3. On every file and option set, quantiles and median offsets differ by 0.0, seeded takes by at most
2.4e-13 ms, and the written MIDI files are identical event for event.
