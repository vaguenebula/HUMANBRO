// HUMANBRO C++ runtime: velocity and timing humanization with models exported by
// export_cpp_model.py.
//
// Typical use (offline, whole clip at once: the features look at past AND future notes):
//
//     humanbro::Humanizer h("velocity.hbm");
//     humanbro::Score score;                 // fill from a MIDI file, a DAW clip, ...
//     std::vector<int> v = h.humanize(score); // one velocity per score.notes[i]
//
//     humanbro::TimingHumanizer t("timing.hbm");
//     humanbro::TimingResult r = t.humanize(score);  // new note-on/off ticks per score.notes[i]
//
// The bar/beat grid comes from the score's own tempo map and time signatures (what a DAW
// host provides). This matches Python's humanize_midi.py with --beat_source midi, and the
// auto mode for grid-aligned input, which is what the quantized model is meant for.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace humanbro {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Note {
    int64_t onset_tick = 0;
    int64_t offset_tick = 0;  // key release; sustain pedal is not applied
    int pitch = 60;
    int velocity = 64;        // input velocity: only used by residual baselines and Options::mix
    int channel = 0;          // MIDI channel 0-15 and source track: when re-timing, notes of the same
    int track = 0;            // (track, channel, pitch) keep their order and never cut each other short
};

struct TempoChange {
    int64_t tick = 0;
    double us_per_quarter = 500000.0;
};

struct TimeSignature {
    int64_t tick = 0;
    int numerator = 4;
    int denominator = 4;
};

// A clip or piece in MIDI tick time. Notes may be in any order; results follow this order.
// Leave drums out: every note is treated as part of one piano texture.
struct Score {
    int ticks_per_quarter = 480;
    std::vector<Note> notes;
    std::vector<TempoChange> tempo_changes;       // empty -> 120 BPM; several events per tick: last wins
    std::vector<TimeSignature> time_signatures;   // empty -> 4/4; assumed to sit on bar lines
    int64_t end_tick = 0;                         // end of the clip; 0 -> last note-off
};

struct Options {
    bool smoothing = false;           // group-level zero-phase EMA that keeps voicing and accents
    double smoothing_alpha = 0.35;
    double smoothing_strength = 0.5;
    double accent_threshold = 8.0;
    double dynamics_scale = 1.0;      // >1 widens, <1 narrows the dynamic range around the median
    double offset = 0.0;              // shift all velocities
    double mix = 1.0;                 // 1 = model only, 0 = input velocities
    bool constant_baseline = false;   // residual models: baseline = base_velocity instead of input dynamics
    double base_velocity = 64.0;
};

// Features in canonical order (onset groups, then pitch), row-major.
struct FeatureTable {
    std::vector<std::string> names;
    std::vector<float> values;
    std::vector<int> note_index;  // canonical row -> index into Score::notes
    std::vector<int> group_id;    // onset group (chord) of each row

    std::size_t rows() const { return note_index.size(); }
    std::size_t cols() const { return names.size(); }
    float at(std::size_t r, std::size_t c) const { return values[r * names.size() + c]; }
};

struct BeatInfo {
    std::size_t beats = 0;
    double median_tempo_bpm = 0.0;
    int beats_per_bar = 4;
    int beat_unit = 4;
};

class Humanizer {
public:
    explicit Humanizer(const std::string& model_path);
    ~Humanizer();
    Humanizer(Humanizer&&) noexcept;
    Humanizer& operator=(Humanizer&&) noexcept;

    // One output velocity (1..127) per score.notes[i].
    std::vector<int> humanize(const Score& score, const Options& options = {}) const;

    // Lower-level steps, e.g. for diagnostics or a custom post-processing chain.
    FeatureTable features(const Score& score, BeatInfo* beat_info = nullptr) const;  // columns = model features
    std::vector<double> predict_raw(const FeatureTable& table) const;                // velocity or delta, canonical order

    const std::vector<std::string>& feature_names() const;
    const std::string& target_mode() const;  // "absolute" or "residual"
    bool quantizes_input() const;
    std::size_t num_trees() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Micro-timing humanization with a quantile timing model (train_timing.py), the C++ counterpart
// of humanize_timing.py for quantized input (the grid is the score's own tempo map).
struct TimingOptions {
    bool sample = true;                   // false: every note gets its median offset (deterministic)
    double temperature = 1.0;             // spread of a sampled take: 0 = middle of each distribution
    double scale = 1.0;                   // multiply every final offset
    double max_offset_ms = 150.0;         // clamp
    std::optional<uint64_t> seed;         // same seed -> same take, also in Python; empty -> random
    std::optional<double> chord_coupling;     // override the fitted within-chord correlation (0..1)
    std::optional<double> correlation_beats;  // override the fitted correlation length (beats)
    bool snap = false;                    // first snap each chord to the 1/16 or triplet grid
};

struct TimingResult {
    std::vector<double> offset_ms;   // per score.notes[i]: the onset shift that was applied
    std::vector<int64_t> onset_tick; // new note-on tick per score.notes[i]
    std::vector<int64_t> offset_tick;// new note-off tick (durations kept)
    uint64_t seed = 0;               // seed of a sampled take: pass it back to reproduce it
    BeatInfo grid;                   // the beat grid the offsets were computed on
};

class TimingHumanizer {
public:
    explicit TimingHumanizer(const std::string& model_path);
    ~TimingHumanizer();
    TimingHumanizer(TimingHumanizer&&) noexcept;
    TimingHumanizer& operator=(TimingHumanizer&&) noexcept;

    TimingResult humanize(const Score& score, const TimingOptions& options = {}) const;

    // Lower-level steps (canonical order, as in Humanizer::features).
    FeatureTable features(const Score& score, bool snap = false, BeatInfo* beat_info = nullptr) const;
    // rows x levels, sorted per row (no quantile crossing), offsets in ms.
    std::vector<double> predict_quantiles(const FeatureTable& table) const;

    const std::vector<double>& quantile_levels() const;
    double chord_coupling() const;     // fitted rho
    double correlation_beats() const;  // fitted ell
    std::size_t num_trees() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Minimal Standard MIDI File support (format 0/1). Velocities are written by patching the
// original note-on bytes, so every other event (pedals, tempo, controllers, note-offs) is kept.
class MidiFile {
public:
    static MidiFile load(const std::string& path);
    void save_with_velocities(const std::string& path, const std::vector<int>& velocities) const;
    // Rewrite the file with every note moved to new ticks (TimingResult), optionally with new
    // velocities too (empty = keep). All other events keep their ticks and bytes.
    void save_with_timing(const std::string& path, const std::vector<int64_t>& onset_tick,
                          const std::vector<int64_t>& offset_tick, const std::vector<int>& velocities = {}) const;

    const Score& score() const { return score_; }

private:
    struct Event {
        int64_t tick;
        std::vector<uint8_t> bytes;  // complete message: status + data, FF type len data, or F0/F7 len data
        uint8_t order_class;         // same-tick order when rewriting: 0 meta, 1 release, 2 other, 3 note-on
    };
    std::vector<uint8_t> bytes_;
    uint16_t format_ = 1;
    uint16_t division_ = 480;
    std::vector<std::vector<Event>> tracks_;
    std::vector<int64_t> track_end_;         // tick of each track's end-of-track event
    Score score_;                            // non-drum notes in file order
    std::vector<std::size_t> velocity_pos_;  // byte offset of each note's note-on velocity
    std::vector<std::size_t> on_event_;      // index of each note's note-on in its track's events
    std::vector<long> off_event_;            // index of the event that released it (-1: re-strike / track end)
};

}  // namespace humanbro
