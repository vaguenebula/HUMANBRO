// Port of the Python feature pipeline (midi_io.TempoMap, beat_tracking.grid_from_midi /
// BeatGrid, midi_features.quantize_notes / extract_features / residual_baseline and
// postprocess). Arithmetic mirrors numpy operation by operation so features match the
// Python implementation exactly; tests/test_cpp_parity.py checks this.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "humanbro/humanbro.hpp"
#include "model.hpp"

namespace humanbro::detail {

struct TempoMap {
    int ticks_per_quarter = 480;
    std::vector<int64_t> ticks;   // sorted, starts at 0
    std::vector<double> tempos;   // microseconds per quarter
    std::vector<double> seconds;  // absolute time of each change

    static TempoMap from_score(const Score& score);
    double to_seconds(double tick) const;
    double to_ticks(double seconds) const;  // inverse of to_seconds (midi_io.TempoMap.seconds_to_ticks)
};

struct BeatGrid {
    std::vector<double> times;
    std::vector<int> beat_in_bar;
    std::vector<int> beats_per_bar;
    std::vector<int> beat_unit;
    std::vector<int> measure_index;  // derived
    std::vector<double> tempo_bpm;   // derived

    void finalize();
    double time_to_beat(double t) const;
    double beat_to_time(double b) const;
    BeatGrid extended(double t_start, double t_end) const;
};

// Notes as Python's NoteArray: sorted by (onset_tick, pitch), stable w.r.t. Score order.
struct PreparedNotes {
    std::vector<int> score_index;
    std::vector<double> onset;
    std::vector<double> offset;
    std::vector<int> pitch;
    std::vector<double> velocity;
    int64_t end_tick = 0;
};

PreparedNotes prepare_notes(const Score& score, const TempoMap& tempo_map);
BeatGrid grid_from_score(const Score& score, const TempoMap& tempo_map, int64_t end_tick);
void quantize_notes(std::vector<double>& onset, std::vector<double>& offset, const BeatGrid& grid,
                    const std::vector<int>& subdivisions);

struct Features {
    std::vector<std::string> names;
    std::vector<std::vector<float>> columns;  // one per name, canonical row order
    std::vector<int> order;                   // canonical row -> index into the input arrays
    std::vector<int> group_id;
};

Features extract_features(const std::vector<double>& onset, const std::vector<double>& offset,
                          const std::vector<int>& pitch, const BeatGrid& grid, const PipelineConfig& cfg);

std::vector<double> residual_baseline(const std::vector<double>& velocity, const std::vector<int>& group_id,
                                      int window, const std::string& stat);
std::vector<double> smooth_velocities(const std::vector<double>& velocity, const std::vector<int>& group_id,
                                      double alpha, double strength, double accent_threshold);
std::vector<double> shape_dynamics(const std::vector<double>& velocity, double scale, double offset);
std::vector<int> clip_velocities(const std::vector<double>& velocity);
double median_of(std::vector<double> values);

}  // namespace humanbro::detail
