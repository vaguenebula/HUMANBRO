#include <algorithm>
#include <cmath>
#include <thread>
#include <unordered_map>

#include "humanbro/humanbro.hpp"
#include "model.hpp"
#include "pipeline.hpp"

namespace humanbro {

using detail::TreeModel;

struct Humanizer::Impl {
    TreeModel model;
};

Humanizer::Humanizer(const std::string& model_path) : impl_(std::make_unique<Impl>()) {
    impl_->model = TreeModel::load(model_path);
    if (impl_->model.config.task != "velocity")
        throw Error(model_path + " is a " + impl_->model.config.task + " model; use TimingHumanizer");
}
Humanizer::~Humanizer() = default;
Humanizer::Humanizer(Humanizer&&) noexcept = default;
Humanizer& Humanizer::operator=(Humanizer&&) noexcept = default;

const std::vector<std::string>& Humanizer::feature_names() const { return impl_->model.features; }
const std::string& Humanizer::target_mode() const { return impl_->model.config.target_mode; }
bool Humanizer::quantizes_input() const { return impl_->model.config.quantize; }
std::size_t Humanizer::num_trees() const { return impl_->model.trees.size(); }

FeatureTable Humanizer::features(const Score& score, BeatInfo* beat_info) const {
    const auto& cfg = impl_->model.config;
    const detail::TempoMap tempo_map = detail::TempoMap::from_score(score);
    const detail::PreparedNotes notes = detail::prepare_notes(score, tempo_map);

    // Beat grid from the score's tempo map / time signatures, extended over every note
    // (beat_tracking.build_beat_grid with source="midi").
    detail::BeatGrid grid = detail::grid_from_score(score, tempo_map, notes.end_tick);
    const double first_onset = *std::min_element(notes.onset.begin(), notes.onset.end());
    const double last_offset = *std::max_element(notes.offset.begin(), notes.offset.end());
    grid = grid.extended(std::min(0.0, first_onset), last_offset + 1.0);

    std::vector<double> onset = notes.onset, offset = notes.offset;
    if (cfg.quantize) detail::quantize_notes(onset, offset, grid, cfg.quantize_subdivisions);
    detail::Features f = detail::extract_features(onset, offset, notes.pitch, grid, cfg);

    std::unordered_map<std::string, std::size_t> col_of;
    for (std::size_t c = 0; c < f.names.size(); ++c) col_of[f.names[c]] = c;
    const auto& wanted = impl_->model.features;
    std::vector<std::size_t> src(wanted.size());
    for (std::size_t c = 0; c < wanted.size(); ++c) {
        auto it = col_of.find(wanted[c]);
        if (it == col_of.end()) throw Error("model feature '" + wanted[c] + "' is not produced by the C++ pipeline");
        src[c] = it->second;
    }

    FeatureTable t;
    t.names = wanted;
    const std::size_t n = f.order.size();
    t.values.resize(n * wanted.size());
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < wanted.size(); ++c) t.values[r * wanted.size() + c] = f.columns[src[c]][r];
    t.note_index.resize(n);
    for (std::size_t r = 0; r < n; ++r) t.note_index[r] = notes.score_index[f.order[r]];
    t.group_id = f.group_id;

    if (beat_info) {
        beat_info->beats = grid.times.size();
        beat_info->median_tempo_bpm = detail::median_of(grid.tempo_bpm);
        beat_info->beats_per_bar = grid.beats_per_bar[grid.beats_per_bar.size() / 2];
        beat_info->beat_unit = grid.beat_unit[grid.beat_unit.size() / 2];
    }
    return t;
}

std::vector<double> Humanizer::predict_raw(const FeatureTable& table) const {
    const TreeModel& m = impl_->model;
    if (table.cols() != m.features.size()) throw Error("feature table does not match the model");
    const std::size_t n = table.rows(), cols = table.cols();
    std::vector<double> out(n);
    auto work = [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) out[r] = m.predict(&table.values[r * cols]);
    };
    const std::size_t threads = std::min<std::size_t>(std::max(1u, std::thread::hardware_concurrency()), n / 512 + 1);
    if (threads <= 1) {
        work(0, n);
    } else {
        std::vector<std::thread> pool;
        const std::size_t chunk = (n + threads - 1) / threads;
        for (std::size_t t = 0; t < threads; ++t) pool.emplace_back(work, t * chunk, std::min(n, (t + 1) * chunk));
        for (auto& th : pool) th.join();
    }
    return out;
}

std::vector<int> Humanizer::humanize(const Score& score, const Options& opt) const {
    if (opt.mix < 0.0 || opt.mix > 1.0) throw Error("mix must be between 0 and 1");
    const FeatureTable table = features(score);
    const std::vector<double> raw = predict_raw(table);
    const std::size_t n = table.rows();

    std::vector<double> in_vel(n);
    for (std::size_t r = 0; r < n; ++r) in_vel[r] = score.notes[table.note_index[r]].velocity;

    std::vector<double> vel = raw;
    if (target_mode() == "residual") {
        const auto& cfg = impl_->model.config;
        const std::vector<double> base =
            opt.constant_baseline ? std::vector<double>(n, opt.base_velocity)
                                  : detail::residual_baseline(in_vel, table.group_id, cfg.residual_window_notes,
                                                              cfg.residual_stat);
        for (std::size_t r = 0; r < n; ++r) vel[r] = base[r] + raw[r];
    }
    if (opt.smoothing)
        vel = detail::smooth_velocities(vel, table.group_id, opt.smoothing_alpha, opt.smoothing_strength,
                                        opt.accent_threshold);
    vel = detail::shape_dynamics(vel, opt.dynamics_scale, opt.offset);
    for (std::size_t r = 0; r < n; ++r) vel[r] = opt.mix * vel[r] + (1.0 - opt.mix) * in_vel[r];

    const std::vector<int> clipped = detail::clip_velocities(vel);
    std::vector<int> out(score.notes.size(), 64);
    for (std::size_t r = 0; r < n; ++r) out[table.note_index[r]] = clipped[r];
    return out;
}

}  // namespace humanbro
