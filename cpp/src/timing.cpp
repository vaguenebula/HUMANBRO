// Port of the timing inference path for quantized input:
//   timing_features.featurize_quantized (+ snap_groups / snap_beats)   -> TimingHumanizer::features
//   TimingModel.predict_quantiles                                       -> predict_quantiles
//   timing_sampling (PortableRng, copula, recalibration, quantile fn)  -> sample_offsets / median
//   humanize_timing.retime + midi_io.with_timing (_same_key_consistent) -> humanize
// Operations follow the numpy code step by step; tests/test_cpp_parity.py compares the two.
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>

#include "humanbro/humanbro.hpp"
#include "model.hpp"
#include "pipeline.hpp"

namespace humanbro {
namespace detail {
namespace {

constexpr double kPi = 3.141592653589793;   // numpy.pi
constexpr double kSqrt1_2 = 0.7071067811865476;
constexpr double kZMax = 2.5;              // timing_sampling.Z_MAX
constexpr double kZClip = 3.5;             // quantile_function clip

// timing_sampling.PortableRng: SplitMix64 uniforms, Box-Muller normals in (cos, sin) pairs.
class PortableRng {
public:
    explicit PortableRng(uint64_t seed) : seed_(seed) {}

    double uniform() {
        ++count_;
        uint64_t z = seed_ + count_ * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return (static_cast<double>(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

    std::vector<double> normals(std::size_t n) {
        const std::size_t m = (n + 1) / 2;
        std::vector<double> out(2 * m);
        for (std::size_t j = 0; j < m; ++j) {
            const double u1 = uniform(), u2 = uniform();
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double t = 2.0 * kPi * u2;
            out[2 * j] = r * std::cos(t);
            out[2 * j + 1] = r * std::sin(t);
        }
        out.resize(n);
        return out;
    }

private:
    uint64_t seed_;
    uint64_t count_ = 0;
};

// numpy.interp for increasing xp, clamped to fp's ends outside the range.
double np_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    const std::size_t n = xp.size();
    if (std::isnan(x)) return x;
    if (x < xp[0]) return fp[0];
    if (x > xp[n - 1]) return fp[n - 1];
    const std::size_t j = static_cast<std::size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
    if (j == n - 1 || xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

// scipy.special.ndtr (Cephes formula).
double ndtr(double a) {
    if (std::isnan(a)) return a;
    const double x = a * kSqrt1_2;
    const double z = std::fabs(x);
    if (z < kSqrt1_2) return 0.5 + 0.5 * std::erf(x);
    const double y = 0.5 * std::erfc(z);
    return x > 0 ? 1.0 - y : y;
}

// timing_sampling.quantile_function for one sorted row.
double quantile_function(const double* q, const std::vector<double>& knots, double z) {
    z = std::clamp(z, -kZClip, kZClip);
    const long last = static_cast<long>(knots.size()) - 2;
    long k = static_cast<long>(std::lower_bound(knots.begin(), knots.end(), z) - knots.begin()) - 1;
    k = std::clamp<long>(k, 0, last);
    const double lo = q[k], hi = q[k + 1];
    const double t = (z - knots[k]) / (knots[k + 1] - knots[k]);
    return lo + t * (hi - lo);
}

// timing_sampling.sample_latent: canonical rows, groups 0..G-1 non-decreasing.
std::vector<double> sample_latent(const std::vector<int>& g, const std::vector<double>& group_beat, double rho,
                                  double ell, PortableRng& rng) {
    const std::size_t n = g.size();
    const std::size_t n_groups = n ? static_cast<std::size_t>(*std::max_element(g.begin(), g.end())) + 1 : 0;
    std::vector<double> beat(n_groups, 0.0);
    for (std::size_t r = 0; r < n; ++r) beat[static_cast<std::size_t>(g[r])] = group_beat[r];
    std::vector<double> phi(n_groups);
    const double denom = std::max(ell, 1e-6);
    for (std::size_t k = 0; k < n_groups; ++k) {
        const double dt = k ? beat[k] - beat[k - 1] : 0.0;  // numpy.diff(..., prepend=beat[0])
        phi[k] = std::exp(-std::max(dt, 0.0) / denom);
    }
    const std::vector<double> eta = rng.normals(n_groups);
    std::vector<double> shared(n_groups);
    double acc = n_groups ? eta[0] : 0.0;
    for (std::size_t k = 0; k < n_groups; ++k) {
        if (k) acc = phi[k] * acc + std::sqrt(1.0 - phi[k] * phi[k]) * eta[k];
        shared[k] = acc;
    }
    rho = std::clamp(rho, 0.0, 1.0);
    const double a = std::sqrt(rho), b = std::sqrt(1.0 - rho);
    const std::vector<double> e = rng.normals(n);
    std::vector<double> z(n);
    for (std::size_t r = 0; r < n; ++r) z[r] = a * shared[static_cast<std::size_t>(g[r])] + b * e[r];
    return z;
}

// timing_features.snap_beats: one subdivision per beat (integer part of the position).
std::vector<double> snap_beats(const std::vector<double>& b, const std::vector<int>& subs, double switch_cost) {
    const std::size_t n = b.size();
    std::vector<std::vector<double>> cands(subs.size(), std::vector<double>(n));
    for (std::size_t j = 0; j < subs.size(); ++j)
        for (std::size_t i = 0; i < n; ++i) cands[j][i] = std::nearbyint(b[i] * subs[j]) / subs[j];
    if (subs.size() == 1 || n == 0) return cands[0];
    std::map<double, std::size_t> beat_id;  // numpy.unique(floor(b), return_inverse=True)
    for (double x : b) beat_id.emplace(std::floor(x), 0);
    std::size_t next = 0;
    for (auto& kv : beat_id) kv.second = next++;
    std::vector<std::size_t> inv(n);
    for (std::size_t i = 0; i < n; ++i) inv[i] = beat_id[std::floor(b[i])];
    auto sq_err = [&](std::size_t j) {
        std::vector<double> err(beat_id.size(), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = cands[j][i] - b[i];
            err[inv[i]] += d * d;
        }
        return err;
    };
    std::vector<std::size_t> choice(beat_id.size(), 0);
    std::vector<double> best = sq_err(0);
    for (std::size_t j = 1; j < subs.size(); ++j) {
        std::vector<double> err = sq_err(j);
        for (std::size_t k = 0; k < err.size(); ++k) {
            err[k] += switch_cost;
            if (err[k] < best[k]) {
                choice[k] = j;
                best[k] = err[k];
            }
        }
    }
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = cands[choice[inv[i]]][i];
    return out;
}

// timing_features.snap_groups: each onset group (anchored chord tolerance) snapped at its mean onset.
std::vector<double> snap_groups(const std::vector<double>& onset, const BeatGrid& grid, const std::vector<int>& subs,
                                double switch_cost, double chord_tol) {
    const std::size_t n = onset.size();
    std::vector<std::size_t> by_time(n);
    std::iota(by_time.begin(), by_time.end(), 0);
    std::stable_sort(by_time.begin(), by_time.end(), [&](std::size_t a, std::size_t b) { return onset[a] < onset[b]; });
    std::vector<std::size_t> gid(n);
    std::size_t cur = 0;
    double start = n ? onset[by_time[0]] : 0.0;
    for (std::size_t i : by_time) {
        if (onset[i] - start > chord_tol) {
            ++cur;
            start = onset[i];
        }
        gid[i] = cur;
    }
    const std::size_t groups = n ? cur + 1 : 0;
    std::vector<double> sum(groups, 0.0), count(groups, 0.0);
    for (std::size_t i = 0; i < n; ++i) {  // numpy.bincount: index order
        sum[gid[i]] += onset[i];
        count[gid[i]] += 1.0;
    }
    std::vector<double> b(groups);
    for (std::size_t k = 0; k < groups; ++k) b[k] = grid.time_to_beat(sum[k] / count[k]);
    const std::vector<double> snapped = snap_beats(b, subs, switch_cost);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = snapped[gid[i]];
    return out;
}

// midi_io._same_key_consistent: per key, in note order, strictly increasing onsets and each
// release no later than the next strike of its key.
void same_key_consistent(const std::vector<int64_t>& key, std::vector<double>& on, std::vector<double>& off) {
    const std::size_t n = key.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return key[a] < key[b]; });
    std::size_t s = 0;
    while (s < n) {
        std::size_t e = s;
        while (e < n && key[order[e]] == key[order[s]]) ++e;
        double run = -std::numeric_limits<double>::infinity();
        for (std::size_t p = s; p < e; ++p) {  // running max of (on - j), then + j
            const double j = static_cast<double>(p - s);
            run = std::max(run, on[order[p]] - j);
            on[order[p]] = run + j;
        }
        for (std::size_t p = s; p < e; ++p) {
            const double nxt = p + 1 < e ? on[order[p + 1]] : std::numeric_limits<double>::infinity();
            off[order[p]] = std::min(std::max(off[order[p]], on[order[p]] + 1.0), nxt);
        }
        s = e;
    }
}

}  // namespace

BeatInfo beat_info(const BeatGrid& g) {
    BeatInfo b;
    b.beats = g.times.size();
    b.median_tempo_bpm = median_of(g.tempo_bpm);
    b.beats_per_bar = g.beats_per_bar[g.beats_per_bar.size() / 2];
    b.beat_unit = g.beat_unit[g.beat_unit.size() / 2];
    return b;
}

struct TimingPrepared {
    TempoMap tempo_map;
    PreparedNotes notes;
    BeatGrid grid;
    std::vector<double> onset;      // score time per prepared note (snapped if requested)
    FeatureTable table;             // model columns, canonical order
    std::vector<int> row_note;      // canonical row -> prepared-note index
    std::vector<double> onset_beat; // per canonical row (float32 feature value), for the copula
};

}  // namespace detail

using detail::TreeModel;

struct TimingHumanizer::Impl {
    TreeModel model;

    detail::TimingPrepared prepare(const Score& score, bool snap) const {
        const auto& cfg = model.config;
        detail::TimingPrepared p;
        p.tempo_map = detail::TempoMap::from_score(score);
        p.notes = detail::prepare_notes(score, p.tempo_map);
        const auto& nt = p.notes;
        p.grid = detail::grid_from_score(score, p.tempo_map, nt.end_tick);
        const double first_onset = *std::min_element(nt.onset.begin(), nt.onset.end());
        const double last_offset = *std::max_element(nt.offset.begin(), nt.offset.end());
        p.grid = p.grid.extended(std::min(0.0, first_onset), last_offset + 1.0);

        p.onset = nt.onset;
        std::vector<double> offset = nt.offset;
        if (snap) {
            const auto& subs = cfg.timing_subdivisions;
            const int finest = *std::max_element(subs.begin(), subs.end());
            const std::vector<double> on_b =
                detail::snap_groups(p.onset, p.grid, subs, cfg.timing_switch_cost, cfg.chord_tolerance_s);
            std::vector<double> rel(offset.size());
            for (std::size_t i = 0; i < offset.size(); ++i) rel[i] = p.grid.time_to_beat(offset[i]);
            const std::vector<double> off_b = detail::snap_beats(rel, subs, cfg.timing_switch_cost);
            for (std::size_t i = 0; i < offset.size(); ++i) {
                p.onset[i] = p.grid.beat_to_time(on_b[i]);
                offset[i] = p.grid.beat_to_time(std::max(off_b[i], on_b[i] + 1.0 / finest));
            }
        }
        detail::Features f = detail::extract_features(p.onset, offset, nt.pitch, p.grid, cfg);

        std::unordered_map<std::string, std::size_t> col_of;
        for (std::size_t c = 0; c < f.names.size(); ++c) col_of[f.names[c]] = c;
        auto column = [&](const std::string& name) {
            auto it = col_of.find(name);
            if (it == col_of.end()) throw Error("model feature '" + name + "' is not produced by the C++ pipeline");
            return it->second;
        };
        const auto& wanted = model.features;
        std::vector<std::size_t> src(wanted.size());
        for (std::size_t c = 0; c < wanted.size(); ++c) src[c] = column(wanted[c]);

        const std::size_t n = f.order.size();
        FeatureTable& t = p.table;
        t.names = wanted;
        t.values.resize(n * wanted.size());
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < wanted.size(); ++c) t.values[r * wanted.size() + c] = f.columns[src[c]][r];
        t.note_index.resize(n);
        p.row_note = f.order;
        for (std::size_t r = 0; r < n; ++r) t.note_index[r] = nt.score_index[f.order[r]];
        t.group_id = f.group_id;
        const auto& ob = f.columns[column("onset_beat")];
        p.onset_beat.assign(ob.begin(), ob.end());
        return p;
    }
};

TimingHumanizer::TimingHumanizer(const std::string& model_path) : impl_(std::make_unique<Impl>()) {
    impl_->model = TreeModel::load(model_path);
    if (impl_->model.config.task != "timing")
        throw Error(model_path + " is a " + impl_->model.config.task + " model; use Humanizer");
}
TimingHumanizer::~TimingHumanizer() = default;
TimingHumanizer::TimingHumanizer(TimingHumanizer&&) noexcept = default;
TimingHumanizer& TimingHumanizer::operator=(TimingHumanizer&&) noexcept = default;

const std::vector<double>& TimingHumanizer::quantile_levels() const { return impl_->model.sampler.levels; }
double TimingHumanizer::chord_coupling() const { return impl_->model.sampler.rho; }
double TimingHumanizer::correlation_beats() const { return impl_->model.sampler.ell; }
std::size_t TimingHumanizer::num_trees() const { return impl_->model.trees.size(); }

FeatureTable TimingHumanizer::features(const Score& score, bool snap, BeatInfo* beat_info) const {
    detail::TimingPrepared p = impl_->prepare(score, snap);
    if (beat_info) *beat_info = detail::beat_info(p.grid);
    return std::move(p.table);
}

std::vector<double> TimingHumanizer::predict_quantiles(const FeatureTable& table) const {
    const TreeModel& m = impl_->model;
    if (table.cols() != m.features.size()) throw Error("feature table does not match the model");
    const std::size_t n = table.rows(), cols = table.cols(), k = static_cast<std::size_t>(m.num_outputs);
    std::vector<double> out(n * k);
    auto work = [&](std::size_t begin, std::size_t end) {
        std::vector<float> acc(k);
        for (std::size_t r = begin; r < end; ++r) {
            m.predict(&table.values[r * cols], acc.data());
            double* row = &out[r * k];
            for (std::size_t j = 0; j < k; ++j) row[j] = acc[j];
            std::sort(row, row + k);  // timing_sampling.sort_quantiles
        }
    };
    const std::size_t threads = std::min<std::size_t>(std::max(1u, std::thread::hardware_concurrency()), n / 64 + 1);
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

TimingResult TimingHumanizer::humanize(const Score& score, const TimingOptions& opt) const {
    if (opt.temperature < 0 || opt.scale < 0) throw Error("temperature and scale must be >= 0");
    const TreeModel& m = impl_->model;
    const detail::SamplerParams& sp = m.sampler;
    detail::TimingPrepared p = impl_->prepare(score, opt.snap);
    const std::vector<double> q = predict_quantiles(p.table);
    const std::size_t n = p.table.rows(), k = sp.levels.size();

    TimingResult res;
    res.grid = detail::beat_info(p.grid);
    std::vector<double> ms(n);
    if (!opt.sample) {
        for (std::size_t r = 0; r < n; ++r) ms[r] = detail::quantile_function(&q[r * k], sp.knots, 0.0);
    } else {
        if (opt.seed) {
            res.seed = *opt.seed;
        } else {
            std::random_device rd;
            res.seed = ((uint64_t(rd()) << 32) | rd()) & 0x7FFFFFFFFFFFFFFFULL;
        }
        const double rho = opt.chord_coupling ? std::clamp(*opt.chord_coupling, 0.0, 1.0) : sp.rho;
        const double ell = opt.correlation_beats ? std::max(*opt.correlation_beats, 1e-3) : sp.ell;
        detail::PortableRng rng(res.seed);
        const std::vector<double> z = detail::sample_latent(p.table.group_id, p.onset_beat, rho, ell, rng);
        for (std::size_t r = 0; r < n; ++r) {
            const double zc = std::clamp(opt.temperature * z[r], -detail::kZMax, detail::kZMax);
            const double zm = detail::np_interp(detail::ndtr(zc), sp.calib_u, sp.calib_z);  // recalibration
            ms[r] = detail::quantile_function(&q[r * k], sp.knots, zm);
        }
    }
    for (double& v : ms) v = std::clamp(opt.scale * v, -opt.max_offset_ms, opt.max_offset_ms);

    // Re-time (humanize_timing.retime + midi_io.with_timing), in prepared-note order.
    const auto& nt = p.notes;
    const std::size_t notes = nt.onset.size();
    std::vector<double> shift(notes, 0.0);
    for (std::size_t r = 0; r < n; ++r) shift[static_cast<std::size_t>(p.row_note[r])] = ms[r];
    std::vector<double> on(notes), off(notes);
    std::vector<int64_t> key(notes);
    for (std::size_t i = 0; i < notes; ++i) {
        const double new_on = p.onset[i] + shift[i] / 1000.0;
        on[i] = std::max(std::nearbyint(p.tempo_map.to_ticks(new_on)), 0.0);
        off[i] = std::nearbyint(p.tempo_map.to_ticks(new_on + (nt.offset[i] - nt.onset[i])));
        off[i] = std::max(off[i], on[i] + 1.0);
        const Note& src = score.notes[static_cast<std::size_t>(nt.score_index[i])];
        key[i] = (static_cast<int64_t>(src.track) * 16 + src.channel) * 128 + src.pitch;
    }
    detail::same_key_consistent(key, on, off);

    res.offset_ms.assign(score.notes.size(), 0.0);
    res.onset_tick.resize(score.notes.size());
    res.offset_tick.resize(score.notes.size());
    for (std::size_t i = 0; i < notes; ++i) {
        const auto s = static_cast<std::size_t>(nt.score_index[i]);
        res.offset_ms[s] = shift[i];
        res.onset_tick[s] = static_cast<int64_t>(on[i]);
        res.offset_tick[s] = static_cast<int64_t>(off[i]);
    }
    return res;
}

}  // namespace humanbro
