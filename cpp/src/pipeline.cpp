#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>

namespace humanbro::detail {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr std::size_t kMaxGridBeats = 500000;
constexpr double kDefaultTempo = 500000.0;

int py_mod(long long a, long long m) {
    const long long r = a % m;
    return static_cast<int>(r < 0 ? r + m : r);
}

// numpy.searchsorted(side="left" / "right")
template <class V>
long lower_idx(const std::vector<V>& v, double x) {
    return static_cast<long>(std::lower_bound(v.begin(), v.end(), x) - v.begin());
}
template <class V>
long upper_idx(const std::vector<V>& v, double x) {
    return static_cast<long>(std::upper_bound(v.begin(), v.end(), x) - v.begin());
}

double rint_even(double x) { return std::nearbyint(x); }  // numpy.rint: round half to even

std::vector<double> running_median5(const std::vector<double>& x) {
    const std::size_t n = x.size();
    if (n <= 1) return x;
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        double w[5];
        for (int k = -2; k <= 2; ++k) {
            const long j = std::clamp<long>(static_cast<long>(i) + k, 0, static_cast<long>(n) - 1);
            w[k + 2] = x[static_cast<std::size_t>(j)];
        }
        std::sort(w, w + 5);
        out[i] = w[2];
    }
    return out;
}

std::string fmt_g(double x) {  // Python f"{x:g}".replace(".", "p")
    char buf[64];
    std::snprintf(buf, sizeof buf, "%g", x);
    std::string s(buf);
    std::replace(s.begin(), s.end(), '.', 'p');
    return s;
}

}  // namespace

double median_of(std::vector<double> v) {
    if (v.empty()) return kNaN;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// ---------------------------------------------------------------------------
// Tempo map and beat grid
// ---------------------------------------------------------------------------

TempoMap TempoMap::from_score(const Score& score) {
    if (score.ticks_per_quarter <= 0) throw Error("ticks_per_quarter must be positive");
    std::vector<TempoChange> changes = score.tempo_changes;
    std::stable_sort(changes.begin(), changes.end(),
                     [](const TempoChange& a, const TempoChange& b) { return a.tick < b.tick; });
    std::map<int64_t, double> by_tick;  // later events on the same tick win
    for (const auto& c : changes)
        if (c.us_per_quarter > 0) by_tick[c.tick] = c.us_per_quarter;
    by_tick.emplace(0, kDefaultTempo);

    TempoMap m;
    m.ticks_per_quarter = score.ticks_per_quarter;
    for (const auto& [tick, tempo] : by_tick) {
        m.ticks.push_back(tick);
        m.tempos.push_back(tempo);
    }
    m.seconds.assign(m.ticks.size(), 0.0);
    for (std::size_t i = 1; i < m.ticks.size(); ++i) {
        const double sec_per_tick = m.tempos[i - 1] / 1e6 / m.ticks_per_quarter;
        m.seconds[i] = m.seconds[i - 1] + static_cast<double>(m.ticks[i] - m.ticks[i - 1]) * sec_per_tick;
    }
    return m;
}

double TempoMap::to_seconds(double tick) const {
    long i = upper_idx(ticks, tick) - 1;
    i = std::clamp<long>(i, 0, static_cast<long>(ticks.size()) - 1);
    return seconds[i] + (tick - static_cast<double>(ticks[i])) * tempos[i] / 1e6 / ticks_per_quarter;
}

double TempoMap::to_ticks(double s) const {
    long i = upper_idx(seconds, s) - 1;
    i = std::clamp<long>(i, 0, static_cast<long>(ticks.size()) - 1);
    return static_cast<double>(ticks[i]) + (s - seconds[i]) * 1e6 * ticks_per_quarter / tempos[i];
}

void BeatGrid::finalize() {
    const std::size_t n = times.size();
    if (n < 2) throw Error("beat grid needs at least two beats");
    for (std::size_t i = 1; i < n; ++i)
        if (!(times[i] > times[i - 1])) throw Error("beat times must be strictly increasing");
    measure_index.resize(n);
    int count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        count += beat_in_bar[i] == 0;
        measure_index[i] = count;
    }
    std::vector<double> ibi(n);
    for (std::size_t i = 0; i + 1 < n; ++i) ibi[i] = times[i + 1] - times[i];
    ibi[n - 1] = ibi[n - 2];
    const std::vector<double> med = running_median5(ibi);
    tempo_bpm.resize(n);
    for (std::size_t i = 0; i < n; ++i) tempo_bpm[i] = 60.0 / med[i];
}

// numpy.interp(t, times, arange(n)) with linear extrapolation outside the grid.
double BeatGrid::time_to_beat(double t) const {
    const std::size_t n = times.size();
    if (t < times[0]) return (t - times[0]) / (times[1] - times[0]);
    if (t > times[n - 1]) return static_cast<double>(n - 1) + (t - times[n - 1]) / (times[n - 1] - times[n - 2]);
    const long j = upper_idx(times, t) - 1;
    if (j == static_cast<long>(n) - 1) return static_cast<double>(n - 1);
    if (times[j] == t) return static_cast<double>(j);
    const double slope = (static_cast<double>(j + 1) - static_cast<double>(j)) / (times[j + 1] - times[j]);
    return slope * (t - times[j]) + static_cast<double>(j);
}

// numpy.interp(b, arange(n), times) with linear extrapolation outside the grid.
double BeatGrid::beat_to_time(double b) const {
    const std::size_t n = times.size();
    if (b < 0) return times[0] + b * (times[1] - times[0]);
    if (b > static_cast<double>(n - 1)) return times[n - 1] + (b - static_cast<double>(n - 1)) * (times[n - 1] - times[n - 2]);
    const long j = static_cast<long>(std::floor(b));
    if (j == static_cast<long>(n) - 1) return times[n - 1];
    if (static_cast<double>(j) == b) return times[j];
    const double slope = (times[j + 1] - times[j]) / (static_cast<double>(j + 1) - static_cast<double>(j));
    return slope * (b - static_cast<double>(j)) + times[j];
}

BeatGrid BeatGrid::extended(double t_start, double t_end) const {
    auto edge_ibi = [](const std::vector<double>& t, std::size_t from, std::size_t to) {
        std::vector<double> d;
        for (std::size_t i = from + 1; i < to; ++i) d.push_back(t[i] - t[i - 1]);
        return std::max(median_of(d), 0.05);
    };
    const std::size_t n = times.size();
    const double ibi0 = edge_ibi(times, 0, std::min<std::size_t>(5, n));
    const double ibi1 = edge_ibi(times, n - std::min<std::size_t>(5, n), n);

    std::vector<double> pre_t;
    std::vector<int> pre_b;
    double t = times[0];
    int b = beat_in_bar[0];
    while (t > t_start && pre_t.size() < kMaxGridBeats) {
        t -= ibi0;
        b = py_mod(b - 1, beats_per_bar[0]);
        pre_t.push_back(t);
        pre_b.push_back(b);
    }
    BeatGrid g;
    const std::size_t k = pre_t.size();
    g.times.assign(pre_t.rbegin(), pre_t.rend());
    g.beat_in_bar.assign(pre_b.rbegin(), pre_b.rend());
    g.beats_per_bar.assign(k, beats_per_bar[0]);
    g.beat_unit.assign(k, beat_unit[0]);
    g.times.insert(g.times.end(), times.begin(), times.end());
    g.beat_in_bar.insert(g.beat_in_bar.end(), beat_in_bar.begin(), beat_in_bar.end());
    g.beats_per_bar.insert(g.beats_per_bar.end(), beats_per_bar.begin(), beats_per_bar.end());
    g.beat_unit.insert(g.beat_unit.end(), beat_unit.begin(), beat_unit.end());
    while (g.times.back() < t_end && g.times.size() < kMaxGridBeats) {
        g.times.push_back(g.times.back() + ibi1);
        g.beat_in_bar.push_back(py_mod(g.beat_in_bar.back() + 1, g.beats_per_bar.back()));
        g.beats_per_bar.push_back(g.beats_per_bar.back());
        g.beat_unit.push_back(g.beat_unit.back());
    }
    g.finalize();
    return g;
}

PreparedNotes prepare_notes(const Score& score, const TempoMap& tm) {
    const std::size_t n = score.notes.size();
    if (n == 0) throw Error("no notes");
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        const Note &x = score.notes[a], &y = score.notes[b];
        return x.onset_tick != y.onset_tick ? x.onset_tick < y.onset_tick : x.pitch < y.pitch;
    });
    PreparedNotes p;
    p.end_tick = score.end_tick;
    for (int i : idx) {
        const Note& note = score.notes[i];
        if (note.pitch < 0 || note.pitch > 127) throw Error("note pitch out of range 0..127");
        const int64_t off = std::max(note.offset_tick, note.onset_tick);
        p.score_index.push_back(i);
        p.onset.push_back(tm.to_seconds(static_cast<double>(note.onset_tick)));
        p.offset.push_back(tm.to_seconds(static_cast<double>(off)));
        p.pitch.push_back(note.pitch);
        p.velocity.push_back(static_cast<double>(note.velocity));
        p.end_tick = std::max(p.end_tick, off);
    }
    return p;
}

BeatGrid grid_from_score(const Score& score, const TempoMap& tm, int64_t end_tick) {
    std::vector<TimeSignature> sigs = score.time_signatures;
    std::stable_sort(sigs.begin(), sigs.end(),
                     [](const TimeSignature& a, const TimeSignature& b) { return a.tick < b.tick; });
    std::map<int64_t, std::pair<int, int>> by_tick;
    for (const auto& s : sigs)
        if (s.numerator > 0 && s.denominator > 0) by_tick[s.tick] = {s.numerator, s.denominator};
    by_tick.emplace(0, std::make_pair(4, 4));
    std::vector<std::tuple<int64_t, int, int>> ts;
    for (const auto& [tick, nd] : by_tick) ts.emplace_back(tick, nd.first, nd.second);

    const double tpq = score.ticks_per_quarter;
    std::vector<double> beat_ticks;
    BeatGrid g;
    for (std::size_t i = 0; i < ts.size(); ++i) {
        const auto [tick0, num, den] = ts[i];
        const double step = tpq * 4.0 / den;
        const double seg_end = i + 1 < ts.size() ? static_cast<double>(std::get<0>(ts[i + 1]))
                                                 : static_cast<double>(end_tick) + num * step;
        long long k = 0;
        double x = static_cast<double>(tick0);
        while (x < seg_end - 1e-9) {
            beat_ticks.push_back(x);
            g.beat_in_bar.push_back(py_mod(k, num));
            g.beats_per_bar.push_back(num);
            g.beat_unit.push_back(den);
            ++k;
            x = static_cast<double>(tick0) + static_cast<double>(k) * step;
            if (beat_ticks.size() > kMaxGridBeats) throw Error("tempo/time-signature grid is implausibly long");
        }
    }
    if (beat_ticks.size() < 2) {
        beat_ticks = {0.0, tpq};
        g.beat_in_bar = {0, 1};
        g.beats_per_bar = {4, 4};
        g.beat_unit = {4, 4};
    }
    for (double bt : beat_ticks) g.times.push_back(tm.to_seconds(bt));
    g.finalize();
    return g;
}

void quantize_notes(std::vector<double>& onset, std::vector<double>& offset, const BeatGrid& grid,
                    const std::vector<int>& subdivisions) {
    const int finest = *std::max_element(subdivisions.begin(), subdivisions.end());
    auto snap = [&](double b) {
        double best = kNaN, best_err = kInf;
        for (int s : subdivisions) {
            const double q = rint_even(b * s) / s;
            const double err = std::fabs(q - b);
            if (err < best_err) {
                best = q;
                best_err = err;
            }
        }
        return best;
    };
    for (std::size_t i = 0; i < onset.size(); ++i) {
        const double ob = snap(grid.time_to_beat(onset[i]));
        const double eb = std::max(snap(grid.time_to_beat(offset[i])), ob + 1.0 / finest);
        onset[i] = grid.beat_to_time(ob);
        offset[i] = grid.beat_to_time(eb);
    }
}

// ---------------------------------------------------------------------------
// Feature extraction (midi_features.extract_features, bidirectional context)
// ---------------------------------------------------------------------------

Features extract_features(const std::vector<double>& onset_in, const std::vector<double>& offset_in,
                          const std::vector<int>& pitch_in, const BeatGrid& grid, const PipelineConfig& cfg) {
    const std::size_t n = onset_in.size();
    if (n == 0) throw Error("no notes");
    std::vector<double> offset_c(n);
    for (std::size_t i = 0; i < n; ++i) offset_c[i] = std::max(offset_in[i], onset_in[i]);

    // --- onset groups (chords) and canonical order (group, pitch) ---------
    std::vector<int> by_time(n);
    std::iota(by_time.begin(), by_time.end(), 0);
    std::stable_sort(by_time.begin(), by_time.end(), [&](int a, int b) {
        return onset_in[a] != onset_in[b] ? onset_in[a] < onset_in[b] : pitch_in[a] < pitch_in[b];
    });
    std::vector<long> gid(n);
    {
        long cur = 0;
        double start = onset_in[by_time[0]];
        for (int k : by_time) {
            if (onset_in[k] - start > cfg.chord_tolerance_s) {
                ++cur;
                start = onset_in[k];
            }
            gid[k] = cur;
        }
    }
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return gid[a] != gid[b] ? gid[a] < gid[b] : pitch_in[a] < pitch_in[b];
    });

    std::vector<double> on(n), off(n), dur(n);
    std::vector<long> p(n), g(n);
    for (std::size_t i = 0; i < n; ++i) {
        on[i] = onset_in[order[i]];
        off[i] = offset_c[order[i]];
        p[i] = pitch_in[order[i]];
        g[i] = gid[order[i]];
        dur[i] = off[i] - on[i];
    }
    const long G = g[n - 1] + 1;
    std::vector<long> g_start(G), g_end(G);
    for (long k = 0; k < G; ++k) {
        g_start[k] = lower_idx(g, static_cast<double>(k));
        g_end[k] = upper_idx(g, static_cast<double>(k));
    }
    std::vector<double> g_time(G), g_top(G), g_low(G), g_beat(G);
    std::vector<long> g_size(G);
    for (long k = 0; k < G; ++k) {
        g_time[k] = *std::min_element(on.begin() + g_start[k], on.begin() + g_end[k]);
        g_top[k] = static_cast<double>(p[g_end[k] - 1]);
        g_low[k] = static_cast<double>(p[g_start[k]]);
        g_size[k] = g_end[k] - g_start[k];
        g_beat[k] = grid.time_to_beat(g_time[k]);
    }
    std::vector<double> gt(n), gt_b(n), dur_b(n);
    for (std::size_t i = 0; i < n; ++i) {
        gt[i] = g_time[g[i]];
        gt_b[i] = g_beat[g[i]];
        dur_b[i] = std::max(grid.time_to_beat(off[i]) - grid.time_to_beat(on[i]), 0.0);
    }

    Features F;
    F.order = order;
    F.group_id.assign(g.begin(), g.end());
    auto add = [&](const std::string& name, const std::vector<double>& v) {
        F.names.push_back(name);
        std::vector<float> col(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) col[i] = static_cast<float>(v[i]);
        F.columns.push_back(std::move(col));
    };
    auto column = [&](auto fn) {
        std::vector<double> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = fn(i);
        return v;
    };
    auto near_grid = [](double frac, int s, double tol) {
        const double x = frac * s;
        return std::fabs(x - rint_even(x)) / s <= tol;
    };

    // --- core ------------------------------------------------------------
    const long nb = static_cast<long>(grid.times.size());
    const double tol = cfg.strong_beat_tolerance;
    std::vector<long> bi(n);
    std::vector<double> frac(n), level(n), bpb(n), bim(n), on_beat(n), strong(n), down(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double fl = std::floor(gt_b[i]);
        bi[i] = std::clamp<long>(static_cast<long>(fl), 0, nb - 1);
        frac[i] = gt_b[i] - fl;
        const long rb = std::clamp<long>(static_cast<long>(rint_even(gt_b[i])), 0, nb - 1);
        const bool ob = std::fabs(gt_b[i] - rint_even(gt_b[i])) <= tol;
        const int bib_r = grid.beat_in_bar[rb], bpb_r = grid.beats_per_bar[rb];
        const bool is_down = ob && bib_r == 0;
        const bool is_half = ob && bpb_r >= 4 && bpb_r % 2 == 0 && bib_r == bpb_r / 2;
        double lv = 5;
        if (near_grid(frac[i], 3, tol / 2) || near_grid(frac[i], 4, tol / 2)) lv = 4;
        if (near_grid(frac[i], 2, tol)) lv = 3;
        if (ob) lv = 2;
        if (is_half) lv = 1;
        if (is_down) lv = 0;
        level[i] = lv;
        on_beat[i] = ob;
        strong[i] = is_down || is_half;
        down[i] = is_down;
        bpb[i] = grid.beats_per_bar[bi[i]];
        bim[i] = static_cast<double>(grid.beat_in_bar[bi[i]]) + frac[i];
    }
    add("pitch", column([&](std::size_t i) { return double(p[i]); }));
    add("pitch_class", column([&](std::size_t i) { return double(p[i] % 12); }));
    add("octave", column([&](std::size_t i) { return double(p[i] / 12 - 1); }));
    add("beat_in_measure", bim);
    add("beat_in_measure_norm", column([&](std::size_t i) { return bim[i] / bpb[i]; }));
    add("beat_frac", frac);
    add("metrical_level", level);
    add("on_beat", on_beat);
    add("strong_beat", strong);
    add("is_downbeat", down);

    // --- note-level ---------------------------------------------------------
    add("dur_s", dur);
    add("dur_beats", dur_b);
    add("onset_beat", gt_b);
    add("beat_index", column([&](std::size_t i) { return std::floor(gt_b[i]); }));
    add("local_tempo_bpm", column([&](std::size_t i) { return grid.tempo_bpm[bi[i]]; }));
    add("ts_num", bpb);
    add("ts_den", column([&](std::size_t i) { return double(grid.beat_unit[bi[i]]); }));

    // --- melodic ------------------------------------------------------------
    std::vector<long> key(n);
    for (std::size_t i = 0; i < n; ++i) key[i] = g[i] * 128 + p[i];
    auto nearest_in_group = [&](std::size_t i, long tg) -> long {
        if (tg < 0 || tg >= G) return -1;
        const long q = tg * 128 + p[i];
        const long j = static_cast<long>(std::lower_bound(key.begin(), key.end(), q) - key.begin());
        const long hi = std::clamp<long>(j, 0, static_cast<long>(n) - 1);
        const long lo = std::clamp<long>(j - 1, 0, static_cast<long>(n) - 1);
        const bool ok_hi = g[hi] == tg, ok_lo = g[lo] == tg;
        if (!ok_hi && !ok_lo) return -1;
        const double d_hi = ok_hi ? std::fabs(double(p[hi] - p[i])) : kInf;
        const double d_lo = ok_lo ? std::fabs(double(p[lo] - p[i])) : kInf;
        return d_lo <= d_hi ? lo : hi;
    };
    std::vector<long> prev_i(n), next_i(n);
    for (std::size_t i = 0; i < n; ++i) {
        prev_i[i] = nearest_in_group(i, g[i] - 1);
        next_i[i] = nearest_in_group(i, g[i] + 1);
    }
    auto take = [&](const auto& values, long j) { return j >= 0 ? double(values[j]) : kNaN; };
    std::vector<double> prev_int(n), next_int(n), prev_dur(n);
    for (std::size_t i = 0; i < n; ++i) {
        prev_int[i] = double(p[i]) - take(p, prev_i[i]);
        next_int[i] = take(p, next_i[i]) - double(p[i]);
        prev_dur[i] = take(dur, prev_i[i]);
    }
    auto prev_g = [&](const std::vector<double>& v, std::size_t i) { return g[i] > 0 ? v[g[i] - 1] : kNaN; };
    auto next_g = [&](const std::vector<double>& v, std::size_t i) { return g[i] + 1 < G ? v[g[i] + 1] : kNaN; };
    auto sgn = [](double x) { return std::isnan(x) ? 0.0 : (x > 0) - (x < 0); };  // nan_to_num(sign(x))

    add("prev_interval", prev_int);
    add("abs_prev_interval", column([&](std::size_t i) { return std::fabs(prev_int[i]); }));
    add("ioi_prev_s", column([&](std::size_t i) { return gt[i] - prev_g(g_time, i); }));
    add("ioi_prev_beats", column([&](std::size_t i) { return gt_b[i] - prev_g(g_beat, i); }));
    add("prev_dur_s", prev_dur);
    add("dur_ratio_prev", column([&](std::size_t i) { return std::log2((dur[i] + 0.01) / (prev_dur[i] + 0.01)); }));
    add("next_interval", next_int);
    add("abs_next_interval", column([&](std::size_t i) { return std::fabs(next_int[i]); }));
    add("ioi_next_s", column([&](std::size_t i) { return next_g(g_time, i) - gt[i]; }));
    add("ioi_next_beats", column([&](std::size_t i) { return next_g(g_beat, i) - gt_b[i]; }));
    add("next_dur_s", column([&](std::size_t i) { return take(dur, next_i[i]); }));
    add("melodic_direction", column([&](std::size_t i) { return sgn(prev_int[i]) + sgn(next_int[i]); }));
    add("is_local_peak", column([&](std::size_t i) { return double(prev_int[i] > 0 && next_int[i] < 0); }));
    add("is_local_trough", column([&](std::size_t i) { return double(prev_int[i] < 0 && next_int[i] > 0); }));
    auto top_at = [&](long k) { return (k >= 0 && k < G) ? g_top[k] : kNaN; };
    auto nanmean2 = [](double a, double b) {
        const int c = !std::isnan(a) + !std::isnan(b);
        return c ? ((std::isnan(a) ? 0.0 : a) + (std::isnan(b) ? 0.0 : b)) / c : kNaN;
    };
    add("top_line_trend", column([&](std::size_t i) {
            return nanmean2(top_at(g[i] + 1), top_at(g[i] + 2)) - nanmean2(top_at(g[i] - 2), top_at(g[i] - 1));
        }));
    {
        std::vector<int> sp(n);
        std::iota(sp.begin(), sp.end(), 0);
        std::stable_sort(sp.begin(), sp.end(), [&](int a, int b) { return p[a] != p[b] ? p[a] < p[b] : on[a] < on[b]; });
        std::vector<double> prev_same(n, kNaN), next_same(n, kNaN);
        for (std::size_t k = 0; k + 1 < n; ++k) {
            if (p[sp[k + 1]] == p[sp[k]]) {
                const double dt = on[sp[k + 1]] - on[sp[k]];
                prev_same[sp[k + 1]] = dt;
                next_same[sp[k]] = dt;
            }
        }
        add("same_pitch_prev_dt", prev_same);
        add("same_pitch_next_dt", next_same);
    }

    // --- chord / polyphony ----------------------------------------------------
    std::vector<double> sorted_on = on, sorted_off = off;
    std::sort(sorted_on.begin(), sorted_on.end());
    std::sort(sorted_off.begin(), sorted_off.end());
    std::vector<double> held(n), chord_size(n), polyphony(n), span(n);
    for (std::size_t i = 0; i < n; ++i) {
        held[i] = double(std::max<long>(lower_idx(sorted_on, gt[i]) - upper_idx(sorted_off, gt[i]), 0));
        chord_size[i] = double(g_size[g[i]]);
        polyphony[i] = held[i] + chord_size[i];
        span[i] = g_top[g[i]] - g_low[g[i]];
    }
    const double ctol = cfg.chord_tolerance_s;
    add("chord_size", chord_size);
    add("n_onsets_within_tol", column([&](std::size_t i) {
            return double(upper_idx(sorted_on, on[i] + ctol) - lower_idx(sorted_on, on[i] - ctol));
        }));
    add("held_notes_at_onset", held);
    add("polyphony_at_onset", polyphony);
    add("chord_highest", column([&](std::size_t i) { return g_top[g[i]]; }));
    add("chord_lowest", column([&](std::size_t i) { return g_low[g[i]]; }));
    add("chord_span", span);
    add("chord_pos_norm", column([&](std::size_t i) { return span[i] > 0 ? (double(p[i]) - g_low[g[i]]) / span[i] : kNaN; }));
    add("dist_from_chord_top", column([&](std::size_t i) { return g_top[g[i]] - double(p[i]); }));
    add("dist_from_chord_bottom", column([&](std::size_t i) { return double(p[i]) - g_low[g[i]]; }));
    add("chord_rank_from_top", column([&](std::size_t i) { return double(g_end[g[i]] - 1 - long(i)); }));
    add("chord_rank_from_bottom", column([&](std::size_t i) { return double(long(i) - g_start[g[i]]); }));
    add("is_chord_top", column([&](std::size_t i) { return double(g_end[g[i]] - 1 - long(i) == 0); }));
    add("is_chord_bottom", column([&](std::size_t i) { return double(long(i) - g_start[g[i]] == 0); }));

    // --- local context windows ------------------------------------------------
    std::vector<double> cs_p(n + 1, 0.0), cs_dur(n + 1, 0.0), cs_poly(n + 1, 0.0), cs_first(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {  // sequential, like numpy.cumsum
        cs_p[i + 1] = cs_p[i] + double(p[i]);
        cs_dur[i + 1] = cs_dur[i] + dur[i];
        cs_poly[i + 1] = cs_poly[i] + polyphony[i];
        cs_first[i + 1] = cs_first[i] + double(long(i) == g_start[g[i]]);
    }
    auto window_stats = [&](const std::vector<double>& times, double w, const std::string& tag) {
        std::vector<double> nd(n), od(n), mp(n), rel(n), rng(n), md(n), mpoly(n);
        const double width = 2 * w;
        for (std::size_t i = 0; i < n; ++i) {
            const long lo = lower_idx(times, times[i] - w);
            const long hi = upper_idx(times, times[i] + w);
            const double cnt = double(hi - lo);
            long mx = p[lo], mn = p[lo];
            for (long k = lo; k < hi; ++k) {
                mx = std::max(mx, p[k]);
                mn = std::min(mn, p[k]);
            }
            nd[i] = cnt / width;
            od[i] = (cs_first[hi] - cs_first[lo]) / width;
            mp[i] = (cs_p[hi] - cs_p[lo]) / cnt;
            rel[i] = double(p[i]) - mp[i];
            rng[i] = double(mx - mn);
            md[i] = (cs_dur[hi] - cs_dur[lo]) / cnt;
            mpoly[i] = (cs_poly[hi] - cs_poly[lo]) / cnt;
        }
        add(tag + "_note_density", nd);
        add(tag + "_onset_density", od);
        add(tag + "_mean_pitch", mp);
        add(tag + "_pitch_rel_mean", rel);
        add(tag + "_pitch_range", rng);
        add(tag + "_mean_dur", md);
        add(tag + "_mean_polyphony", mpoly);
    };
    for (double w : cfg.time_windows_s) window_stats(gt, w, "win" + fmt_g(w * 1000) + "ms");
    for (double w : cfg.beat_windows) window_stats(gt_b, w, "win" + fmt_g(w) + "b");
    add("notes_in_prev_beat", column([&](std::size_t i) {
            return double(lower_idx(gt_b, gt_b[i]) - lower_idx(gt_b, gt_b[i] - 1.0));
        }));
    add("notes_in_next_beat", column([&](std::size_t i) {
            return double(upper_idx(gt_b, gt_b[i] + 1.0) - upper_idx(gt_b, gt_b[i]));
        }));

    for (int nn : cfg.nearest_n) {
        std::vector<double> rel_med(n), rel_mean(n), rng(n), pct(n), tpn(n);
        std::vector<double> buf;
        for (std::size_t i = 0; i < n; ++i) {
            const long lo = std::max<long>(0, long(i) - nn);
            const long hi = std::min<long>(long(n) - 1, long(i) + nn);
            buf.clear();
            double sum = 0, below = 0;
            for (long k = lo; k <= hi; ++k) {
                buf.push_back(double(p[k]));
                sum += double(p[k]);
                below += p[k] < p[i];
            }
            const double valid = double(hi - lo + 1);
            rel_med[i] = double(p[i]) - median_of(buf);
            rel_mean[i] = double(p[i]) - sum / valid;
            rng[i] = *std::max_element(buf.begin(), buf.end()) - *std::min_element(buf.begin(), buf.end());
            pct[i] = below / valid;
            double tmax = gt[lo], tmin = gt[lo];
            for (long k = lo; k <= hi; ++k) {
                tmax = std::max(tmax, gt[k]);
                tmin = std::min(tmin, gt[k]);
            }
            tpn[i] = valid > 1 ? (tmax - tmin) / std::max(valid - 1, 1.0) : kNaN;
        }
        const std::string tag = "n" + std::to_string(nn);
        add(tag + "_pitch_rel_median", rel_med);
        add(tag + "_pitch_rel_mean", rel_mean);
        add(tag + "_pitch_range", rng);
        add(tag + "_pitch_pct", pct);
        add(tag + "_time_per_note", tpn);
    }

    // --- structure ------------------------------------------------------------
    add("measure_number", column([&](std::size_t i) { return double(grid.measure_index[bi[i]]); }));
    std::vector<double> cummax_off(n);
    for (std::size_t i = 0; i < n; ++i) cummax_off[i] = i ? std::max(cummax_off[i - 1], off[i]) : off[i];
    std::vector<double> sound_end(G);
    for (long k = 0; k < G; ++k) sound_end[k] = cummax_off[g_end[k] - 1];
    std::vector<char> rest_start(G, 1), phrase_start(G, 1);
    for (long k = 0; k + 1 < G; ++k) {
        const double silence = g_time[k + 1] - sound_end[k];
        const double ioi_s = g_time[k + 1] - g_time[k];
        const double ioi_b = g_beat[k + 1] - g_beat[k];
        rest_start[k + 1] = silence >= cfg.rest_min_s;
        phrase_start[k + 1] = silence >= cfg.rest_min_s || (ioi_b >= cfg.phrase_gap_beats && ioi_s >= cfg.phrase_gap_min_s);
    }
    auto segments = [&](const std::vector<char>& starts, std::vector<long>& first, std::vector<long>& last) {
        first.assign(G, 0);
        last.assign(G, G - 1);
        long cur = 0;
        for (long k = 0; k < G; ++k) {
            if (starts[k]) cur = k;
            first[k] = cur;
        }
        long end = G - 1;
        for (long k = G - 1; k >= 0; --k) {
            last[k] = end;
            if (starts[k]) end = k - 1;
        }
    };
    std::vector<long> r_first, r_last, ph_first, ph_last;
    segments(rest_start, r_first, r_last);
    segments(phrase_start, ph_first, ph_last);
    std::vector<double> since_b(n), to_b(n);
    for (std::size_t i = 0; i < n; ++i) {
        since_b[i] = g_beat[g[i]] - g_beat[ph_first[g[i]]];
        to_b[i] = g_beat[ph_last[g[i]]] - g_beat[g[i]];
    }
    add("time_since_rest_s", column([&](std::size_t i) { return g_time[g[i]] - g_time[r_first[g[i]]]; }));
    add("time_since_phrase_start_beats", since_b);
    add("notes_since_phrase_start", column([&](std::size_t i) { return double(long(i) - g_start[ph_first[g[i]]]); }));
    add("time_to_rest_s", column([&](std::size_t i) { return sound_end[r_last[g[i]]] - g_time[g[i]]; }));
    add("phrase_pos_norm", column([&](std::size_t i) {
            const double len = since_b[i] + to_b[i];
            return len > 0 ? since_b[i] / len : kNaN;
        }));
    add("time_to_phrase_end_beats", to_b);
    add("phrase_length_beats", column([&](std::size_t i) { return since_b[i] + to_b[i]; }));
    add("notes_to_phrase_end", column([&](std::size_t i) { return double(g_end[ph_last[g[i]]] - 1 - long(i)); }));
    {
        const double total = std::max(g_time[G - 1] - g_time[0], 1e-6);
        add("rel_position", column([&](std::size_t i) { return (gt[i] - g_time[0]) / total; }));
    }

    // --- piece-level -------------------------------------------------------------
    if (cfg.include_piece_features) {
        const double on_min = *std::min_element(on.begin(), on.end());
        const double off_max = *std::max_element(off.begin(), off.end());
        const double length_s = std::max(off_max - on_min, 1e-3);
        std::vector<double> tempo;
        for (std::size_t b = 0; b < grid.times.size(); ++b)
            if (grid.times[b] >= on_min && grid.times[b] <= off_max) tempo.push_back(grid.tempo_bpm[b]);
        if (tempo.empty()) tempo = grid.tempo_bpm;
        // Integer sums are exact, so these means match numpy bit for bit; the std uses an exact
        // integer formula (numpy's SIMD summation order is not reproducible, but both round to
        // the same float32 value).
        long long s1 = 0, s2 = 0, gsum = 0;
        double poly_sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            s1 += p[i];
            s2 += static_cast<long long>(p[i]) * p[i];
            poly_sum += polyphony[i];
        }
        for (long k = 0; k < G; ++k) gsum += g_size[k];
        const double nd = double(n);
        const double mean_p = double(s1) / nd;
        const double var_p = double(static_cast<long long>(n) * s2 - s1 * s1) / (nd * nd);
        const double values[] = {length_s, nd / length_s, mean_p, std::sqrt(var_p), poly_sum / nd,
                                 double(gsum) / double(G), median_of(tempo)};
        const char* names[] = {"piece_duration_s", "piece_note_rate", "piece_mean_pitch", "piece_pitch_std",
                               "piece_mean_polyphony", "piece_mean_chord_size", "piece_median_tempo_bpm"};
        for (int k = 0; k < 7; ++k) add(names[k], std::vector<double>(n, values[k]));
        add("pitch_rel_piece_mean", column([&](std::size_t i) { return double(p[i]) - mean_p; }));
    }
    return F;
}

// ---------------------------------------------------------------------------
// Targets/post-processing used at inference
// ---------------------------------------------------------------------------

std::vector<double> residual_baseline(const std::vector<double>& v, const std::vector<int>& group_id, int window,
                                      const std::string& stat) {
    const long n = static_cast<long>(v.size());
    std::vector<double> out(n), buf;
    const double fallback = median_of(v);
    for (long i = 0; i < n; ++i) {
        buf.clear();
        for (long k = std::max(0L, i - window); k <= std::min(n - 1, i + window); ++k)
            if (group_id[k] != group_id[i]) buf.push_back(v[k]);
        if (buf.empty()) {
            out[i] = fallback;
        } else if (stat == "median") {
            out[i] = median_of(buf);
        } else {
            out[i] = std::accumulate(buf.begin(), buf.end(), 0.0) / double(buf.size());
        }
    }
    return out;
}

std::vector<double> smooth_velocities(const std::vector<double>& v, const std::vector<int>& group_id, double alpha,
                                      double strength, double accent_threshold) {
    if (v.empty() || strength <= 0) return v;
    const int G = *std::max_element(group_id.begin(), group_id.end()) + 1;
    std::vector<double> sum(G, 0.0), cnt(G, 0.0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        sum[group_id[i]] += v[i];
        cnt[group_id[i]] += 1;
    }
    std::vector<double> gmean(G), fwd(G), bwd(G);
    for (int k = 0; k < G; ++k) gmean[k] = sum[k] / std::max(cnt[k], 1.0);
    double acc = gmean[0];
    for (int k = 0; k < G; ++k) fwd[k] = acc = alpha * gmean[k] + (1.0 - alpha) * acc;
    acc = gmean[G - 1];
    for (int k = G - 1; k >= 0; --k) bwd[k] = acc = alpha * gmean[k] + (1.0 - alpha) * acc;
    std::vector<double> shift(G);
    for (int k = 0; k < G; ++k) {
        const double trend = 0.5 * (fwd[k] + bwd[k]);
        double smoothed = gmean[k] + strength * (trend - gmean[k]);
        if (gmean[k] - trend > accent_threshold) smoothed = gmean[k];  // keep accents
        shift[k] = smoothed - gmean[k];
    }
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i] + shift[group_id[i]];  // voicing preserved
    return out;
}

std::vector<double> shape_dynamics(const std::vector<double>& v, double scale, double offset) {
    const double centre = v.empty() ? 0.0 : median_of(v);
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = centre + scale * (v[i] - centre) + offset;
    return out;
}

std::vector<int> clip_velocities(const std::vector<double>& v) {
    std::vector<int> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double x = std::isfinite(v[i]) ? v[i] : 64.0;
        out[i] = static_cast<int>(std::clamp(rint_even(x), 1.0, 127.0));
    }
    return out;
}

}  // namespace humanbro::detail
