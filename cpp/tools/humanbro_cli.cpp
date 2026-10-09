// Command-line humanizer: the C++ counterpart of humanize_midi.py and humanize_timing.py (beat
// grid from the file's tempo map and time signatures, i.e. --beat_source midi).
//
//   humanbro --model velocity.hbm --input in.mid --output out.mid [--smoothing] [--dynamics-scale 1.2]
//   humanbro --timing-model timing.hbm --input in.mid --output out.mid [--seed 7] [--temperature 0.8]
//   humanbro --model velocity.hbm --timing-model timing.hbm --input in.mid --output out.mid
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "humanbro/humanbro.hpp"

namespace {

void usage() {
    std::cerr <<
        "usage: humanbro [--model VELOCITY.hbm] [--timing-model TIMING.hbm] --input IN.mid --output OUT.mid [options]\n"
        "velocity (--model):\n"
        "  --smoothing                 smooth chord-to-chord jitter (keeps voicing and accents)\n"
        "  --smoothing-alpha A         EMA coefficient (default 0.35)\n"
        "  --smoothing-strength S      0..1 (default 0.5)\n"
        "  --accent-threshold T        groups this far above the trend are kept (default 8)\n"
        "  --dynamics-scale X          >1 widens, <1 narrows the dynamic range (default 1)\n"
        "  --offset Y                  shift all velocities (default 0)\n"
        "  --mix M                     1 = model only, 0 = input velocities (default 1)\n"
        "  --constant-baseline         residual models: use --base-velocity instead of input dynamics\n"
        "  --base-velocity V           (default 64)\n"
        "  --dump-features FILE.csv    write the feature table (canonical order)\n"
        "  --dump-raw FILE.csv         write raw model outputs (canonical order)\n"
        "timing (--timing-model):\n"
        "  --timing-mode sample|median correlated random take (default) or each note's median offset\n"
        "  --temperature T             spread of a take: 0 = middle of each distribution, 1 = data (default)\n"
        "  --timing-scale S            multiply every offset (default 1)\n"
        "  --seed N                    reproducible take (same seed = same take as humanize_timing.py)\n"
        "  --max-offset-ms M           clamp (default 150)\n"
        "  --chord-coupling R          override the fitted within-chord correlation (0..1)\n"
        "  --correlation-beats B       override the fitted correlation length (beats)\n"
        "  --snap                      snap chords to the 1/16 or triplet grid first\n"
        "  --dump-timing FILE.csv      write quantiles and offsets (canonical order)\n";
}

std::string fmt17(double v) {
    if (std::isnan(v)) return "nan";
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

std::string fmt9(double v) {
    if (std::isnan(v)) return "nan";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, timing_path, input, output, dump_features, dump_raw, dump_timing;
    humanbro::Options opt;
    humanbro::TimingOptions topt;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw humanbro::Error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--model") model_path = value();
            else if (a == "--timing-model") timing_path = value();
            else if (a == "--input") input = value();
            else if (a == "--output") output = value();
            else if (a == "--smoothing") opt.smoothing = true;
            else if (a == "--smoothing-alpha") opt.smoothing_alpha = std::stod(value());
            else if (a == "--smoothing-strength") opt.smoothing_strength = std::stod(value());
            else if (a == "--accent-threshold") opt.accent_threshold = std::stod(value());
            else if (a == "--dynamics-scale") opt.dynamics_scale = std::stod(value());
            else if (a == "--offset") opt.offset = std::stod(value());
            else if (a == "--mix") opt.mix = std::stod(value());
            else if (a == "--constant-baseline") opt.constant_baseline = true;
            else if (a == "--base-velocity") opt.base_velocity = std::stod(value());
            else if (a == "--dump-features") dump_features = value();
            else if (a == "--dump-raw") dump_raw = value();
            else if (a == "--timing-mode") {
                const std::string m = value();
                if (m != "sample" && m != "median") throw humanbro::Error("--timing-mode must be sample or median");
                topt.sample = m == "sample";
            }
            else if (a == "--temperature") topt.temperature = std::stod(value());
            else if (a == "--timing-scale") topt.scale = std::stod(value());
            else if (a == "--seed") topt.seed = std::stoull(value());
            else if (a == "--max-offset-ms") topt.max_offset_ms = std::stod(value());
            else if (a == "--chord-coupling") topt.chord_coupling = std::stod(value());
            else if (a == "--correlation-beats") topt.correlation_beats = std::stod(value());
            else if (a == "--snap") topt.snap = true;
            else if (a == "--dump-timing") dump_timing = value();
            else if (a == "-h" || a == "--help") { usage(); return 0; }
            else throw humanbro::Error("unknown argument " + a);
        }
        if ((model_path.empty() && timing_path.empty()) || input.empty() || output.empty()) {
            usage();
            return 2;
        }

        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<humanbro::Humanizer> h;
        std::unique_ptr<humanbro::TimingHumanizer> th;
        if (!model_path.empty()) h = std::make_unique<humanbro::Humanizer>(model_path);
        if (!timing_path.empty()) th = std::make_unique<humanbro::TimingHumanizer>(timing_path);
        const auto t1 = std::chrono::steady_clock::now();
        const humanbro::MidiFile midi = humanbro::MidiFile::load(input);
        const humanbro::Score& score = midi.score();

        // Both models read the original clip; the file is written once at the end.
        std::vector<int> velocities;
        humanbro::BeatInfo info;
        if (h) {
            if (!dump_features.empty() || !dump_raw.empty()) {
                const humanbro::FeatureTable table = h->features(score);
                if (!dump_features.empty()) {
                    std::ofstream f(dump_features);
                    f << "note_index,group_id";
                    for (const auto& name : table.names) f << ',' << name;
                    f << '\n';
                    for (std::size_t r = 0; r < table.rows(); ++r) {
                        f << table.note_index[r] << ',' << table.group_id[r];
                        for (std::size_t c = 0; c < table.cols(); ++c) f << ',' << fmt9(table.at(r, c));
                        f << '\n';
                    }
                }
                if (!dump_raw.empty()) {
                    const std::vector<double> raw = h->predict_raw(table);
                    std::ofstream f(dump_raw);
                    f << "note_index,raw\n";
                    for (std::size_t r = 0; r < raw.size(); ++r) f << table.note_index[r] << ',' << fmt9(raw[r]) << '\n';
                }
            }
            h->features(score, &info);
            velocities = h->humanize(score, opt);
        }
        humanbro::TimingResult timing;
        if (th) {
            timing = th->humanize(score, topt);
            info = timing.grid;
            if (!dump_timing.empty()) {
                const humanbro::FeatureTable table = th->features(score, topt.snap);
                const std::vector<double> q = th->predict_quantiles(table);
                const std::size_t k = th->quantile_levels().size();
                std::ofstream f(dump_timing);
                f << "note_index,group_id";
                for (std::size_t j = 0; j < k; ++j) f << ",q" << j;
                f << ",offset_ms\n";
                for (std::size_t r = 0; r < table.rows(); ++r) {
                    f << table.note_index[r] << ',' << table.group_id[r];
                    for (std::size_t j = 0; j < k; ++j) f << ',' << fmt17(q[r * k + j]);
                    f << ',' << fmt17(timing.offset_ms[static_cast<std::size_t>(table.note_index[r])]) << '\n';
                }
            }
        }
        const auto t2 = std::chrono::steady_clock::now();
        if (th) midi.save_with_timing(output, timing.onset_tick, timing.offset_tick, velocities);
        else midi.save_with_velocities(output, velocities);

        std::printf("%zu notes; grid %d/%d, median tempo %.0f bpm; humanized in %.0f ms (models loaded in %.0f ms)\n",
                    score.notes.size(), info.beats_per_bar, info.beat_unit, info.median_tempo_bpm, ms(t1, t2),
                    ms(t0, t1));
        if (h) {
            double sum = 0;
            int lo = 127, hi = 1;
            for (int v : velocities) {
                sum += v;
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            std::printf("velocity model: %zu trees, target=%s%s; output mean %.1f, range %d-%d\n", h->num_trees(),
                        h->target_mode().c_str(), h->quantizes_input() ? ", quantized input" : "",
                        velocities.empty() ? 0.0 : sum / velocities.size(), lo, hi);
        }
        if (th) {
            double sum = 0, sq = 0;
            for (double v : timing.offset_ms) {
                sum += v;
                sq += v * v;
            }
            const double nn = timing.offset_ms.empty() ? 1.0 : double(timing.offset_ms.size());
            const double mean = sum / nn;
            if (topt.sample)
                std::printf("timing model: %zu trees; sampled take with --seed %llu, temperature %g; offsets sd %.1f ms\n",
                            th->num_trees(), static_cast<unsigned long long>(timing.seed), topt.temperature,
                            std::sqrt(std::max(sq / nn - mean * mean, 0.0)));
            else
                std::printf("timing model: %zu trees; median offsets, sd %.1f ms\n", th->num_trees(),
                            std::sqrt(std::max(sq / nn - mean * mean, 0.0)));
        }
        std::printf("-> %s\n", output.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
