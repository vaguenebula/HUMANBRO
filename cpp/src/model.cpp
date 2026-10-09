#include "model.hpp"

#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#include "humanbro/humanbro.hpp"

namespace humanbro::detail {
namespace {

class Reader {
public:
    explicit Reader(std::vector<char> data) : data_(std::move(data)) {}

    template <typename T>
    T scalar() {
        T v;
        need(sizeof(T));
        std::memcpy(&v, data_.data() + pos_, sizeof(T));  // file is little-endian, as are x86/ARM
        pos_ += sizeof(T);
        return v;
    }

    template <typename T>
    std::vector<T> array(std::size_t n) {
        std::vector<T> v(n);
        need(n * sizeof(T));
        if (n) std::memcpy(v.data(), data_.data() + pos_, n * sizeof(T));
        pos_ += n * sizeof(T);
        return v;
    }

    std::string bytes(std::size_t n) {
        need(n);
        std::string s(data_.data() + pos_, n);
        pos_ += n;
        return s;
    }

private:
    void need(std::size_t n) const {
        if (pos_ + n > data_.size()) throw Error("model file is truncated");
    }
    std::vector<char> data_;
    std::size_t pos_ = 0;
};

std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    for (std::string tok; in >> tok;) out.push_back(tok);
    return out;
}

std::vector<double> to_doubles(const std::string& s) {
    std::vector<double> out;
    for (const auto& tok : split_ws(s)) out.push_back(std::stod(tok));
    return out;
}

std::vector<int> to_ints(const std::string& s) {
    std::vector<int> out;
    for (const auto& tok : split_ws(s)) out.push_back(std::stoi(tok));
    return out;
}

}  // namespace

TreeModel TreeModel::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open model file: " + path);
    Reader r(std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));

    if (r.bytes(4) != "HBRO") throw Error("not a HUMANBRO model file (bad magic): " + path);
    const auto version = r.scalar<uint32_t>();
    if (version != 1 && version != 2) throw Error("unsupported model file version " + std::to_string(version));

    std::map<std::string, std::string> kv;
    std::istringstream header(r.bytes(r.scalar<uint32_t>()));
    for (std::string line; std::getline(header, line);) {
        const auto eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    auto get = [&](const char* key) -> const std::string& {
        auto it = kv.find(key);
        if (it == kv.end()) throw Error(std::string("model header lacks '") + key + "'");
        return it->second;
    };
    auto get_or = [&](const char* key, const std::string& fallback) {
        auto it = kv.find(key);
        return it == kv.end() ? fallback : it->second;
    };

    TreeModel m;
    PipelineConfig& c = m.config;
    c.task = get_or("task", "velocity");  // version-1 files predate timing models
    if (c.task != "velocity" && c.task != "timing") throw Error("unknown model task '" + c.task + "'");
    c.target_mode = get("target_mode");
    c.experiment = get("experiment");
    c.context = get("context");
    c.chord_tolerance_s = std::stod(get("chord_tolerance_s"));
    c.time_windows_s = to_doubles(get("time_windows_s"));
    c.beat_windows = to_doubles(get("beat_windows"));
    c.nearest_n = to_ints(get("nearest_n"));
    c.strong_beat_tolerance = std::stod(get("strong_beat_tolerance"));
    c.rest_min_s = std::stod(get("rest_min_s"));
    c.phrase_gap_beats = std::stod(get("phrase_gap_beats"));
    c.phrase_gap_min_s = std::stod(get("phrase_gap_min_s"));
    c.include_piece_features = get("include_piece_features") == "1";
    c.quantize = get("quantize") == "1";
    c.quantize_subdivisions = to_ints(get("quantize_subdivisions"));
    m.base_score = static_cast<float>(std::stod(get("base_score")));
    m.num_outputs = std::stoi(get_or("num_outputs", "1"));
    m.features = split_ws(get("features"));

    if (c.task == "velocity") {
        c.residual_window_notes = std::stoi(get("residual_window_notes"));
        c.residual_stat = get("residual_stat");
        if (m.num_outputs != 1) throw Error("velocity models have one output");
    } else {
        c.timing_subdivisions = to_ints(get("timing_subdivisions"));
        c.timing_switch_cost = std::stod(get("timing_switch_cost"));
        SamplerParams& sp = m.sampler;
        sp.levels = to_doubles(get("quantile_levels"));
        sp.knots = to_doubles(get("quantile_knots"));
        sp.rho = std::stod(get("copula_rho"));
        sp.ell = std::stod(get("copula_ell"));
        sp.calib_u = to_doubles(get("calib_u"));
        sp.calib_z = to_doubles(get("calib_z"));
        if (sp.levels.size() < 2 || sp.knots.size() != sp.levels.size() ||
            static_cast<int>(sp.levels.size()) != m.num_outputs)
            throw Error("timing model: quantile levels, knots and outputs disagree");
        if (sp.calib_u.size() != sp.calib_z.size() || sp.calib_u.size() < 2)
            throw Error("timing model: bad recalibration table");
        if (c.timing_subdivisions.empty()) throw Error("timing model: timing_subdivisions is empty");
    }

    if (c.context != "bidirectional") throw Error("only bidirectional-context models are supported");
    if (c.experiment == "performance_conditioned")
        throw Error("performance-conditioned models need true velocities as input");
    if (c.quantize && c.quantize_subdivisions.empty()) throw Error("quantize_subdivisions is empty");
    if (m.num_outputs < 1) throw Error("model has no outputs");

    const auto n_trees = r.scalar<uint32_t>();
    m.trees.resize(n_trees);
    m.tree_output.assign(n_trees, 0);
    const auto n_features = static_cast<int32_t>(m.features.size());
    for (uint32_t ti = 0; ti < n_trees; ++ti) {
        Tree& t = m.trees[ti];
        if (version >= 2) {
            const auto out = r.scalar<uint32_t>();
            if (out >= static_cast<uint32_t>(m.num_outputs)) throw Error("tree output out of range in model file");
            m.tree_output[ti] = static_cast<int32_t>(out);
        }
        const auto n = r.scalar<uint32_t>();
        if (n == 0) throw Error("empty tree in model file");
        t.left = r.array<int32_t>(n);
        t.right = r.array<int32_t>(n);
        t.feature = r.array<int32_t>(n);
        t.value = r.array<float>(n);
        t.default_left = r.array<uint8_t>(n);
        for (uint32_t i = 0; i < n; ++i) {  // validate once so predict() can skip bounds checks
            if (t.left[i] == -1) continue;
            if (t.left[i] <= static_cast<int32_t>(i) || t.left[i] >= static_cast<int32_t>(n) ||
                t.right[i] <= static_cast<int32_t>(i) || t.right[i] >= static_cast<int32_t>(n) ||
                t.feature[i] < 0 || t.feature[i] >= n_features)
                throw Error("corrupt tree in model file");
        }
    }
    return m;
}

}  // namespace humanbro::detail
