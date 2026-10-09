// Tree-ensemble model + the feature-pipeline settings it was trained with (from a .hbm file).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace humanbro::detail {

struct PipelineConfig {
    std::string task = "velocity";  // "velocity" or "timing"
    std::string target_mode = "absolute";
    std::string experiment = "primary";
    std::string context = "bidirectional";
    double chord_tolerance_s = 0.035;
    std::vector<double> time_windows_s;
    std::vector<double> beat_windows;
    std::vector<int> nearest_n;
    double strong_beat_tolerance = 0.12;
    double rest_min_s = 0.2;
    double phrase_gap_beats = 1.5;
    double phrase_gap_min_s = 0.35;
    bool include_piece_features = true;
    bool quantize = false;
    std::vector<int> quantize_subdivisions;
    int residual_window_notes = 24;
    std::string residual_stat = "median";
    // Timing models: optional grid snap (timing_features.snap_groups).
    std::vector<int> timing_subdivisions;
    double timing_switch_cost = 0.01;
};

// Timing models: quantile levels and the sampler fitted on validation data (timing_sampling.py).
struct SamplerParams {
    std::vector<double> levels;  // quantile levels, increasing
    std::vector<double> knots;   // Phi^-1(levels), exported from Python
    double rho = 0.5;            // within-chord correlation of the normal scores
    double ell = 1.0;            // correlation length of the shared part, beats
    std::vector<double> calib_u, calib_z;  // marginal recalibration table
};

struct Tree {
    std::vector<int32_t> left;   // -1 marks a leaf
    std::vector<int32_t> right;
    std::vector<int32_t> feature;
    std::vector<float> value;    // threshold (internal) or leaf value
    std::vector<uint8_t> default_left;
};

struct TreeModel {
    PipelineConfig config;
    SamplerParams sampler;
    float base_score = 0.0f;
    int num_outputs = 1;
    std::vector<std::string> features;
    std::vector<Tree> trees;
    std::vector<int32_t> tree_output;  // output each tree adds to

    static TreeModel load(const std::string& path);

    static int32_t leaf(const Tree& t, const float* row) {
        int32_t node = 0;
        while (t.left[node] != -1) {
            const float v = row[t.feature[node]];
            const bool go_left = (v != v) ? t.default_left[node] != 0 : v < t.value[node];
            node = go_left ? t.left[node] : t.right[node];
        }
        return node;
    }

    // XGBoost semantics: go left if x < threshold; NaN follows default_left;
    // float32 accumulation starting from base_score, in tree order.
    float predict(const float* row) const {
        float acc = base_score;
        for (const Tree& t : trees) acc += t.value[leaf(t, row)];
        return acc;
    }

    // Multi-output models (one tree per output per round): out[k] for k < num_outputs.
    void predict(const float* row, float* out) const {
        for (int k = 0; k < num_outputs; ++k) out[k] = base_score;
        for (std::size_t i = 0; i < trees.size(); ++i) out[tree_output[i]] += trees[i].value[leaf(trees[i], row)];
    }
};

}  // namespace humanbro::detail
