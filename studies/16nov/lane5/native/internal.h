#pragma once
#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

using T = torch::Tensor;
struct Model {
    int arm;
    torch::Device device;
    int64_t seed;
    std::vector<T> p, anchor;
    T mu, sd, raw_scale;
    std::deque<T> prefix;
    int64_t windows = 0, updates = 0, last_prefix = 0;
    double max_grad = 0;
    int64_t fit_steps = 0;
    double consistency_sum = 0, consistency_grad = 0, projection_grad_max = 0;
    uint64_t shuffle_hash = 14695981039346656037ULL;
    Model(int arm_, int device_, int64_t seed_);
    Model(const Model &other);
    void reset();
};

T from_float(const float *p, std::vector<int64_t> shape, torch::Device device);
T from_labels(const int64_t *p, int64_t n, torch::Device device);
void copy_float(const T &x, float *dst);
void finite(const T &x, const char *what);
T reference_features(const T &raw);
T embedding(Model &model, const T &raw);
T head(Model &model, const T &features);
T predict(Model &model, const T &raw);
T weighted_loss(Model &model, const T &x, const T &y, const T &w, const T &mask);
void fit(Model &model, const float *raw, const int64_t *labels, const int32_t *person,
         int64_t n, int32_t heldout, int epochs, int consistency);
T step(Model &model, const T &raw, bool update);
void save(Model &model, const char *path);
void load(Model &model, const char *path);
