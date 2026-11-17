#include "adam.h"
#include "ni26/hash.h"
#include "ni26/initialization.h"
#include "ni26/math.h"
#include "training_internal.h"
#include <chrono>
#include <numeric>
#include <set>
#include <torch/torch.h>

namespace ni26::neural {
RawTrainingSource prepare_raw_training_source(const Params &input) {
  require(input.integer("source_only") == 1 && input.integer("epochs") == 40,
          "raw FBC requires registered source-only40-epoch recipe");
  auto people = input.texts("source_people"),
       excluded = input.texts("excluded_people");
  require(std::set<std::string>(people.begin(), people.end()) ==
              std::set<std::string>({"P04", "P05", "P06", "P07", "P08", "P10"}),
          "native source6 identities differ");
  require(std::set<std::string>(excluded.begin(), excluded.end()) ==
              std::set<std::string>({"P02", "P03", "P09"}),
          "native exposed query exclusion differs");
  require(!input.has("source_X") && !input.has("batch_indices") &&
              !input.has("batch_sizes"),
          "production raw FBC rejects prepared input or imported batch tables");
  for (const auto &[key, value] : input.tensors)
    require(key.rfind("initial_", 0) != 0,
            "production raw FBC rejects imported initial weights");
  SourceData data{input.get("x"), input.get("stream"), input.get("person"),
                  input.get("fit_mask"), input.get("held_person")};
  auto source = prepare_raw_neural_source(data);
  require(source.x.sizes() == torch::IntArrayRef({1898, 41, 480}),
          "native source official tensor differs");
  auto labels = input.get("y");
  require(labels.device().is_cpu() && labels.scalar_type() == torch::kInt64 &&
              labels.dim() == 1 && labels.numel() == data.raw.size(0),
          "native source label schema differs");
  // Excluded labels are never inspected; only rows selected by the source guard
  // enter loss.
  labels = labels.index_select(0, source.rows);
  require(labels.min().item<int64_t>() >= 0 && labels.max().item<int64_t>() < 3,
          "native selected source labels outside three classes");
  auto seed = input.integer("initialization_seed");
  require(seed >= 0, "native source seed must be nonnegative");
  auto initial =
      initialize_raw_fbc(input.subset("fir_"), static_cast<uint64_t>(seed));
  std::vector<uint8_t> coefficients;
  for (int64_t i = 0; i < 9; ++i) {
    auto f = initial.get("spectral_filtering.fir_list." + std::to_string(i));
    const auto *bytes = static_cast<const uint8_t *>(f.const_data_ptr());
    coefficients.insert(coefficients.end(), bytes, bytes + f.nbytes());
  }
  require(
      sha256(coefficients.data(), coefficients.size()) ==
          "607a44ee25576803092c03d77feb67411fe7a2266475fc730ae7ae28c81506c6",
      "native source fixed FIR fingerprint differs");
  std::vector<int64_t> rows(1898), sessions(1898);
  std::iota(rows.begin(), rows.end(), 0);
  std::copy(source.session.data_ptr<int64_t>(),
            source.session.data_ptr<int64_t>() + 1898, sessions.begin());
  require(session_batch_count(rows, sessions, 32, 64) == 31,
          "native source batch count differs");
  Pcg64 rng(static_cast<uint64_t>(seed));
  std::vector<Batches> epochs;
  for (int64_t epoch = 0; epoch < 40; ++epoch)
    epochs.push_back(session_batches(rows, sessions, 32, 64, rng));
  return {std::move(source), labels, std::move(initial), std::move(epochs)};
}
SourceUpdate raw_source_update(RawFBC &net, Adam &optimizer,
                               const Tensor &x, const Tensor &y, int64_t step) {
  torch::AutoGradMode enable_grad(true);
  auto [lr, beta] = one_cycle(step, 1240);
  optimizer.zero_grad();
  auto begun = std::chrono::steady_clock::now();
  auto logits = net.forward(x, true);
  auto loss = torch::nn::functional::cross_entropy(logits, y);
  finite(loss, "native source loss");
  double forward =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begun)
          .count();
  begun = std::chrono::steady_clock::now();
  loss.backward();
  for (const auto &p : net.parameters()) {
    require(p.grad().defined(), "native source gradient inactive");
    finite(p.grad(), "native source gradient");
  }
  double backward =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begun)
          .count();
  begun = std::chrono::steady_clock::now();
  optimizer.step(lr, beta);
  double update =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begun)
          .count();
  return {logits.detach(), loss.detach(), forward, backward, update, lr, beta};
}
} // namespace ni26::neural
