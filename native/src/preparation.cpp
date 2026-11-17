#include "ni26/preparation.h"
#include "networks_internal.h"
#include <cmath>
#include <torch/torch.h>

namespace ni26 {
RawNeuralSource prepare_raw_neural_source(const SourceData &source) {
  torch::NoGradGuard guard;
  auto selection = select_source_rows(source);
  auto prior = torch::zeros({41, 41}, torch::kFloat64);
  auto eye = torch::eye(41, torch::kFloat64);
  std::vector<int64_t> stream_order;
  std::map<int64_t, std::vector<int64_t>> grouped;
  for (int64_t i = 0; i < selection.rows.numel(); ++i) {
    auto row = selection.rows[i].item<int64_t>();
    auto raw = source.raw[row];
    finite(raw, "raw neural source window");
    auto centered = (raw - neural::temporal_row_mean_float32(raw)).to(torch::kFloat64);
    auto covariance = torch::matmul(centered, centered.t()) / 480.;
    prior.add_((1. - .001) * covariance + .001 * covariance.trace() / 41 * eye);
    auto stream = source.stream[row].item<int64_t>();
    if (!grouped.count(stream))
      stream_order.push_back(stream);
    grouped[stream].push_back(row);
  }
  // NumPy's non-contiguous source-axis mean sums in source order then divides.
  auto *values = prior.data_ptr<double>();
  for (int64_t i = 0; i < prior.numel(); ++i)
    values[i] /= static_cast<double>(selection.rows.numel());
  auto scale = std::sqrt(prior.trace().item<double>() / 41.);
  require(std::isfinite(scale) && scale > 0, "invalid raw neural source scale");
  std::vector<int64_t> order, sessions;
  for (size_t i = 0; i < stream_order.size(); ++i) {
    const auto &rows = grouped.at(stream_order[i]);
    order.insert(order.end(), rows.begin(), rows.end());
    sessions.insert(sessions.end(), rows.size(), static_cast<int64_t>(i));
  }
  auto rows = torch::tensor(order, torch::kInt64);
  auto x = (source.raw.index_select(0, rows).to(torch::kFloat64) / scale).to(torch::kFloat32);
  finite(x, "prepared raw neural source");
  return {x, rows, torch::tensor(sessions, torch::kInt64), prior, scale};
}
} // namespace ni26
