#include "ni26/source.h"
#include "ni26/math.h"
#include <set>
#include <torch/torch.h>

namespace ni26 {
namespace {
void vector_field(const Tensor &value, int64_t n, torch::ScalarType type,
                  const std::string &name) {
  require(value.device().is_cpu() && value.is_contiguous() && value.scalar_type() == type &&
              value.dim() == 1 && value.size(0) == n,
          "invalid source vector: " + name);
}
struct Session {
  int64_t person = -1, total = 0, selected = 0;
};
} // namespace
SourceSelection select_source_rows(const SourceData &source) {
  auto raw = source.raw, stream = source.stream, person = source.person;
  auto mask = source.fit_mask, held = source.held_person;
  require(raw.device().is_cpu() && raw.scalar_type() == torch::kFloat32 && raw.dim() == 3 &&
              raw.size(1) == 41 && raw.size(2) == 480 && raw.size(0) > 0 &&
              raw.is_contiguous() && !raw.requires_grad(),
          "raw source must be CPU float32[N,41,480]");
  auto n = raw.size(0);
  vector_field(stream, n, torch::kInt64, "stream");
  vector_field(person, n, torch::kInt64, "person");
  vector_field(mask, n, torch::kBool, "fit_mask");
  require(held.dim() == 1, "held_person must be a vector");
  vector_field(held, held.numel(), torch::kInt64, "held_person");
  std::set<int64_t> excluded;
  for (int64_t i = 0; i < held.numel(); ++i)
    require(held[i].item<int64_t>() >= 0 && excluded.insert(held[i].item<int64_t>()).second,
            "held_person IDs must be nonnegative and unique");
  auto rows = torch::nonzero(mask).flatten().contiguous();
  require(rows.numel() >= 3, "insufficient selected source rows");
  std::map<int64_t, Session> sessions;
  for (int64_t i = 0; i < n; ++i) {
    auto id = stream[i].item<int64_t>(), who = person[i].item<int64_t>();
    require(id >= 0 && who >= 0, "source stream/person IDs must be nonnegative");
    auto &session = sessions[id];
    require(session.person < 0 || session.person == who, "stream ID crosses people");
    session.person = who;
    ++session.total;
    if (mask[i].item<bool>()) {
      require(!excluded.count(who), "selected source row belongs to an explicitly held person");
      ++session.selected;
    }
  }
  SourceSelection result{rows, {}};
  for (const auto &[id, session] : sessions) {
    require(session.selected == 0 || session.selected == session.total,
            "partial-session source fit mask is unsupported");
    if (session.selected)
      result.selected_per_stream.emplace(id, session.selected);
  }
  return result;
}
} // namespace ni26
