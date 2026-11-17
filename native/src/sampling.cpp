#include "ni26/sampling.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_map>
namespace ni26 {
SessionRows group_session_rows(const std::vector<int64_t> &source,
                               const std::vector<std::string> &streams) {
  std::vector<std::vector<int64_t>> groups;
  std::unordered_map<std::string, size_t> positions;
  for (auto row : source) {
    if (row < 0 || size_t(row) >= streams.size())
      throw std::invalid_argument("source row outside stream metadata");
    auto inserted = positions.emplace(streams[row], groups.size());
    if (inserted.second)
      groups.emplace_back();
    groups[inserted.first->second].push_back(row);
  }
  SessionRows result;
  for (size_t j = 0; j < groups.size(); j++)
    for (auto row : groups[j]) {
      result.order.push_back(row);
      result.session.push_back(j);
    }
  return result;
}
namespace {
std::map<int64_t, std::vector<int64_t>>
by_session(const std::vector<int64_t> &rows,
           const std::vector<int64_t> &sessions, int64_t group, int64_t batch) {
  if (group <= 0 || batch <= 0)
    throw std::invalid_argument("positive session group/batch required");
  std::map<int64_t, std::vector<int64_t>> selected;
  for (auto row : rows) {
    if (row < 0 || size_t(row) >= sessions.size())
      throw std::invalid_argument("training row outside session metadata");
    selected[sessions[row]].push_back(row);
  }
  return selected;
}
} // namespace
int64_t session_batch_count(const std::vector<int64_t> &rows,
                            const std::vector<int64_t> &sessions, int64_t group,
                            int64_t batch) {
  auto selected = by_session(rows, sessions, group, batch);
  int64_t chunks = 0;
  for (const auto &[session, values] : selected)
    chunks += (values.size() + group - 1) / group;
  auto groups = std::max<int64_t>(1, batch / group);
  return (chunks + groups - 1) / groups;
}
Batches session_batches(const std::vector<int64_t> &rows,
                        const std::vector<int64_t> &sessions, int64_t group,
                        int64_t batch, Pcg64 &rng) {
  auto selected = by_session(rows, sessions, group, batch);
  Batches chunks;
  for (auto &[session, original] : selected) {
    auto permutation = rng.permutation(original.size());
    std::vector<int64_t> values;
    values.reserve(original.size() + group);
    for (auto index : permutation)
      values.push_back(original[index]);
    auto size = values.size();
    auto pad = (group - size % group) % group;
    if (pad) {
      auto padding = rng.choice(size, pad, pad > size);
      for (auto index : padding)
        values.push_back(values[index]);
    }
    for (size_t start = 0; start < values.size(); start += group)
      chunks.emplace_back(values.begin() + start,
                          values.begin() + start + group);
  }
  auto order = rng.permutation(chunks.size());
  Batches batches;
  auto groups = std::max<int64_t>(1, batch / group);
  for (size_t start = 0; start < order.size(); start += groups) {
    std::vector<int64_t> values;
    for (size_t j = start; j < std::min<size_t>(order.size(), start + groups);
         j++)
      values.insert(values.end(), chunks[order[j]].begin(),
                    chunks[order[j]].end());
    batches.push_back(std::move(values));
  }
  return batches;
}
} // namespace ni26
