#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#if !defined(__SIZEOF_INT128__)
#error "NumPy PCG64 compatibility requires native unsigned 128-bit integers"
#endif

namespace ni26 {
// NumPy 2.5.4 default_rng scalar-uint64 seed / PCG64 compatibility.
// No global state, Python dependency, neural initialization or data access.
class Pcg64 {
public:
  explicit Pcg64(uint64_t seed);
  uint64_t next64();
  uint32_t next32();
  uint64_t interval(uint64_t inclusive_max);
  uint64_t bounded(uint64_t inclusive_max);
  std::vector<int64_t> permutation(int64_t size);
  std::vector<int64_t> choice(int64_t population, int64_t count, bool replace);
  // state high/low, increment high/low, cached32-present, cached32-value.
  std::array<uint64_t, 6> state() const;

private:
  __uint128_t state_ = 0, increment_ = 0;
  bool has32_ = false;
  uint32_t cached32_ = 0;
};
struct SessionRows {
  std::vector<int64_t> order;
  std::vector<int64_t> session;
};
// Exact first-seen stream grouping from AlignedNet._fit/SessionNet._fit.
SessionRows group_session_rows(const std::vector<int64_t> &source_rows,
                               const std::vector<std::string> &global_streams);
using Batches = std::vector<std::vector<int64_t>>;
Batches session_batches(const std::vector<int64_t> &rows,
                        const std::vector<int64_t> &row_session, int64_t group,
                        int64_t batch, Pcg64 &rng);
int64_t session_batch_count(const std::vector<int64_t> &rows,
                            const std::vector<int64_t> &row_session,
                            int64_t group, int64_t batch);
} // namespace ni26
