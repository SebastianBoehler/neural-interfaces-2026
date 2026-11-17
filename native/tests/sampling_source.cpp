#include "ni26/sampling.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace ni26;
namespace {
void check(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
std::vector<int64_t> integers(std::istream &f, size_t n) {
  std::vector<int64_t> out(n);
  for (auto &v : out)
    f >> v;
  check(bool(f), "truncated metadata oracle");
  return out;
}
void grouping(const std::filesystem::path &dir) {
  std::ifstream f(dir / "grouping.txt");
  size_t n, m;
  f >> n >> m;
  std::vector<std::string> streams(n);
  for (auto &s : streams)
    f >> s;
  auto fit = integers(f, m), order = integers(f, m), sessions = integers(f, m);
  auto actual = group_session_rows(fit, streams);
  check(actual.order == order && actual.session == sessions,
        "original first-seen grouping mismatch");
  // A non-contiguous metadata-only case prevents a sorted/no-op grouping
  // shortcut.
  auto edge = group_session_rows({3, 0, 2, 1}, {"B", "A", "B", "A"});
  check(edge.order == std::vector<int64_t>({3, 1, 0, 2}) &&
            edge.session == std::vector<int64_t>({0, 0, 1, 1}),
        "first-seen edge grouping mismatch");
}
void plan(const std::filesystem::path &dir) {
  std::ifstream f(dir / "source.txt");
  int64_t n, group, batch, epochs;
  uint64_t seed;
  f >> n >> group >> batch >> seed >> epochs;
  check(n == 1898 && group == 32 && batch == 64 && seed == 0 && epochs == 40,
        "original source recipe metadata mismatch");
  auto sessions = integers(f, n);
  std::vector<int64_t> rows(n);
  std::iota(rows.begin(), rows.end(), 0);
  Pcg64 rng(seed), unrelated(77);
  auto before = rng.state();
  check(session_batch_count(rows, sessions, group, batch) == 31 &&
            before == rng.state(),
        "batch count consumes private training RNG");
  for (int64_t epoch = 0; epoch < epochs; epoch++) {
    Pcg64 twin = rng;
    auto batches = session_batches(rows, sessions, group, batch, rng);
    unrelated.next32();
    unrelated.next64();
    check(batches == session_batches(rows, sessions, group, batch, twin) &&
              rng.state() == twin.state(),
          "private RNG copy/interleave mismatch");
    size_t count;
    f >> count;
    check(count == batches.size() && count == 31, "batch cardinality mismatch");
    for (size_t b = 0; b < count; b++) {
      size_t length;
      f >> length;
      auto expected = integers(f, length);
      check(expected == batches[b], "row/chunk/padding mismatch at epoch" +
                                        std::to_string(epoch) + " batch" +
                                        std::to_string(b));
    }
    auto state = rng.state();
    for (size_t i = 0; i < state.size(); i++) {
      uint64_t value;
      f >> value;
      check(value == state[i], "PCG64 epoch state mismatch");
    }
  }
  check(bool(f), "truncated source plan");
  std::cout
      << "SAMPLER_SOURCE_PASS epochs=40 batches=1240 exact_rows_and_rng_states"
      << std::endl;
}
void primitives(const std::filesystem::path &dir) {
  std::ifstream f(dir / "rng.txt");
  size_t cases;
  f >> cases;
  for (size_t c = 0; c < cases; c++) {
    uint64_t seed;
    size_t count;
    f >> seed >> count;
    Pcg64 rng(seed);
    for (size_t j = 0; j < count; j++) {
      std::string op;
      f >> op;
      std::vector<uint64_t> actual;
      if (op == "STATE") {
        auto s = rng.state();
        actual.assign(s.begin(), s.end());
      } else if (op == "U32")
        actual.push_back(rng.next32());
      else if (op == "U64")
        actual.push_back(rng.next64());
      else if (op == "BOUND") {
        uint64_t maximum;
        f >> maximum;
        actual.push_back(rng.bounded(maximum));
      } else if (op == "PERM") {
        int64_t n;
        f >> n;
        auto v = rng.permutation(n);
        actual.assign(v.begin(), v.end());
      } else if (op == "CHOICE") {
        int64_t n, k;
        int replacement;
        f >> n >> k >> replacement;
        auto v = rng.choice(n, k, replacement);
        actual.assign(v.begin(), v.end());
      } else
        throw std::runtime_error("unexpected oracle operation");
      size_t n;
      f >> n;
      check(n == actual.size(), "RNG operation output shape mismatch");
      for (auto a : actual) {
        uint64_t expected;
        f >> expected;
        check(a == expected, "NumPy private RNG mismatch case" +
                                 std::to_string(c) + " op" + std::to_string(j) +
                                 " " + op);
      }
    }
  }
  check(bool(f), "truncated RNG oracle");
  std::cout << "NUMPY254_PRIVATE_RNG_PASS cases=" << cases << std::endl;
}
} // namespace
int main(int argc, char **argv) {
  try {
    check(argc == 2, "usage: sampling_source ORACLE_DIRECTORY");
    grouping(argv[1]);
    primitives(argv[1]);
    plan(argv[1]);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
