// Algorithm-compatible port of NumPy 2.5.4 SeedSequence and PCG64.
// Attribution/license: native/tests/SAMPLING_NOTICE.md.
#include "ni26/sampling.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ni26 {
namespace {
constexpr __uint128_t multiplier =
    (__uint128_t(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
uint32_t hash_mix(uint32_t value, uint32_t &constant) {
  value ^= constant;
  constant *= 0x931e8875U;
  value *= constant;
  return value ^ (value >> 16);
}
uint32_t mix(uint32_t a, uint32_t b) {
  auto result = 0xca01f9ddU * a - 0x4973f715U * b;
  return result ^ (result >> 16);
}
std::array<uint64_t, 4> seed_words(uint64_t seed) {
  std::vector<uint32_t> entropy{uint32_t(seed)};
  if (seed >> 32)
    entropy.push_back(seed >> 32);
  std::array<uint32_t, 4> pool{};
  uint32_t constant = 0x43b0d7e5U;
  for (size_t i = 0; i < pool.size(); i++)
    pool[i] = hash_mix(i < entropy.size() ? entropy[i] : 0, constant);
  for (size_t source = 0; source < pool.size(); source++)
    for (size_t target = 0; target < pool.size(); target++)
      if (source != target)
        pool[target] = mix(pool[target], hash_mix(pool[source], constant));
  std::array<uint32_t, 8> generated{};
  constant = 0x8b51f9ddU;
  for (size_t i = 0; i < generated.size(); i++) {
    auto value = pool[i % 4] ^ constant;
    constant *= 0x58f38dedU;
    value *= constant;
    generated[i] = value ^ (value >> 16);
  }
  std::array<uint64_t, 4> words{};
  for (size_t i = 0; i < 4; i++)
    words[i] =
        uint64_t(generated[2 * i]) | (uint64_t(generated[2 * i + 1]) << 32);
  return words;
}
} // namespace
Pcg64::Pcg64(uint64_t seed) {
  auto words = seed_words(seed);
  auto initial = (__uint128_t(words[0]) << 64) | words[1];
  auto sequence = (__uint128_t(words[2]) << 64) | words[3];
  increment_ = (sequence << 1) | 1;
  state_ = increment_;
  state_ += initial;
  state_ = state_ * multiplier + increment_;
}
uint64_t Pcg64::next64() {
  state_ = state_ * multiplier + increment_;
  auto value = uint64_t(state_ >> 64) ^ uint64_t(state_);
  unsigned rotation = state_ >> 122;
  return (value >> rotation) | (value << ((-rotation) & 63));
}
uint32_t Pcg64::next32() {
  if (has32_) {
    has32_ = false;
    return cached32_;
  }
  auto value = next64();
  has32_ = true;
  cached32_ = value >> 32;
  return uint32_t(value);
}
std::array<uint64_t, 6> Pcg64::state() const {
  return {uint64_t(state_ >> 64), uint64_t(state_), uint64_t(increment_ >> 64),
          uint64_t(increment_),   uint64_t(has32_), cached32_};
}
uint64_t Pcg64::interval(uint64_t max) {
  if (!max)
    return 0;
  auto mask = max;
  for (unsigned shift : {1, 2, 4, 8, 16, 32})
    mask |= mask >> shift;
  uint64_t value;
  do {
    value = (max <= UINT32_MAX ? next32() : next64()) & mask;
  } while (value > max);
  return value;
}
uint64_t Pcg64::bounded(uint64_t max) {
  if (!max)
    return 0;
  if (max <= UINT32_MAX) {
    if (max == UINT32_MAX)
      return next32();
    uint32_t bound = max + 1;
    auto product = uint64_t(next32()) * bound;
    uint32_t low = product;
    if (low < bound) {
      uint32_t threshold = (UINT32_MAX - uint32_t(max)) % bound;
      while (low < threshold) {
        product = uint64_t(next32()) * bound;
        low = product;
      }
    }
    return product >> 32;
  }
  if (max == UINT64_MAX)
    return next64();
  auto bound = max + 1;
  auto product = __uint128_t(next64()) * bound;
  uint64_t low = product;
  if (low < bound) {
    auto threshold = (UINT64_MAX - max) % bound;
    while (low < threshold) {
      product = __uint128_t(next64()) * bound;
      low = product;
    }
  }
  return product >> 64;
}
} // namespace ni26
