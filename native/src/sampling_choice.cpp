// NumPy 2.5.4 Generator permutation/choice algorithm compatibility.
// See native/tests/SAMPLING_NOTICE.md.
#include "ni26/sampling.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
namespace ni26 {
std::vector<int64_t> Pcg64::permutation(int64_t size) {
  if (size < 0)
    throw std::invalid_argument("negative permutation size");
  std::vector<int64_t> values(size);
  std::iota(values.begin(), values.end(), 0);
  for (int64_t i = size - 1; i > 0; i--)
    std::swap(values[i], values[interval(i)]);
  return values;
}
std::vector<int64_t> Pcg64::choice(int64_t population, int64_t count,
                                   bool replace) {
  if (population < 0 || count < 0 || (!replace && count > population))
    throw std::invalid_argument("invalid uniform choice population/count");
  std::vector<int64_t> result;
  if (!count)
    return result;
  if (!population)
    throw std::invalid_argument("empty choice population");
  result.reserve(count);
  if (replace) {
    for (int64_t i = 0; i < count; i++)
      result.push_back(bounded(population - 1));
    return result;
  }
  // NumPy's unweighted shuffle=True branch uses a tail shuffle only here.
  if (population > 10000 && count > population / 50) {
    result.resize(population);
    std::iota(result.begin(), result.end(), 0);
    for (int64_t i = population - 1;
         i >= std::max<int64_t>(population - count, 1); i--)
      std::swap(result[i], result[bounded(i)]);
    result.erase(result.begin(), result.end() - count);
    return result;
  }
  std::unordered_set<int64_t> selected;
  for (int64_t j = population - count; j < population; j++) {
    int64_t value = bounded(j);
    if (selected.count(value))
      value = j;
    selected.insert(value);
    result.push_back(value);
  }
  // Unlike permutation(), NumPy choice's final shuffle uses Lemire draws.
  for (int64_t i = count - 1; i > 0; i--)
    std::swap(result[i], result[bounded(i)]);
  return result;
}
} // namespace ni26
