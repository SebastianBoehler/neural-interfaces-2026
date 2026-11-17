#pragma once
#include "ni26/artifact.h"

namespace ni26 {
enum class SpectralFunction { Log, Exp, InverseRoot, NegativeHalfExp };
Tensor spectral(const Tensor& x, SpectralFunction fn, double floor = 1e-12);
Tensor normalize(const Tensor& p);
Tensor posterior(const Tensor& z);
Tensor lda_posterior(const Tensor& z, const Tensor& means, const Tensor& inverse);
Tensor covariance(const Tensor& x, bool unbiased = true);
Tensor median(const Tensor& x, int64_t dim);
Tensor upper_features(const Tensor& x);
void require(bool condition, const std::string& message);
void finite(const Tensor& x, const std::string& name);
double ramp(double score, double threshold, double width);

class SosFilter {
 public:
  explicit SosFilter(const Tensor& sections);
  Tensor operator()(const Tensor& x) const;
 private:
  Tensor coefficients_, initial_;
  int64_t pad_;
};
std::vector<SosFilter> filters(const Params& z, const std::string& key);
}
