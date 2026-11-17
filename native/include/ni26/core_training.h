#pragma once
#include "ni26/math.h"
#include "ni26/linear.h"

namespace ni26 {
struct ClassicalWindow {
  Tensor covariance, log_covariance, slow;
};
// Stateless per-window math shared by native source training and inference.
// Only the two actual CAR/full+two-half real and Hilbert presets are supported.
class ClassicalPreprocessor {
 public:
  explicit ClassicalPreprocessor(const Params& recipe);
  ClassicalWindow operator()(const Tensor& raw) const;
  int64_t feature_width() const;
 private:
  std::vector<SosFilter> bank_;
  SosFilter slow_;
  int64_t nsub_,slow_step_;
  bool analytic_;
  double shrink_;
};
struct CoreTrainingResult {
  Params model;
  Tensor source_features, source_rows;
  LinearFit head;
};
// Source schema: native_format=1, x[N,41,480]float32, y/stream/person int64[N],
// fit_mask bool[N], held_person int64[H]. Excluded EEG/labels never enter fits.
// Every selected stream must be complete. No learned fields are accepted in recipe.
CoreTrainingResult train_classical_core(const Params& source,const Params& recipe);
}
