#pragma once
#include "ni26/artifact.h"
#include <map>

namespace ni26 {
// Split metadata is explicit. Raw excluded EEG may be poisoned and is never read.
struct SourceData {
  Tensor raw, stream, person, fit_mask, held_person;
};
struct SourceSelection {
  Tensor rows;
  std::map<int64_t, int64_t> selected_per_stream;
};
SourceSelection select_source_rows(const SourceData &source);
} // namespace ni26
