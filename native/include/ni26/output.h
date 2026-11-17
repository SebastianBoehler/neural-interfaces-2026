#pragma once
#include "ni26/math.h"
#include <memory>
#include <optional>
namespace ni26 {
struct OutputStats { int64_t windows=0,refits=0,fused=0,single_class=0,errors=0; std::optional<int64_t> stopped; bool dead=false; };
class CountFilter {
 public: explicit CountFilter(const Params& z); ~CountFilter(); CountFilter(const CountFilter&); CountFilter& operator=(const CountFilter&); void reset(); Tensor step(const Tensor& x,const Tensor& p); OutputStats stats() const;
 private: struct Impl; std::unique_ptr<Impl> impl_;
};
class CausalPers {
 public: explicit CausalPers(const Params& z); ~CausalPers(); CausalPers(const CausalPers&); CausalPers& operator=(const CausalPers&); void reset(); Tensor step(const Tensor& x,const Tensor& p,const Tensor& count_p,bool recognized); OutputStats stats() const;
 private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
