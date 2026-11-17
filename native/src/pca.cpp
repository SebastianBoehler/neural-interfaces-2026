#include "ni26/pca.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Same LP64 symbol/signature as installed scipy.linalg.cython_lapack.pxd.
// Native optimized OpenBLAS/LAPACK provider; no Python or SciPy runtime callback.
extern "C" void scipy_dgesdd_(char*,int*,int*,double*,int*,double*,double*,int*,
                               double*,int*,double*,int*,int*,int*);
namespace ni26 {
TsemBasis fit_tsem_basis(const Tensor& source) {
  torch::NoGradGuard guard;
  require(source.device().is_cpu() && source.scalar_type()==torch::kFloat64 &&
          source.dim()==2 && source.is_contiguous() && !source.requires_grad(),
          "TSEM source must be detached contiguous CPU float64[N,D]");
  auto rows=source.size(0),width=source.size(1);
  require(rows>=256 && width>=rows,"TSEM exact wide-SVD requires 256<=N<=D");
  const auto int_max=std::numeric_limits<int>::max();
  require(rows<=int_max && width<=int_max && width*rows<=int_max,
          "TSEM matrix exceeds the qualified LP64 thin-SVD indexing limit");
  auto mean=torch::zeros({width},source.options());
  auto* means=mean.data_ptr<double>();const auto* values=source.data_ptr<double>();
  // Match NumPy's non-fast-axis float64 mean without a full centering temporary.
  for (int64_t i=0;i<rows;++i) for (int64_t j=0;j<width;++j) {
    auto value=values[i*width+j];
    if (!std::isfinite(value)) require(false,"nonfinite TSEM source");
    means[j]+=value;
  }
  for (int64_t j=0;j<width;++j) {
    means[j]/=rows;require(std::isfinite(means[j]),"TSEM source mean overflow");
  }
  // One Fortran-order centered input: DGESDD destroys this buffer, never source.
  auto centered=torch::empty({width,rows},source.options());
  auto* matrix=centered.data_ptr<double>();
  for (int64_t i0=0;i0<rows;i0+=32) for (int64_t j0=0;j0<width;j0+=64)
    for (auto i=i0;i<std::min(rows,i0+32);++i) for (auto j=j0;j<std::min(width,j0+64);++j) {
      auto value=values[i*width+j]-means[j];
      if (!std::isfinite(value)) require(false,"TSEM source centering overflow");
      matrix[j*rows+i]=value;
    }
  auto singular=torch::empty({rows},source.options());
  auto left=torch::empty({rows,rows},source.options());
  auto right=torch::empty({width,rows},source.options());
  std::vector<int> integer_work(8*rows);
  int m=int(rows),n=int(width),lda=m,ldu=m,ldvt=m,work_size=-1,info=0;
  char job='S';double query=0.;
  scipy_dgesdd_(&job,&m,&n,matrix,&lda,singular.data_ptr<double>(),left.data_ptr<double>(),
    &ldu,right.data_ptr<double>(),&ldvt,&query,&work_size,integer_work.data(),&info);
  require(info==0,"native DGESDD workspace query failed info="+std::to_string(info));
  require(std::isfinite(query) && query>=1 && query<=int_max,"invalid DGESDD LP64 workspace size");
  work_size=int(std::ceil(query));
  std::vector<double> work(work_size);
  scipy_dgesdd_(&job,&m,&n,matrix,&lda,singular.data_ptr<double>(),left.data_ptr<double>(),
    &ldu,right.data_ptr<double>(),&ldvt,work.data(),&work_size,integer_work.data(),&info);
  require(info==0,"native exact DGESDD failed info="+std::to_string(info));
  finite(singular,"TSEM singular values");
  auto basis=torch::empty({width,256},source.options().dtype(torch::kFloat32));
  auto* out=basis.data_ptr<float>();const auto* vt=right.data_ptr<double>();
  // Vt is column-major [N,D]. Strict > preserves argmax's first tie.
  for (int64_t component=0;component<256;++component) {
    int64_t best=0;double maximum=-1.;
    for (int64_t j=0;j<width;++j) {
      auto value=vt[j*rows+component];
      if (!std::isfinite(value)) require(false,"nonfinite TSEM right singular vector");
      if (std::abs(value)>maximum) {maximum=std::abs(value);best=j;}
    }
    auto pivot=vt[best*rows+component];
    double sign=(pivot>0.)-(pivot<0.); // sklearn numpy.sign, including zero
    for (int64_t j=0;j<width;++j) out[j*256+component]=float(sign*vt[j*rows+component]);
  }
  // U/Vt/work/centered are released on return; no eigensystem cache survives.
  return {mean,basis,singular,int64_t(work_size)};
}
}
