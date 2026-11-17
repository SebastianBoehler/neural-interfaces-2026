#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <vector>

extern "C" void ni26_cov_witness(const float* x, double* out) {
  for (int row=0; row<41; ++row) {
    for (int col=0; col<=row; ++col) {
      const auto *a=x+480*row, *b=x+480*col;
      double lanes[2]={};
      for (int i=0; i<480; i+=8)
        for (int j=6; j>=0; j-=2)
          for (int l=0; l<2; ++l)
            lanes[l]+=double(a[i+j+l])*double(b[i+j+l]);
      out[row*41+col]=out[col*41+row]=(lanes[0]+lanes[1])/480.;
    }
  }
}

using i64 = std::int64_t;
using Eigh = void (*)(char*,char*,i64*,double*,i64*,double*,double*,i64*,i64*,i64*,i64*);
using Gemm = void (*)(int,int,int,i64,i64,i64,double,const double*,i64,const double*,i64,double,double*,i64);

extern "C" int ni26_inverse_root(const char* path, const double* r, double* p) {
  void* lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if (!lib) return 1;
  auto eigh=reinterpret_cast<Eigh>(dlsym(lib,"scipy_dsyevd_64_"));
  auto gemm=reinterpret_cast<Gemm>(dlsym(lib,"scipy_cblas_dgemm64_"));
  if (!eigh || !gemm) { dlclose(lib); return 2; }
  char job='V', lower='L';
  i64 n=41, info=0, lwork=-1, liwork=-1, iw=0;
  double wq=0.;
  std::array<double,41> eval;
  std::array<double,41*41> vec, vr, v_scaled;
  for (int i=0;i<41;++i)
    for (int j=0;j<41;++j) vec[j*41+i]=r[i*41+j];
  eigh(&job,&lower,&n,vec.data(),&n,eval.data(),&wq,&lwork,&iw,&liwork,&info);
  if (info || wq < 1 || iw < 1) { dlclose(lib);return 3; }
  lwork=static_cast<i64>(wq);
  liwork=iw;
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<i64> iwork(static_cast<std::size_t>(liwork));
  eigh(&job,&lower,&n,vec.data(),&n,eval.data(),work.data(),&lwork,iwork.data(),&liwork,&info);
  if (info) { dlclose(lib); return 4; }
  for (int col=0;col<41;++col) {
    double scale=1./std::sqrt(std::max(eval[col],1e-12));
    for (int row=0;row<41;++row) {
      vr[row*41+col]=vec[col*41+row];
      v_scaled[row*41+col]=vr[row*41+col]*scale;
    }
  }
  gemm(101,111,112,n,n,n,1.,v_scaled.data(),n,vr.data(),n,0.,p,n);
  dlclose(lib);
  return 0;
}
