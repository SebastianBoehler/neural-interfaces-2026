// Source-span math behind the existing SciPy reference optimizer.
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
namespace {
thread_local std::string error;
const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
using Clock = std::chrono::steady_clock;
void require(bool value, const char* why) { if (!value) throw std::invalid_argument(why); }
struct Context { torch::Tensor x, y, z, basis, logits, gradient; int64_t n, d, rank; double c; bool full=false; };
template<class F> int checked(F operation) {
    try { error.clear(); torch::NoGradGuard guard; operation(); return 0; }
    catch (const std::exception& e) { error=e.what(); return -1; }
    catch (...) { error="unknown reference-span error"; return -1; }
}
Context& context(void* raw) {
    require(raw, "missing source-span context"); return *static_cast<Context*>(raw);
}
torch::Tensor parameters(Context& h, const double* raw) {
    require(raw, "missing reduced parameters");
    auto result = torch::from_blob(const_cast<double*>(raw), {h.rank+1,3}, options);
    for (int64_t i=0;i<3*(h.rank+1);++i) require(std::isfinite(raw[i]),"nonfinite reduced parameters");
    return result;
}
}
extern "C" const char* s343r_error() { return error.c_str(); }
extern "C" void s343r_destroy(void* raw) { delete static_cast<Context*>(raw); }
extern "C" int s343r_create(const double* x, const int64_t* y, int64_t n, int64_t d,
    double c, void** output, double* stats) {
    return checked([&] {
        require(x && y && output && stats && n>3 && d>=n && std::isfinite(c) && c>0, "invalid source-span input");
        *output=nullptr;
        auto tick=Clock::now();
        for (int64_t i=0;i<n*d;++i) require(std::isfinite(x[i]), "nonfinite source matrix");
        bool seen[3]{};
        for (int64_t i=0;i<n;++i) { require(y[i]>=0 && y[i]<3, "invalid source labels"); seen[y[i]]=true; }
        require(seen[0] && seen[1] && seen[2], "training requires all three classes");
        auto h=std::make_unique<Context>(); h->n=n;h->d=d;h->c=c;
        h->x=torch::from_blob(const_cast<double*>(x),{n,d},options);
        h->y=torch::from_blob(const_cast<int64_t*>(y),{n},torch::kInt64);
        auto gram=torch::matmul(h->x,h->x.t());
        auto decomposition=torch::linalg_eigh(gram);
        auto values=std::get<0>(decomposition), vectors=std::get<1>(decomposition);
        auto largest=values.max().item<double>(), smallest=values.min().item<double>();
        auto cutoff=std::numeric_limits<double>::epsilon()*n*largest;
        require(largest>0 && smallest>=-cutoff,"materially negative Gram eigenvalue");
        auto retain=values>cutoff;
        auto roots=torch::sqrt(values.index({retain}));
        auto kept=vectors.index({torch::indexing::Slice(),retain});
        h->z=(kept*roots).contiguous(); h->basis=(kept/roots).contiguous(); h->rank=h->z.size(1);
        auto relative=(gram-torch::matmul(h->z,h->z.t())).norm().item<double>()/gram.norm().item<double>();
        require(relative<=1e-10,"source-span Gram reconstruction failed");
        stats[0]=std::chrono::duration<double>(Clock::now()-tick).count();
        stats[1]=h->rank;stats[2]=smallest;stats[3]=largest;stats[4]=cutoff;
        stats[5]=values.index({~retain}).abs().sum().item<double>();stats[6]=relative;
        *output=h.release();
    });
}
extern "C" int s343f_create(const double* x, const int64_t* y, int64_t n, int64_t d,
    double c, void** output, double* stats) {
    return checked([&] {
        require(x && y && output && stats && n>3 && d>0 && std::isfinite(c) && c>0,"invalid full-width input");
        *output=nullptr;auto tick=Clock::now();
        for (int64_t i=0;i<n*d;++i) require(std::isfinite(x[i]),"nonfinite source matrix");
        bool seen[3]{};
        for (int64_t i=0;i<n;++i) { require(y[i]>=0 && y[i]<3,"invalid source labels");seen[y[i]]=true; }
        require(seen[0] && seen[1] && seen[2],"training requires all three classes");
        auto h=std::make_unique<Context>();h->n=n;h->d=d;h->rank=d;h->c=c;h->full=true;
        h->x=torch::from_blob(const_cast<double*>(x),{n,d},options);h->z=h->x;
        h->y=torch::from_blob(const_cast<int64_t*>(y),{n},torch::kInt64);
        std::fill(stats,stats+7,0.);stats[0]=std::chrono::duration<double>(Clock::now()-tick).count();stats[1]=d;
        *output=h.release();
    });
}
extern "C" int s343r_objective(void* raw, const double* theta, double* loss, double* gradient) {
    return checked([&] {
        auto& h=context(raw);require(loss && gradient,"missing objective outputs");
        auto p=parameters(h,theta), weights=p.slice(0,0,h.rank);
        if (!h.logits.defined()) {
            h.logits=torch::empty({h.n,3},options);h.gradient=torch::empty({h.rank+1,3},options);
        }
        torch::mm_out(h.logits,h.z,weights);
        auto residual=h.logits.data_ptr<double>();auto target=h.y.data_ptr<int64_t>();
        double value=0.,bias[3]{};
        for (int64_t i=0;i<h.n;++i) {
            double z[3],largest=-INFINITY,denominator=0.;
            for (int k=0;k<3;++k) { z[k]=residual[3*i+k]+theta[3*h.rank+k];largest=std::max(largest,z[k]); }
            for (double v:z) denominator+=std::exp(v-largest);
            value+=largest+std::log(denominator)-z[target[i]];
            for (int k=0;k<3;++k) { double v=(std::exp(z[k]-largest)/denominator-(target[i]==k))/h.n;
                residual[3*i+k]=v;bias[k]+=v; }
        }
        auto dw=h.gradient.slice(0,0,h.rank);
        torch::mm_out(dw,h.z.t(),h.logits);
        auto g=h.gradient.data_ptr<double>();double penalty=0.;
        for (int64_t i=0;i<3*h.rank;++i) {
            penalty+=theta[i]*theta[i];g[i]+=theta[i]/(h.c*h.n);
            require(std::isfinite(g[i]),"nonfinite objective gradient");
        }
        for (int k=0;k<3;++k) { g[3*h.rank+k]=bias[k];require(std::isfinite(bias[k]),"nonfinite bias gradient"); }
        *loss=value/h.n+penalty/(2*h.c*h.n);
        require(std::isfinite(*loss),"nonfinite objective");
        std::memcpy(gradient,g,3*(h.rank+1)*sizeof(double));
    });
}
extern "C" int s343r_original_gradient(void* raw, const double* gradient, double* maximum) {
    return checked([&] {
        auto& h=context(raw);require(maximum,"missing gradient maximum");
        auto reduced=parameters(h,gradient);
        if (h.full) { *maximum=0.;for (int64_t i=0;i<3*(h.rank+1);++i) *maximum=std::max(*maximum,std::abs(gradient[i]));return; }
        auto dual=torch::matmul(h.basis,reduced.slice(0,0,h.rank));
        auto full=torch::matmul(h.x.t(),dual);
        *maximum=0.;auto data=full.data_ptr<double>();
        for (int64_t i=0;i<3*h.d;++i) *maximum=std::max(*maximum,std::abs(data[i]));
        for (int k=0;k<3;++k) *maximum=std::max(*maximum,std::abs(gradient[3*h.rank+k]));
        require(std::isfinite(*maximum),"nonfinite original-space gradient");
    });
}
extern "C" int s343r_reconstruct(void* raw, const double* theta, double* output) {
    return checked([&] {
        auto& h=context(raw);require(output,"missing full-width parameters");
        auto p=parameters(h,theta);
        if (h.full) { auto full=p.t().contiguous();std::memcpy(output,full.data_ptr<double>(),3*(h.d+1)*sizeof(double));return; }
        auto dual=torch::matmul(h.basis,p.slice(0,0,h.rank));
        auto weights=torch::matmul(h.x.t(),dual);
        auto full=torch::cat({weights,p.select(0,h.rank).unsqueeze(0)},0).t().contiguous();
        require(torch::isfinite(full).all().item<bool>(),"nonfinite full parameters");
        std::memcpy(output,full.data_ptr<double>(),3*(h.d+1)*sizeof(double));
    });
}
extern "C" int s343r_gradient_vector(void* raw, const double* gradient, double* output) {
    return checked([&] {
        auto& h=context(raw);require(output,"missing full gradient output");
        auto reduced=parameters(h,gradient);
        torch::Tensor weights;
        if (h.full) weights=reduced.slice(0,0,h.rank);
        else weights=torch::matmul(h.x.t(),torch::matmul(h.basis,reduced.slice(0,0,h.rank)));
        auto full=torch::cat({weights,reduced.select(0,h.rank).unsqueeze(0)},0).t().contiguous();
        std::memcpy(output,full.data_ptr<double>(),3*(h.d+1)*sizeof(double));
    });
}
