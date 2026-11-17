#include "ducc0/fft/fft.h"
#include "ducc0/fft/fftnd_impl.h"
#include <complex>
#include <cstddef>
#include <vector>

extern "C" int ni26_hilbert_probe(const float* input, std::complex<float>* output,
                                 size_t bands, size_t channels, size_t samples) {
  if (!input || !output || !bands || !channels || samples != 480)
    return -1;
  try {
    const ducc0::fmav_info::shape_t shape{bands,channels,samples}, axes{2};
    std::vector<std::complex<float>> spectrum(bands*channels*samples);
    ducc0::cfmav<float> raw(input,shape);
    ducc0::vfmav<std::complex<float>> full(spectrum.data(),shape);
    auto half_shape = shape;
    half_shape.back() = samples/2 + 1;
    ducc0::vfmav<std::complex<float>> half(full, half_shape, full.stride());
    ducc0::r2c(raw,half,axes,true,1.0f,1);
    for (size_t group=0; group<bands*channels; ++group)
      for (size_t j=1; j<samples/2; ++j)
        spectrum[group*samples+j] *= 2.0f;
    ducc0::vfmav<std::complex<float>> analytic(output,shape);
    ducc0::c2c(full,analytic,axes,false,static_cast<float>(1.0L/samples),1);
    return 0;
  } catch (...) { return -2; }
}
