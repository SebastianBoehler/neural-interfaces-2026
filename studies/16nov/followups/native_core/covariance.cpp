// Full window and two half-window trace-shrunk covariances for 17 bands.
#include <cmath>

extern "C" int nc_covariances(const float* filtered, double* output) {
    if (!filtered || !output) return -1;
    for (int band = 0; band < 17; ++band) {
        for (int segment = 0; segment < 3; ++segment) {
            const int length = segment == 0 ? 480 : 240;
            const int start = segment == 2 ? 240 : 0;
            float mean[41] = {};
            for (int c = 0; c < 41; ++c) {
                const float* x = filtered + (band * 41 + c) * 480 + start;
                for (int t = 0; t < length; ++t) mean[c] += x[t];
                mean[c] /= length;
            }
            double* dst = output + (band * 3 + segment) * 41 * 41;
            double trace = 0;
            for (int c = 0; c < 41; ++c) {
                const float* x = filtered + (band * 41 + c) * 480 + start;
                for (int d = c; d < 41; ++d) {
                    const float* y = filtered + (band * 41 + d) * 480 + start;
                    double sum = 0;
                    for (int t = 0; t < length; ++t) {
                        const float xc = x[t] - mean[c], yd = y[t] - mean[d];
                        sum += static_cast<double>(xc) * static_cast<double>(yd);
                    }
                    dst[c * 41 + d] = dst[d * 41 + c] = sum / length;
                }
                trace += dst[c * 41 + c];
            }
            for (int c = 0; c < 41; ++c)
                for (int d = 0; d < 41; ++d)
                    dst[c * 41 + d] = .9 * dst[c * 41 + d] + (c == d ? .1 * trace / 41 : 0.0);
        }
    }
    return 0;
}
