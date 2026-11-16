// Common-average reference and per-window fixed SOS filters for the strong core.
#include <vector>

extern "C" int nc_sosfiltfilt(const double*, int, const double*, const double*, int, int, double*);

extern "C" int nc_preprocess(const float* input, const double* bands, const double* bands_zi,
                             const double* low, const double* low_zi, float* filtered, float* slow) {
    if (!input || !bands || !bands_zi || !low || !low_zi || !filtered || !slow) return -1;
    try {
        std::vector<float> car(41 * 480);
        for (int t = 0; t < 480; ++t) {
            float sum = 0;
            for (int c = 0; c < 41; ++c) sum += input[c * 480 + t];
            const float mean = sum / 41.0f;
            for (int c = 0; c < 41; ++c) car[c * 480 + t] = input[c * 480 + t] - mean;
        }
        double x[480], y[480];
        for (int c = 0; c < 41; ++c) {
            for (int t = 0; t < 480; ++t) x[t] = car[c * 480 + t];
            if (nc_sosfiltfilt(x, 480, low, low_zi, 2, 15, y)) return -1;
            for (int i = 0; i < 8; ++i) slow[c * 8 + i] = static_cast<float>(y[30 + 60 * i]);
            for (int b = 0; b < 17; ++b) {
                if (nc_sosfiltfilt(x, 480, bands + b * 24, bands_zi + b * 8, 4, 27, y)) return -1;
                for (int t = 0; t < 480; ++t) filtered[(b * 41 + c) * 480 + t] = static_cast<float>(y[t]);
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
