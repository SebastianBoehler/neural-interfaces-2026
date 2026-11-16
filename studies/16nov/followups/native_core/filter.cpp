// SciPy-compatible forward/backward SOS filtering for one independent window.
#include <algorithm>
#include <cmath>
#include <vector>

static void forward(std::vector<double>& x, const double* sos, const double* zi, int sections) {
    const double first = x.front();
    for (int s = 0; s < sections; ++s) {
        const double* c = sos + 6 * s;
        double z0 = zi[2 * s] * first, z1 = zi[2 * s + 1] * first;
        for (double& v : x) {
            const double y = c[0] * v + z0;
            z0 = c[1] * v - c[4] * y + z1;
            z1 = c[2] * v - c[5] * y;
            v = y;
        }
    }
}

extern "C" int nc_sosfiltfilt(const double* input, int n, const double* sos,
                              const double* zi, int sections, int pad, double* output) {
    if (!input || !sos || !zi || !output || n <= 0 || sections <= 0 || pad < 0 || pad >= n)
        return -1;
    for (int s = 0; s < sections; ++s)
        if (sos[6 * s + 3] != 1.0) return -1;
    try {
        std::vector<double> x(n + 2 * pad);
        for (int i = 0; i < pad; ++i) x[i] = 2 * input[0] - input[pad - i];
        std::copy(input, input + n, x.begin() + pad);
        for (int i = 0; i < pad; ++i) x[pad + n + i] = 2 * input[n - 1] - input[n - 2 - i];
        forward(x, sos, zi, sections);
        std::reverse(x.begin(), x.end());
        forward(x, sos, zi, sections);
        std::reverse(x.begin(), x.end());
        std::copy(x.begin() + pad, x.begin() + pad + n, output);
        return 0;
    } catch (...) {
        return -1;
    }
}
