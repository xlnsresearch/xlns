#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <cmath>

// special “zero” value in LNS space.
static constexpr double LNS_ZERO = static_cast<double>( (-(1LL << 53)) | 1LL );

// pre-compute 1 / log(base) once per call (cheaper than calling log inside the loop)
inline double inv_log_base(const torch::Tensor& base) {
    return 1.0 / std::log(base.item<double>());
}

torch::Tensor float_to_lns_forward(const torch::Tensor& x, const torch::Tensor& base) {

    const double inv_log_b = inv_log_base(base);
    auto out = torch::empty_like(x, x.options().dtype(torch::kFloat64));

    // TensorIterator lets us write a single element-wise kernel
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .build();

    at::native::cpu_kernel(
        iter,
        [inv_log_b](double v) -> double {
            if (v == 0.0) {
                return LNS_ZERO;
            }
            long long e = llround(std::log(std::abs(v)) * inv_log_b);
            long long s = (v < 0.0) ? 1LL : 0LL;
            return static_cast<double>((e << 1) | s);
        });

    return out;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float_to_lns_forward", &float_to_lns_forward, "LNS packing forward pass");
}