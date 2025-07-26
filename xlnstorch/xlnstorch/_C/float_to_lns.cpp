#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <cmath>

// special “zero” value in LNS space.
static constexpr long long LNS_ZERO_INT = (-(1LL << 53)) | 1LL;
static constexpr double LNS_ZERO = static_cast<double>(LNS_ZERO_INT);

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

torch::Tensor float_to_lns_backward(const torch::Tensor& grad_output, const torch::Tensor& base) {

    const double b = base.item<double>();
    auto out = torch::empty_like(grad_output, grad_output.options().dtype(torch::kFloat64));

    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(grad_output)
        .build();

    at::native::cpu_kernel(
        iter,
        [b](double grad) -> double {

            long long p = static_cast<long long>(grad);

            if ((p | 1LL) == LNS_ZERO_INT) {
                return 0.0;
            }

            double exponent = static_cast<double>(p >> 1);
            double sign = (p & 1LL) ? -1.0 : 1.0;

            return sign * std::pow(b, exponent);

        });

    return out;

}

void init_float_to_lns(py::module& m) {
    m.def("float_to_lns_forward", &float_to_lns_forward, "LNS packing forward pass");
    m.def("float_to_lns_backward", &float_to_lns_backward, "LNS packing backward pass");
}