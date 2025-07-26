#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <cmath>

// special “zero” value in LNS space.
static constexpr long long LNS_ZERO_INT = (-(1LL << 53)) | 1LL;
static constexpr double LNS_ZERO = static_cast<double>(LNS_ZERO_INT);

// pre-compute log(base1) / log(base2) once per call (cheaper than calling log inside the loop)
inline double ratio_log_base(const torch::Tensor& base1, const torch::Tensor& base2) {
    return std::log(base1.item<double>()) / std::log(base2.item<double>());
}

torch::Tensor change_base_forward(const torch::Tensor& x, const torch::Tensor& old_base, const torch::Tensor& new_base) {

    const double ratio_log_b = ratio_log_base(old_base, new_base);
    auto out = torch::empty_like(x, x.options().dtype(torch::kFloat64));

    // TensorIterator lets us write a single element-wise kernel
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .build();

    at::native::cpu_kernel(
        iter,
        [ratio_log_b](double v) -> double {

            long long p = static_cast<long long>(v);

            if ((p | 1LL) == LNS_ZERO_INT) {
                return LNS_ZERO;
            }

            double exponent = static_cast<double>(p >> 1);
            long long exponent_new = llround(exponent * ratio_log_b);
            long long sign_bit = p & 1LL;

            return static_cast<double>((exponent_new << 1) | sign_bit);

        });

    return out;

}

torch::Tensor change_base_backward(const torch::Tensor& grad_output, const torch::Tensor& old_base, const torch::Tensor& new_base) {

    const double ratio_log_b = ratio_log_base(new_base, old_base);
    auto out = torch::empty_like(grad_output, grad_output.options().dtype(torch::kFloat64));

    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(grad_output)
        .build();

    at::native::cpu_kernel(
        iter,
        [ratio_log_b](double grad) -> double {

            long long p = static_cast<long long>(grad);

            if ((p | 1LL) == LNS_ZERO_INT) {
                return LNS_ZERO;
            }

            double exponent = static_cast<double>(p >> 1);
            long long exponent_new = llround(exponent * ratio_log_b);
            long long sign_bit = p & 1LL;

            return static_cast<double>((exponent_new << 1) | sign_bit);

        });

    return out;

}

void init_change_base(py::module& m) {
    m.def("change_base_forward", &change_base_forward, "LNS change base forward pass");
    m.def("change_base_backward", &change_base_backward, "LNS change base backward pass");
}