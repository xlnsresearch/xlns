#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <cmath>
#include <cstdint>

#include "lns_constants.h"

// pre-compute log(base1) / log(base2) once per call (cheaper than calling log inside the loop)
inline double ratio_log_base(const torch::Tensor& base1, const torch::Tensor& base2) {
    return std::log(base1.item<double>()) / std::log(base2.item<double>());
}

// pre-compute 1 / log(base) once per call (cheaper than calling log inside the loop)
inline double inv_log_base(const torch::Tensor& base) {
    return 1.0 / std::log(base.item<double>());
}


/*
float_to_lns_forward and float_to_lns_backward are the forward and backward passes
for converting floating-point tensors to LNS representation. We can abuse the torch
autograd system to use the backward pass to convert the gradients back to floating-point
representation.
*/

torch::Tensor float_to_lns_forward(const torch::Tensor& x, const torch::Tensor& base) {

    const double inv_log_b = inv_log_base(base);
    auto out = torch::empty_like(x, x.options().dtype(torch::kFloat64));

    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .build();

    at::native::cpu_kernel(
        iter,
        [inv_log_b](double v) -> double {

            if (v == 0.0) {
                return lns::zero;
            }

            int64_t e = llround(std::log(std::abs(v)) * inv_log_b);
            int64_t s = (v < 0.0) ? 1LL : 0LL;

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

            int64_t p = static_cast<int64_t>(grad);

            if ((p | 1LL) == lns::zero_int) {
                return 0.0;
            }

            double exponent = static_cast<double>(p >> 1);
            double sign = (p & 1LL) ? -1.0 : 1.0;

            return sign * std::pow(b, exponent);

        });

    return out;

}


/*
change_base_forward and change_base_backward are the forward and backward passes
for changing the base of the LNS representation. The forward pass converts the LNS
representation from one base to another, while the backward pass converts the gradients
back to the original base. Again, we abuse the backward pass to modify the gradients
despite the fact that the forward pass does not change the true value of the tensor,
just its representation.
*/

torch::Tensor change_base_forward(const torch::Tensor& x, const torch::Tensor& old_base, const torch::Tensor& new_base) {

    const double ratio_log_b = ratio_log_base(old_base, new_base);
    auto out = torch::empty_like(x, x.options().dtype(torch::kFloat64));

    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .build();

    at::native::cpu_kernel(
        iter,
        [ratio_log_b](double v) -> double {

            int64_t p = static_cast<int64_t>(v);

            if ((p | 1LL) == lns::zero_int) {
                return lns::zero;
            }

            double exponent = static_cast<double>(p >> 1);
            int64_t exponent_new = llround(exponent * ratio_log_b);
            int64_t sign_bit = p & 1LL;

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

            int64_t p = static_cast<int64_t>(grad);

            if ((p | 1LL) == lns::zero_int) {
                return lns::zero;
            }

            double exponent = static_cast<double>(p >> 1);
            int64_t exponent_new = llround(exponent * ratio_log_b);
            int64_t sign_bit = p & 1LL;

            return static_cast<double>((exponent_new << 1) | sign_bit);

        });

    return out;

}


void init_lns_utils(py::module& m) {
    m.def(
        "float_to_lns_forward",
        &float_to_lns_forward,
        "Convert float tensor to LNS representation (forward pass)",
        py::arg("x"),
        py::arg("base_t")
    );
    m.def(
        "float_to_lns_backward",
        &float_to_lns_backward,
        "Convert gradients from LNS representation back to float (backward pass)",
        py::arg("grad_output"),
        py::arg("base_t")
    );
    m.def(
        "change_base_forward",
        &change_base_forward,
        "Change base of LNS representation (forward pass)",
        py::arg("x"),
        py::arg("old_base"),
        py::arg("new_base")
    );
    m.def(
        "change_base_backward",
        &change_base_backward,
        "Change base of gradients back to original base (backward pass)",
        py::arg("grad_output"),
        py::arg("old_base"),
        py::arg("new_base")
    );
}