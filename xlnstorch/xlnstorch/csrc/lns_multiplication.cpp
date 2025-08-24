#include <torch/extension.h>
#include <ATen/native/cpu/Reduce.h>
#include <cstdint>

#include "pointwise_ops.h"
#include "vectorized_ops.h"

torch::Tensor mul_forward(
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    auto result_sizes = at::infer_size(x.sizes(), y.sizes());
    auto out = torch::empty(result_sizes, x.options().dtype(torch::kInt64));

    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .build();

    at::native::cpu_kernel_vec(
        iter,
        [](int64_t a, int64_t b) -> int64_t {
            return lns::mul(a, b);
        },
        [](int64_vec_t a, int64_vec_t b) -> int64_vec_t {
            return lns::mul_vec(a, b);
        });

    return out;

}

void init_lns_multiplication(py::module& m) {
    m.def(
        "mul_forward",
        &mul_forward,
        "LNS multiplication forward pass",
        py::arg("x"),
        py::arg("y")
    );
}