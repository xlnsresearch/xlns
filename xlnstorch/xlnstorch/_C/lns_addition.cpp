#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <cstdint>

#include "lns_constants.h"
#include "pointwise_ops.h"

torch::Tensor add_forward(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& base_t
) {

    const double base = base_t.item<double>();
    auto result_sizes = at::infer_size(x.sizes(), y.sizes());
    auto out = torch::empty(result_sizes, x.options().dtype(torch::kInt64));

    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .build();

    at::native::cpu_kernel(
        iter,
        [base](int64_t a, int64_t b) -> int64_t {
            return lns::add(a, b, base);
        });

    return out.to(torch::kFloat64);

}

void init_lns_addition(py::module& m) {
    m.def("add_forward", &add_forward, "LNS addition forward pass");
}