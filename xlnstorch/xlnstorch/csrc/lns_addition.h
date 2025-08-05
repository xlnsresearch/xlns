#ifndef LNS_ADDITION_H
#define LNS_ADDITION_H

#include <torch/extension.h>

torch::Tensor add_forward(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& base_t
);
torch::Tensor sum_forward(
    const torch::Tensor& x,
    const torch::Tensor& base_t,
    const std::vector<int64_t>& dims = {},
    bool keepdim = false
);
void init_lns_addition(py::module& m);

#endif // LNS_ADDITION_H