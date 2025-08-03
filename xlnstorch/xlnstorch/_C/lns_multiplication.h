#ifndef LNS_MULTIPLICATION_H
#define LNS_MULTIPLICATION_H

#include <torch/extension.h>

torch::Tensor mul_forward(
    const torch::Tensor& x,
    const torch::Tensor& y
);
void init_lns_multiplication(py::module& m);

#endif // LNS_MULTIPLICATION_H