#ifndef LNS_CONVOLUTION_H
#define LNS_CONVOLUTION_H

#include <torch/extension.h>

torch::Tensor conv1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& base_t,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t groups = 1
);
void init_lns_convolution(py::module& m);

#endif // LNS_CONVOLUTION_H