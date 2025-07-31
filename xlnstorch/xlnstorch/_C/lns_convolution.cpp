#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <algorithm>

#include "lns_constants.h"
#include "pointwise_ops.h"
#include "lns_addition.h"

torch::Tensor conv1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& base_t,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t groups = 1
) {

    const double base = base_t.item<double>();

    bool squeeze_batch = false;
    if (input.dim() == 2) {
        input.unsqueeze_(0); // Add batch dimension if missing
        squeeze_batch = true;
    }

    TORCH_CHECK(input.dim() == 3 && weight.dim() == 3,
        "Input and weight must be 3D tensors (N, C_in, L_in) and (C_out, C_in / groups, K) respectively");

    const int64_t N = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Lin = input.size(2);
    const int64_t Cout = weight.size(0);
    const int64_t K = weight.size(2);
    TORCH_CHECK(Cin % groups == 0 && Cout % groups == 0,
        "Input channels and output channels must be divisible by groups");

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;
    TORCH_CHECK(weight.size(1) == Cin_g, "weight second dim must equal C_in / groups");

    if (bias.has_value()) {
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == Cout, "bias must be 1-D with size C_out");
    }

    const int64_t Lout = (Lin + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(Lout > 0 , "output length is non-positive");

    torch::Tensor output = at::empty({N, Cout, Lout}, input.options());
    const int64_t* in = input.data_ptr<int64_t>();
    const int64_t* w = weight.data_ptr<int64_t>();
    int64_t* out = output.data_ptr<int64_t>();
    const int64_t* b = bias.has_value() ? bias->data_ptr<int64_t>() : nullptr;

    const int64_t in_stride_N   = Cin * Lin;
    const int64_t in_stride_C   = Lin;
    const int64_t w_stride_Cout = Cin_g * K;
    const int64_t w_stride_Cin  = K;
    const int64_t out_stride_N  = Cout * Lout;
    const int64_t out_stride_C  = Lout;

    const int64_t work_items = N * Cout;
    const int64_t grain = 64; // good default; change if needed

    at::parallel_for(
        /*begin*/ 0,
        /*end*/ work_items,
        /*grain_size*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            for (int64_t linear = begin; linear < end; ++linear) {
                const int64_t n = linear / Cout; // batch index
                const int64_t oc = linear % Cout; // output channel

                const int64_t g = oc / Cout_g;
                const int64_t oc_in_g = oc % Cout_g;

                const int64_t* w_g = w + (g * Cout_g + oc_in_g) * w_stride_Cout;
                const int64_t* in_g = in + n * in_stride_N + g * Cin_g * Lin;
                int64_t* out_g = out + n * out_stride_N + oc * out_stride_C;

                // slide along output time dimension
                for (int64_t x_out = 0; x_out < Lout; ++x_out) {
                    const int64_t x_in0 = x_out * stride - padding;

                    // valid kernel taps for this x_out
                    const int64_t k_min = std::max<int64_t>(0, (-x_in0 + dilation - 1) / dilation);
                    const int64_t k_max = std::min<int64_t>(K, (Lin - x_in0 + dilation - 1) / dilation);

                    int64_t acc = b ? b[oc] : lns::zero_int;
                    for (int64_t ic_g = 0; ic_g < Cin_g; ++ic_g) {
                        const int64_t* in_c = in_g + ic_g * in_stride_C;
                        const int64_t* w_c  = w_g + ic_g * w_stride_Cin;

                        // kernel loop (already range checked)
                        for (int64_t k = k_min; k < k_max; ++k) {
                            const int64_t in_val = in_c[x_in0 + k * dilation];
                            const int64_t w_val = w_c[k];
                            acc = lns::add(acc, lns::mul(in_val, w_val), base);
                        }
                    }

                    out_g[x_out] = acc;
                }
            }
        });

    return squeeze_batch ? output.squeeze(0) : output;
}

std::vector<torch::Tensor> conv1d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& base_t,
    const bool bias_defined,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t groups = 1
) {

    const double base = base_t.item<double>();

    bool squeeze_batch = false;

    if (input.dim() == 2) {
        input.unsqueeze_(0);
        grad_output.unsqueeze_(0);
        squeeze_batch = true;
    }

    const int64_t N = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Lin = input.size(2);
    const int64_t Cout = weight.size(0);
    const int64_t K = weight.size(2);

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    const int64_t Lout = grad_output.size(2);

    torch::Tensor grad_input = at::empty_like(input);
    torch::Tensor grad_weight = at::empty_like(weight);
    torch::Tensor grad_bias = bias_defined ? at::empty({Cout}, input.options()) : torch::Tensor();

    grad_input.fill_(lns::zero_int);
    grad_weight.fill_(lns::zero_int);
    if (bias_defined) grad_bias.fill_(lns::zero_int);

    const int64_t* in_ptr = input.data_ptr<int64_t>();
    const int64_t* w_ptr = weight.data_ptr<int64_t>();
    const int64_t* go_ptr = grad_output.data_ptr<int64_t>();

    int64_t* gi_ptr = grad_input.data_ptr<int64_t>();
    int64_t* gw_ptr = grad_weight.data_ptr<int64_t>();
    int64_t* gb_ptr = bias_defined ? grad_bias.data_ptr<int64_t>() : nullptr;

    const int64_t in_stride_N = Cin * Lin;
    const int64_t in_stride_C = Lin;

    const int64_t w_stride_Cout = Cin_g * K;
    const int64_t w_stride_Cin = K;

    const int64_t go_stride_N = Cout * Lout;
    const int64_t go_stride_C = Lout;

    const int64_t gi_stride_N = in_stride_N;
    const int64_t gi_stride_C = in_stride_C;

    const int64_t gw_stride_Cout = w_stride_Cout;
    const int64_t gw_stride_Cin = w_stride_Cin;

    const int64_t work_items = N * Cout;
    const int64_t grain = 16; // good default; change if needed

    at::parallel_for(
        /*begin*/ 0,
        /*end*/ work_items,
        /*grain_size*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            // Thread-local scratch copies of grad_weight / grad_bias
            // to avoid costly atomics. We accumulate locally and add
            // to the global tensors at the end of the thread range.
            torch::Tensor gw_private = at::empty_like(weight);
            gw_private.fill_(lns::zero_int);
            torch::Tensor gb_private;
            if (bias_defined) {
                gb_private = at::empty({Cout} , input.options());
                gb_private.fill_(lns::zero_int);
            }

            int64_t* gw_p = gw_private.data_ptr<int64_t>();
            int64_t* gb_p = bias_defined ? gb_private.data_ptr<int64_t>() : nullptr;

            for (int64_t linear = begin ; linear < end ; ++linear) {
                const int64_t n = linear / Cout; // batch index
                const int64_t co = linear % Cout; // output channel

                const int64_t group = co / Cout_g;
                const int64_t ci_group_beg = group * Cin_g;
                const int64_t go_base = n * go_stride_N + co * go_stride_C;

                for (int64_t lo = 0; lo < Lout; ++lo) {
                    const int64_t go_val = go_ptr[go_base + lo];

                    if (bias_defined) gb_p[co] = lns::add(gb_p[co], go_val, base);

                    for (int64_t k = 0; k < K; ++k) {
                        const int64_t li = lo * stride + k * dilation - padding;
                        if (li < 0 || li >= Lin) continue; // skip padding

                        const int64_t w_k_base = co * w_stride_Cout + k;
                        for (int64_t ci_rel = 0; ci_rel < Cin_g; ++ci_rel) {
                            const int64_t ci = ci_group_beg + ci_rel;

                            const int64_t in_idx = n * in_stride_N + ci * in_stride_C + li;
                            const int64_t w_idx = w_k_base + ci_rel * w_stride_Cin;

                            const int64_t prod_in = lns::mul(go_val, w_ptr[w_idx]);
                            gi_ptr[in_idx] = lns::add(gi_ptr[in_idx], prod_in, base);

                            const int64_t prod_w = lns::mul(go_val, in_ptr[in_idx]);
                            gw_p[w_idx] = lns::add(gw_p[w_idx], prod_w, base);
                        }
                    }
                }
            }

            // reduction - add thread-local results to global tensors

            const int64_t W = weight.numel();

            #pragma omp critical
            {
                grad_weight.copy_(add_forward(grad_weight, gw_private, base_t));
                if (bias_defined)
                    grad_bias.copy_(add_forward(grad_bias, gb_private, base_t));
            }

        });

    if (squeeze_batch) {
        grad_input = grad_input.squeeze(0);
    }

    if (bias_defined) return {grad_input, grad_weight, grad_bias};
    else return {grad_input, grad_weight};
}

void init_lns_convolution(py::module& m) {
    m.def("conv1d_forward", &conv1d_forward, "LNS conv1d forward pass");
    m.def("conv1d_backward", &conv1d_backward, "LNS conv1d backward pass");
}