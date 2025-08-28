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

    const int64_t in_stride_N = Cin * Lin;
    const int64_t in_stride_C = Lin;

    const int64_t w_stride_Cout = Cin_g * K;
    const int64_t w_stride_Cin = K;

    const int64_t go_stride_N = Cout * Lout;
    const int64_t go_stride_C = Lout;

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

torch::Tensor conv2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& base_t,
    int64_t stride_h = 1,
    int64_t stride_w = 1,
    int64_t padding_h = 0,
    int64_t padding_w = 0,
    int64_t dilation_h = 1,
    int64_t dilation_w = 1,
    int64_t groups = 1)
{
    const double base = base_t.item<double>();

    bool squeeze_batch = false;
    if (input.dim() == 3) {
        input.unsqueeze_(0);
        squeeze_batch = true;
    }

    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
        "Expected 4-D input (N, C_in, H, W) and 4-D weight "
        "(C_out, C_in/groups, K_h, K_w)");

    const int64_t N = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Hin = input.size(2);
    const int64_t Win = input.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK(Cin % groups == 0 && Cout % groups == 0, "C_in and C_out must be divisible by groups");

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    TORCH_CHECK(weight.size(1) == Cin_g, "weight second dim must equal C_in / groups");

    if (bias.has_value())
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == Cout,
                    "bias must be 1-D with length C_out");

    const int64_t Hout = (Hin + 2 * padding_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
    const int64_t Wout = (Win + 2 * padding_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;

    TORCH_CHECK(Hout > 0 && Wout > 0, "Output size is non-positive");

    torch::Tensor output = at::empty({N, Cout, Hout, Wout}, input.options());

    const int64_t* in = input.data_ptr<int64_t>();
    const int64_t* w = weight.data_ptr<int64_t>();
    int64_t* out = output.data_ptr<int64_t>();
    const int64_t* b = bias.has_value() ? bias->data_ptr<int64_t>() : nullptr;

    const int64_t in_stride_N = Cin * Hin * Win;
    const int64_t in_stride_C = Hin * Win;
    const int64_t in_stride_H = Win;

    const int64_t w_stride_Cout = Cin_g * Kh * Kw;
    const int64_t w_stride_Cin = Kh * Kw;
    const int64_t w_stride_Kh = Kw;

    const int64_t out_stride_N = Cout * Hout * Wout;
    const int64_t out_stride_C = Hout * Wout;

    const int64_t work_items = N * Cout;
    const int64_t grain = 16;

    at::parallel_for(
        /*begin*/ 0,
        /*end*/   work_items,
        /*grain*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            for (int64_t linear = begin; linear < end; ++linear) {

                const int64_t n = linear / Cout; // batch index
                const int64_t oc = linear % Cout; // output channel index

                const int64_t g = oc / Cout_g;
                const int64_t oc_in_g = oc % Cout_g;

                const int64_t* w_g = w + (g * Cout_g + oc_in_g) * w_stride_Cout;
                const int64_t* in_g = in + n * in_stride_N + g * Cin_g * Hin * Win;
                int64_t* out_g = out + n * out_stride_N + oc * out_stride_C;

                for (int64_t y_out = 0; y_out < Hout; ++y_out) {
                    const int64_t y_in0 = y_out * stride_h - padding_h;

                    const int64_t kh_min = std::max<int64_t>(0, (-y_in0 + dilation_h - 1) / dilation_h);
                    const int64_t kh_max = std::min<int64_t>(Kh, (Hin - y_in0 + dilation_h - 1) / dilation_h);

                    for (int64_t x_out = 0; x_out < Wout; ++x_out) {
                        const int64_t x_in0 = x_out * stride_w - padding_w;

                        const int64_t kw_min = std::max<int64_t>(0, (-x_in0 + dilation_w - 1) / dilation_w);
                        const int64_t kw_max = std::min<int64_t>(Kw, (Win - x_in0 + dilation_w - 1) / dilation_w);

                        int64_t acc = b ? b[oc] : lns::zero_int;
                        for (int64_t ic_g = 0; ic_g < Cin_g; ++ic_g) {

                            const int64_t* in_c = in_g + ic_g * in_stride_C;
                            const int64_t* w_c  = w_g + ic_g * w_stride_Cin;

                            for (int64_t kh = kh_min; kh < kh_max; ++kh) {
                                const int64_t in_row_offset = (y_in0 + kh * dilation_h) * in_stride_H;
                                const int64_t* in_row = in_c + in_row_offset;
                                const int64_t* w_row = w_c + kh * w_stride_Kh;

                                for (int64_t kw = kw_min; kw < kw_max; ++kw) {
                                    const int64_t in_val = in_row[x_in0 + kw * dilation_w];
                                    const int64_t w_val = w_row[kw];

                                    acc = lns::add(acc, lns::mul(in_val, w_val), base);
                                }
                            }
                        }

                        out_g[y_out * Wout + x_out] = acc;
                    }
                }
            }

    });

    return squeeze_batch ? output.squeeze(0) : output;
}

std::vector<torch::Tensor> conv2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& base_t,
    const bool bias_defined,
    int64_t stride_h = 1,
    int64_t stride_w = 1,
    int64_t padding_h = 0,
    int64_t padding_w = 0,
    int64_t dilation_h = 1,
    int64_t dilation_w = 1,
    int64_t groups = 1)
{
    const double base = base_t.item<double>();

    bool squeeze_batch = false;
    if (input.dim() == 3) {
        input.unsqueeze_(0);
        grad_output.unsqueeze_(0);
        squeeze_batch = true;
    }

    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "conv2d_backward expects 4-D input and weight");

    const int64_t N = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Hin = input.size(2);
    const int64_t Win = input.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK(Cin % groups == 0 && Cout % groups == 0, "C_in and C_out must be divisible by groups");

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    const int64_t Hout = grad_output.size(2);
    const int64_t Wout = grad_output.size(3);

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

    const int64_t in_stride_N = Cin * Hin * Win;
    const int64_t in_stride_C = Hin * Win;
    const int64_t in_stride_H = Win;

    const int64_t w_stride_Cout = Cin_g * Kh * Kw;
    const int64_t w_stride_Cin = Kh * Kw;
    const int64_t w_stride_Kh = Kw;

    const int64_t go_stride_N = Cout * Hout * Wout;
    const int64_t go_stride_C = Hout * Wout;
    const int64_t go_stride_H = Wout;

    // grad_input strides equal input strides    

    const int64_t work_items = N * Cout;
    const int64_t grain = 16;

    at::parallel_for(
        /*begin*/ 0,
        /*end*/ work_items,
        /*grain_size*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            // Thread-local scratch copies
            torch::Tensor gw_private = at::empty_like(weight);
            gw_private.fill_(lns::zero_int);

            torch::Tensor gb_private;
            if (bias_defined) {
                gb_private = at::empty({Cout}, input.options());
                gb_private.fill_(lns::zero_int);
            }

            int64_t* gw_p = gw_private.data_ptr<int64_t>();
            int64_t* gb_p = bias_defined ? gb_private.data_ptr<int64_t>() : nullptr;

            for (int64_t linear = begin; linear < end; ++linear) {

                const int64_t n = linear / Cout; // batch idx
                const int64_t co = linear % Cout; // output channel idx

                const int64_t g = co / Cout_g;
                const int64_t co_in_group = co % Cout_g;
                const int64_t ci_group_begin = g * Cin_g;

                const int64_t go_base = n * go_stride_N + co * go_stride_C;
                const int64_t in_base = n * in_stride_N;
                const int64_t w_base = (g * Cout_g + co_in_group) * w_stride_Cout;

                for (int64_t y_out = 0; y_out < Hout; ++y_out) {
                    const int64_t go_row_base = go_base + y_out * go_stride_H;
                    const int64_t y_in_origin = y_out * stride_h - padding_h;

                    for (int64_t x_out = 0; x_out < Wout; ++x_out) {
                        const int64_t go_offset = go_row_base + x_out;
                        const int64_t go_val = go_ptr[go_offset];

                        if (bias_defined)
                            gb_p[co] = lns::add(gb_p[co], go_val, base);

                        for (int64_t kh = 0; kh < Kh; ++kh) {
                            const int64_t y_in = y_in_origin + kh * dilation_h;
                            if (y_in < 0 || y_in >= Hin) continue;

                            const int64_t w_row_base = w_base + kh * w_stride_Kh;
                            const int64_t in_row_base = in_base + (y_in * in_stride_H);

                            for (int64_t kw = 0; kw < Kw; ++kw) {
                                const int64_t x_in = x_out * stride_w - padding_w + kw * dilation_w;
                                if (x_in < 0 || x_in >= Win) continue;

                                const int64_t w_col_offset = w_row_base + kw;
                                const int64_t in_col_base = in_row_base + x_in;

                                for (int64_t ci_rel = 0; ci_rel < Cin_g; ++ci_rel) {
                                    const int64_t ci = ci_group_begin + ci_rel;

                                    const int64_t in_idx = in_col_base + ci * in_stride_C;
                                    const int64_t w_idx = w_col_offset + ci_rel * w_stride_Cin;

                                    const int64_t prod_in = lns::mul(go_val, w_ptr[w_idx]);
                                    gi_ptr[in_idx] = lns::add(gi_ptr[in_idx], prod_in, base);

                                    const int64_t prod_w  = lns::mul(go_val, in_ptr[in_idx]);
                                    gw_p[w_idx] = lns::add(gw_p[w_idx], prod_w, base);
                                }
                            }
                        }
                    }
                }
            }

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

torch::Tensor conv3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& base_t,
    int64_t stride_d = 1,
    int64_t stride_h = 1,
    int64_t stride_w = 1,
    int64_t padding_d = 0,
    int64_t padding_h = 0,
    int64_t padding_w = 0,
    int64_t dilation_d = 1,
    int64_t dilation_h = 1,
    int64_t dilation_w = 1,
    int64_t groups = 1
) {

    const double base = base_t.item<double>();

    bool squeeze_batch = false;
    if (input.dim() == 4) {
        input.unsqueeze_(0);
        squeeze_batch = true;
    }

    TORCH_CHECK(input.dim()  == 5 &&
                weight.dim() == 5,
                "Expected 5-D input (N, C_in, D, H, W) and 5-D weight "
                "(C_out, C_in/groups, K_d, K_h, K_w)");

    const int64_t N = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Din = input.size(2);
    const int64_t Hin = input.size(3);
    const int64_t Win = input.size(4);

    const int64_t Cout = weight.size(0);
    const int64_t Kd = weight.size(2);
    const int64_t Kh = weight.size(3);
    const int64_t Kw = weight.size(4);

    TORCH_CHECK(Cin % groups == 0 && Cout % groups == 0,
                "C_in and C_out must be divisible by groups");

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    TORCH_CHECK(weight.size(1) == Cin_g, "weight second dim must equal C_in / groups");

    if (bias.has_value())
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == Cout, "bias must be 1-D with length C_out");

    const int64_t Dout = (Din + 2 * padding_d - dilation_d * (Kd - 1) - 1) / stride_d + 1;
    const int64_t Hout = (Hin + 2 * padding_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
    const int64_t Wout = (Win + 2 * padding_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;

    TORCH_CHECK(Dout > 0 && Hout > 0 && Wout > 0, "Output size is non-positive");
    torch::Tensor output = at::empty({N, Cout, Dout, Hout, Wout}, input.options());

    const int64_t* in = input .data_ptr<int64_t>();
    const int64_t* w = weight.data_ptr<int64_t>();
    int64_t* out = output.data_ptr<int64_t>();
    const int64_t* b = bias.has_value() ? bias->data_ptr<int64_t>() : nullptr;

    const int64_t in_stride_N = Cin * Din * Hin * Win;
    const int64_t in_stride_C = Din * Hin * Win;
    const int64_t in_stride_D = Hin * Win;
    const int64_t in_stride_H = Win;

    const int64_t w_stride_Cout = Cin_g * Kd * Kh * Kw;
    const int64_t w_stride_Cin  = Kd * Kh * Kw;
    const int64_t w_stride_Kd   = Kh * Kw;
    const int64_t w_stride_Kh   = Kw;

    const int64_t out_stride_N = Cout * Dout * Hout * Wout;
    const int64_t out_stride_C = Dout * Hout * Wout;
    const int64_t out_stride_D = Hout * Wout;

    const int64_t work_items = N * Cout;
    const int64_t grain = 16;

    at::parallel_for(
        /*begin*/ 0,
        /*end*/ work_items,
        /*grain*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            for (int64_t linear = begin; linear < end; ++linear) {
                const int64_t n = linear / Cout;
                const int64_t oc = linear % Cout;

                const int64_t g = oc / Cout_g;
                const int64_t oc_in_g = oc % Cout_g;

                const int64_t* w_g = w + (g * Cout_g + oc_in_g) * w_stride_Cout;
                const int64_t* in_g = in + n * in_stride_N + g * Cin_g * Din * Hin * Win;
                int64_t* out_g = out + n * out_stride_N + oc * out_stride_C;

                for (int64_t z_out = 0; z_out < Dout; ++z_out) {
                    const int64_t z_in0 = z_out * stride_d - padding_d;

                    const int64_t kd_min = std::max<int64_t>(0, (-z_in0 + dilation_d - 1) / dilation_d);
                    const int64_t kd_max = std::min<int64_t>(Kd, (Din - z_in0 + dilation_d - 1) / dilation_d);

                    for (int64_t y_out = 0; y_out < Hout; ++y_out) {
                        const int64_t y_in0 = y_out * stride_h - padding_h;

                        const int64_t kh_min = std::max<int64_t>(0, (-y_in0 + dilation_h - 1) / dilation_h);
                        const int64_t kh_max = std::min<int64_t>(Kh, (Hin - y_in0 + dilation_h - 1) / dilation_h);

                        for (int64_t x_out = 0; x_out < Wout; ++x_out) {
                            const int64_t x_in0 = x_out * stride_w - padding_w;

                            const int64_t kw_min = std::max<int64_t>(0, (-x_in0 + dilation_w - 1) / dilation_w);
                            const int64_t kw_max = std::min<int64_t>(Kw, (Win - x_in0 + dilation_w - 1) / dilation_w);

                            int64_t acc = b ? b[oc] : lns::zero_int;
                            for (int64_t ic_g = 0; ic_g < Cin_g; ++ic_g) {
                                const int64_t* in_c = in_g + ic_g * in_stride_C;
                                const int64_t* w_c  = w_g + ic_g * w_stride_Cin;

                                for (int64_t kd = kd_min; kd < kd_max; ++kd) {
                                    const int64_t in_depth_offset = (z_in0 + kd * dilation_d) * in_stride_D;
                                    const int64_t* in_d = in_c + in_depth_offset;
                                    const int64_t* w_d  = w_c + kd * w_stride_Kd;

                                    for (int64_t kh = kh_min; kh < kh_max; ++kh) {
                                        const int64_t in_row_offset = (y_in0 + kh * dilation_h) * in_stride_H;
                                        const int64_t* in_row = in_d + in_row_offset;
                                        const int64_t* w_row  = w_d + kh * w_stride_Kh;

                                        for (int64_t kw = kw_min; kw < kw_max; ++kw) {
                                            const int64_t in_val = in_row[x_in0 + kw * dilation_w];
                                            const int64_t w_val  = w_row[kw];

                                            acc = lns::add(acc, lns::mul(in_val, w_val), base);
                                        }
                                    }
                                }
                            }

                            out_g[z_out * out_stride_D + y_out * Wout + x_out] = acc;
                        }
                    }
                }
            }
        }
    );

    return squeeze_batch ? output.squeeze(0) : output;
}

std::vector<torch::Tensor> conv3d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& base_t,
    const bool bias_defined,
    int64_t stride_d = 1,
    int64_t stride_h = 1,
    int64_t stride_w = 1,
    int64_t padding_d = 0,
    int64_t padding_h = 0,
    int64_t padding_w = 0,
    int64_t dilation_d = 1,
    int64_t dilation_h = 1,
    int64_t dilation_w = 1,
    int64_t groups = 1
) {

    const double base = base_t.item<double>();

    bool squeeze_batch = false;
    if (input.dim() == 4) {
        input.unsqueeze_(0);
        grad_output.unsqueeze_(0);
        squeeze_batch = true;
    }

    TORCH_CHECK(input.dim() == 5 && weight.dim() == 5, "conv3d_backward expects 5-D input and weight");

    const int64_t N = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Din = input.size(2);
    const int64_t Hin = input.size(3);
    const int64_t Win = input.size(4);

    const int64_t Cout = weight.size(0);
    const int64_t Kd = weight.size(2);
    const int64_t Kh = weight.size(3);
    const int64_t Kw = weight.size(4);

    TORCH_CHECK(Cin % groups == 0 && Cout % groups == 0, "C_in and C_out must be divisible by groups");

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    const int64_t Dout = grad_output.size(2);
    const int64_t Hout = grad_output.size(3);
    const int64_t Wout = grad_output.size(4);

    torch::Tensor grad_input = at::empty_like(input);
    torch::Tensor grad_weight = at::empty_like(weight);
    torch::Tensor grad_bias = bias_defined ? at::empty({Cout}, input.options()) : torch::Tensor();

    grad_input.fill_(lns::zero_int);
    grad_weight.fill_(lns::zero_int);
    if (bias_defined) grad_bias.fill_(lns::zero_int);

    const int64_t* in_ptr = input.data_ptr<int64_t>();
    const int64_t* w_ptr = weight.data_ptr<int64_t>();
    const int64_t* go_ptr = grad_output .data_ptr<int64_t>();

    int64_t* gi_ptr = grad_input .data_ptr<int64_t>();

    const int64_t in_stride_N = Cin * Din * Hin * Win;
    const int64_t in_stride_C = Din * Hin * Win;
    const int64_t in_stride_D = Hin * Win;
    const int64_t in_stride_H = Win;

    const int64_t w_stride_Cout = Cin_g * Kd * Kh * Kw;
    const int64_t w_stride_Cin = Kd * Kh * Kw;
    const int64_t w_stride_Kd = Kh * Kw;
    const int64_t w_stride_Kh = Kw;

    const int64_t go_stride_N = Cout * Dout * Hout * Wout;
    const int64_t go_stride_C = Dout * Hout * Wout;
    const int64_t go_stride_D = Hout * Wout;
    const int64_t go_stride_H = Wout;

    const int64_t work_items = N * Cout;
    const int64_t grain = 16;

    at::parallel_for(
        /*begin*/ 0,
        /*end*/ work_items,
        /*grain*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            torch::Tensor gw_private = at::empty_like(weight);
            gw_private.fill_(lns::zero_int);

            torch::Tensor gb_private;
            if (bias_defined) {
                gb_private = at::empty({Cout}, input.options());
                gb_private.fill_(lns::zero_int);
            }

            int64_t* gw_p = gw_private.data_ptr<int64_t>();
            int64_t* gb_p = bias_defined ? gb_private.data_ptr<int64_t>() : nullptr;

            for (int64_t linear = begin; linear < end; ++linear) {

                const int64_t n = linear / Cout;
                const int64_t co = linear % Cout;

                const int64_t g = co / Cout_g;
                const int64_t co_in_group = co % Cout_g;
                const int64_t ci_group_base = g * Cin_g;

                const int64_t go_base = n * go_stride_N + co * go_stride_C;
                const int64_t in_base = n * in_stride_N;
                const int64_t w_base = (g * Cout_g + co_in_group) * w_stride_Cout;

                for (int64_t z_out = 0; z_out < Dout; ++z_out) {
                    const int64_t go_depth_base = go_base + z_out * go_stride_D;
                    const int64_t z_in_origin = z_out * stride_d - padding_d;

                    for (int64_t y_out = 0; y_out < Hout; ++y_out) {
                        const int64_t go_row_base = go_depth_base + y_out * go_stride_H;
                        const int64_t y_in_origin = y_out * stride_h - padding_h;

                        for (int64_t x_out = 0; x_out < Wout; ++x_out) {
                            const int64_t go_offset = go_row_base + x_out;
                            const int64_t go_val = go_ptr[go_offset];

                            if (bias_defined)
                                gb_p[co] = lns::add(gb_p[co], go_val, base);

                            for (int64_t kd = 0; kd < Kd; ++kd) {
                                const int64_t z_in = z_in_origin + kd * dilation_d;
                                if (z_in < 0 || z_in >= Din) continue;

                                const int64_t w_d_base = w_base + kd * w_stride_Kd;
                                const int64_t in_d_base = in_base + z_in * in_stride_D;

                                for (int64_t kh = 0; kh < Kh; ++kh) {
                                    const int64_t y_in = y_in_origin + kh * dilation_h;
                                    if (y_in < 0 || y_in >= Hin) continue;

                                    const int64_t w_kh_base = w_d_base  + kh * w_stride_Kh;
                                    const int64_t in_row_base = in_d_base + y_in * in_stride_H;

                                    for (int64_t kw = 0; kw < Kw; ++kw) {
                                        const int64_t x_in = x_out * stride_w - padding_w + kw * dilation_w;
                                        if (x_in < 0 || x_in >= Win) continue;

                                        const int64_t w_col_offset = w_kh_base + kw;
                                        const int64_t in_col_base = in_row_base + x_in;

                                        for (int64_t ci_rel = 0; ci_rel < Cin_g; ++ci_rel) {
                                            const int64_t ci = ci_group_base + ci_rel;

                                            const int64_t in_idx = in_col_base + ci * in_stride_C;
                                            const int64_t w_idx = w_col_offset + ci_rel * w_stride_Cin;

                                            const int64_t prod_in = lns::mul(go_val, w_ptr[w_idx]);
                                            gi_ptr[in_idx] = lns::add(gi_ptr[in_idx], prod_in, base);

                                            const int64_t prod_w = lns::mul(go_val, in_ptr[in_idx]);
                                            gw_p[w_idx] = lns::add(gw_p[w_idx], prod_w, base);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                grad_weight.copy_(add_forward(grad_weight, gw_private, base_t));
                if (bias_defined)
                    grad_bias.copy_(add_forward(grad_bias, gb_private, base_t));
            }
        });

    if (squeeze_batch)
        grad_input = grad_input.squeeze(0);

    if (bias_defined)
        return {grad_input, grad_weight, grad_bias};
    else
        return {grad_input, grad_weight};
}

void init_lns_convolution(py::module& m) {
    m.def(
        "conv1d_forward",
        &conv1d_forward,
        "LNS conv1d forward pass",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias").none(true),
        py::arg("base_t"),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1,
        py::arg("groups") = 1
    );
    m.def(
        "conv1d_backward",
        &conv1d_backward,
        "LNS conv1d backward pass",
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("base_t"),
        py::arg("bias_defined"),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1,
        py::arg("groups") = 1
    );
    m.def(
        "conv2d_forward",
        &conv2d_forward,
        "LNS conv2d forward pass",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias").none(true),
        py::arg("base_t"),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("padding_h") = 0,
        py::arg("padding_w") = 0,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::arg("groups") = 1
    );
    m.def(
        "conv2d_backward",
        &conv2d_backward,
        "LNS conv2d backward pass",
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("base_t"),
        py::arg("bias_defined"),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("padding_h") = 0,
        py::arg("padding_w") = 0,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::arg("groups") = 1
    );
    m.def(
        "conv3d_forward",
        &conv3d_forward,
        "LNS conv3d forward pass",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias").none(true),
        py::arg("base_t"),
        py::arg("stride_d") = 1,
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("padding_d") = 0,
        py::arg("padding_h") = 0,
        py::arg("padding_w") = 0,
        py::arg("dilation_d") = 1,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::arg("groups") = 1
    );
    m.def(
        "conv3d_backward",
        &conv3d_backward,
        "LNS conv3d backward pass",
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("base_t"),
        py::arg("bias_defined"),
        py::arg("stride_d") = 1,
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("padding_d") = 0,
        py::arg("padding_h") = 0,
        py::arg("padding_w") = 0,
        py::arg("dilation_d") = 1,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::arg("groups") = 1
    );
}