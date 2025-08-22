#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Optional.h>
#include <cmath>

#include "lns_constants.h"
#include "pointwise_ops.h"

torch::Tensor avg_pool1d_forward(
    const torch::Tensor& input,
    int64_t kernel_size,
    const torch::Tensor& base_t,
    c10::optional<int64_t> stride_opt  = c10::nullopt,
    int64_t padding = 0,
    bool ceil_mode = false,
    bool count_include_pad = true
) {

    const double base = base_t.item<double>();
    const double inv_log_base = 1.0 / std::log(base);

    torch::Tensor x = input;
    bool squeeze_batch = false;
    if (x.dim() == 2) {
        x = x.unsqueeze(0);
        squeeze_batch = true;
    }

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t Lin = x.size(2);

    const int64_t stride = stride_opt.has_value() ? *stride_opt : kernel_size;
    torch::Tensor x_padded = torch::constant_pad_nd(x, {padding, padding}, lns::zero_int).contiguous();
    const int64_t Lin_pad = Lin + 2 * padding;

    int64_t Lout;
    if (ceil_mode)
        Lout = static_cast<int64_t>(
                   std::ceil(double(Lin + 2 * padding - kernel_size) / stride)) + 1;

    else
        Lout = (Lin + 2 * padding - kernel_size) / stride + 1;

    torch::Tensor output = at::empty({N, C, Lout}, x.options());
    output.fill_(lns::zero_int);

    const int64_t kernel_size_lns = lns::float_to_lns(kernel_size, inv_log_base);
    const int64_t* in = x_padded.data_ptr<int64_t>();
    int64_t* out = output.data_ptr<int64_t>();

    const int64_t in_stride_N = C * Lin_pad;
    const int64_t in_stride_C = Lin_pad;
    const int64_t out_stride_N = C * Lout;
    const int64_t out_stride_C = Lout;

    const int64_t work_items = N * C * Lout;
    const int64_t grain = 1 << 10; // ~1k elements per grain

    at::parallel_for(
        0,
        work_items,
        grain,
        [&](int64_t begin, int64_t end) {
            for (int64_t linear = begin; linear < end; ++linear) {
                const int64_t l_out = linear % Lout;
                const int64_t tmp = (linear / Lout);
                const int64_t c = tmp % C;
                const int64_t n = tmp / C;

                const int64_t* in_nc = in + n * in_stride_N + c * in_stride_C;
                int64_t* out_el = out + n * out_stride_N + c * out_stride_C + l_out;

                const int64_t start = l_out * stride;
                const int64_t end_pos = start + kernel_size;

                if (end_pos > Lin_pad) {
                    *out_el = lns::zero_int;
                    continue;
                }

                int64_t sm = lns::zero_int;
                for (int64_t k = start; k < end_pos; ++k)
                    sm = lns::add(sm, in_nc[k], base);

                int64_t divisor_lns;
                if (count_include_pad)
                    divisor_lns = kernel_size_lns;
                else {
                    const int64_t left_pad = std::max<int64_t>(0, padding - start);
                    const int64_t right_pad = std::max<int64_t>(0, end_pos - (Lin + padding));

                    const int64_t valid_cnt = kernel_size - (left_pad + right_pad);
                    divisor_lns = lns::float_to_lns(std::max<int64_t>(valid_cnt, 1), inv_log_base);
                }

                *out_el = lns::div(sm, divisor_lns);
            }
    });

    return squeeze_batch ? output.squeeze(0) : output;
}

void init_lns_pool(py::module& m) {
    m.def("avg_pool1d_forward", &avg_pool1d_forward, "LNS avg_pool1d forward");
}