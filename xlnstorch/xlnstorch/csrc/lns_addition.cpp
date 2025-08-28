#include <torch/extension.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <cstdint>
#include <algorithm>
#include <vector>

#include "lns_constants.h"
#include "pointwise_ops.h"
#include "vectorized_ops.h"

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

    at::native::cpu_kernel_vec(
        iter,
        [base](int64_t a, int64_t b) -> int64_t {
            return lns::add(a, b, base);
        },
        [base](int64_vec_t a, int64_vec_t b) -> int64_vec_t {
            return lns::add_vec(a, b, base);
        });

    return out;

}


static inline std::vector<int64_t> canonicalize_dims(
    std::vector<int64_t> dims,
    int64_t ndim
) {

    std::vector<int64_t> out;
    out.reserve(dims.size());

    for (int64_t dim : dims) {
        dim = dim < 0 ? dim + ndim : dim;
        TORCH_CHECK(dim >= 0 && dim < ndim,
            "Dimension ", dim, " out of range for tensor of dim ", ndim);
        out.push_back(dim);
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;

}

static inline std::vector<int64_t> reduced_sizes(
    const torch::Tensor& in,
    const std::vector<int64_t>& dims,
    bool keepdim
) {

    std::vector<int64_t> sizes = in.sizes().vec();
    if (keepdim) {
        for (int64_t dim : dims) sizes[dim] = 1;
    }
    else {
        for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
            sizes.erase(sizes.begin() + *it);
        }
    }

    return sizes;

}

static torch::Tensor reduce_all(const torch::Tensor& in, const double base) {

    const int64_t* src = in.data_ptr<int64_t>();
    const int64_t N = in.numel();

    constexpr int kUnroll = 4; // 4-way unrolling between lns_add calls
    const int64_t grain = 1 << 13;  // ~8 k elements per thread

    int64_t global_acc = at::parallel_reduce(
        /*begin*/ int64_t{0},
        /*end*/ N,
        /*grain_size*/ grain,
        /*identity*/ lns::zero_int,
        /*body*/ [&](int64_t begin, int64_t end, int64_t /*identity*/) -> int64_t {
            const int64_t* ptr = src + begin;
            int64_t local = lns::zero_int;
            int64_t len = end - begin;

            int64_t i = 0;
            for (; i <= len - kUnroll; i += kUnroll) {
                int64_t t0 = ptr[i    ];
                int64_t t1 = ptr[i + 1];
                int64_t t2 = ptr[i + 2];
                int64_t t3 = ptr[i + 3];

                int64_t blk = lns::add(lns::add(t0, t1, base), lns::add(t2, t3, base), base);
                local = lns::add(local, blk, base);
            }

            // tail
            for (; i < len; ++i) {
                local = lns::add(local, ptr[i], base);
            }

            return local;
        },
        /*reduce*/ [base](int64_t a, int64_t b) -> int64_t {
            return lns::add(a, b, base);
        });

    return at::scalar_tensor(global_acc, in.options());
}

static torch::Tensor reduce_dims(
    const torch::Tensor& src,
    const double base,
    const std::vector<int64_t>& rdims,
    bool keepdim
) {

    // allocate output (with keepdim=true shape, squeeze later if needed)
    torch::Tensor out = at::empty(reduced_sizes(src, rdims, /*keepdim=*/true), src.options().dtype(torch::kInt64));
    at::TensorIterator iter = at::meta::make_reduction(src, out, rdims, /*keepdim=*/true, torch::kInt64);

    if (iter.numel() == 0)
        out.fill_(lns::zero_int);

    else {

        at::native::binary_kernel_reduce_vec(
            iter,
            /*scalar op*/ [base](int64_t a, int64_t b) -> int64_t {
                return lns::add(a, b, base);
            },
            /*vector op*/ [base](int64_vec_t a, int64_vec_t b) -> int64_vec_t {
                return lns::add_vec(a, b, base);
            },
            /*initializer*/ lns::zero_int);

    }

    return keepdim ? out : out.squeeze(rdims);
}

torch::Tensor sum_forward(
    const torch::Tensor& x,
    const torch::Tensor& base_t,
    const std::vector<int64_t>& dims = {},
    bool keepdim = false
) {

    const double base = base_t.item<double>();
    torch::Tensor src = x.is_contiguous() ? x : x.contiguous();

    if (dims.empty()) {
        return reduce_all(src, base);
    }

    std::vector<int64_t> rdims = canonicalize_dims(dims, src.dim());
    if (rdims.empty()) {
        return keepdim ? src.clone() : src;
    }

    return reduce_dims(src, base, rdims, keepdim);

}

void init_lns_addition(py::module& m) {
    m.def(
        "add_forward",
        &add_forward,
        "LNS addition forward pass",
        py::arg("x"),
        py::arg("y"),
        py::arg("base_t")
    );
    m.def(
        "sum_forward",
        &sum_forward,
        "LNS summation forward pass",
        py::arg("x"),
        py::arg("base_t"),
        py::arg("dims") = std::vector<int64_t>{},
        py::arg("keepdim") = false
    );
}