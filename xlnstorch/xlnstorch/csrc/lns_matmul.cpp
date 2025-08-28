#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "lns_constants.h"
#include "lns_addition.h"
#include "lns_multiplication.h"
#include "pointwise_ops.h"

/*

See some previous attempts at matmul for reference that were
not used because they were slow for large matrices:

// fast inner-kernel : one (MxK)*(KxN) -> (MxN)
// both operands must be contiguous and row major format
static void matmul_single_batch(
    const int64_t* A,
    const int64_t* B,
    int64_t* C,
    int64_t M,
    int64_t K,
    int64_t N,
    double base
) {
    const int64_t strideB = N;

    at::parallel_for(
        0,
        M,
        0,
        [&](int64_t i0, int64_t i1) {

            for (int64_t i = i0; i < i1; ++i) {
                const int64_t* Arow = A + i * K;
                int64_t* Crow = C + i * N;

                for (int64_t j = 0; j < N; ++j) {
                    int64_t acc = lns::zero_int;

                    int64_t k = 0;
                    for (; k + 3 < K; k += 4) {
                        acc = lns::add(acc, lns::mul(Arow[k    ], B[(k    )*strideB + j]), base);
                        acc = lns::add(acc, lns::mul(Arow[k + 1], B[(k + 1)*strideB + j]), base);
                        acc = lns::add(acc, lns::mul(Arow[k + 2], B[(k + 2)*strideB + j]), base);
                        acc = lns::add(acc, lns::mul(Arow[k + 3], B[(k + 3)*strideB + j]), base);
                    }

                    for (; k < K; ++k)
                        acc = lns::add(acc, lns::mul(Arow[k], B[k*strideB + j]), base);

                    Crow[j] = acc;
                }
            }
        }
    );
}

torch::Tensor matmul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& base_t
) {

    const double base = base_t.item<double>();

    // 1D x 1D -> scalar (dot product)
    if (A.dim() == 1 && B.dim() == 1) {
        const int64_t K = A.size(0);

        const int64_t* a = A.data_ptr<int64_t>();
        const int64_t* b = B.data_ptr<int64_t>();
        int64_t acc = lns::zero_int;

        for (int64_t k = 0; k < K; ++k)
            acc = lns::add(acc, lns::mul(a[k], b[k]), base);

        return torch::scalar_tensor(acc, A.options());
    }

    // bring operands to shape ... x M x K and ... x K x N
    bool A_was_1d = A.dim() == 1;
    bool B_was_1d = B.dim() == 1;

    torch::Tensor A_exp = A_was_1d ? A.unsqueeze(0) : A;
    torch::Tensor B_exp = B_was_1d ? B.unsqueeze(-1) : B;

    TORCH_CHECK(A_exp.size(-1) == B_exp.size(-2), "matmul_forward: inner dimensions do not match");

    // batch shapes
    const int a_bdims = A_exp.dim() - 2;
    const int b_bdims = B_exp.dim() - 2;

    std::vector<int64_t> a_batch(a_bdims), b_batch(b_bdims);
    for (int i = 0;  i < a_bdims; ++i) a_batch[i] = A_exp.size(i);
    for (int i = 0;  i < b_bdims; ++i) b_batch[i] = B_exp.size(i);

    std::vector<int64_t> batch_shape = at::infer_size(at::IntArrayRef(a_batch), at::IntArrayRef(b_batch));

    std::vector<int64_t> A_expand = batch_shape;
    A_expand.push_back(A.size(-2));
    A_expand.push_back(A.size(-1));

    std::vector<int64_t> B_expand = batch_shape;
    B_expand.push_back(B.size(-2));
    B_expand.push_back(B.size(-1));

    A_exp = A_exp.expand(A_expand).contiguous();
    B_exp = B_exp.expand(B_expand).contiguous();

    const int64_t M = A_exp.size(-2);
    const int64_t K = A_exp.size(-1);
    const int64_t N = B_exp.size(-1);

    int64_t batch_elems = 1;
    for (auto s : batch_shape) batch_elems *= s;

    std::vector<int64_t> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);
    torch::Tensor out = at::empty(out_shape, A.options());

    const int64_t* Ap = A_exp.data_ptr<int64_t>();
    const int64_t* Bp = B_exp.data_ptr<int64_t>();
    int64_t* Cp = out.data_ptr<int64_t>();

    const int64_t A_batch_stride = M * K;
    const int64_t B_batch_stride = K * N;
    const int64_t C_batch_stride = M * N;

    at::parallel_for(
        0,
        batch_elems,
        0,
        [&](int64_t b0, int64_t b1) {

            for (int64_t b = b0; b < b1; ++b) {
                const int64_t* Ab = Ap + b * A_batch_stride;
                const int64_t* Bb = Bp + b * B_batch_stride;
                int64_t* Cb = Cp + b * C_batch_stride;

                matmul_single_batch(Ab, Bb, Cb, M, K, N, base);
            }
        }
    );

    // squeeze dimensions if necessary
    if (A_was_1d) out = out.squeeze(-2);
    if (B_was_1d) out = out.squeeze(-1);

    return out;
}




constexpr int64_t TM = 64;
constexpr int64_t TN = 64;
constexpr int64_t TK = 32;

inline void accumulate_K_slice(const int64_t* A, const int64_t* B,
                               int64_t        acc[TM][TN],
                               int64_t strideA,   // K
                               int64_t strideB,   // N
                               int64_t M_rem, int64_t N_rem, int64_t K_eff,
                               double base)
{
    for (int64_t kk = 0; kk < K_eff; ++kk)
    {
        const int64_t* Ap = A + kk;
        const int64_t* Bp = B + kk * strideB;

        for (int64_t i = 0; i < M_rem; ++i, Ap += strideA)
        {
            int64_t Aval = *Ap;
            for (int64_t j = 0; j < N_rem; ++j)
            {
                int64_t Bval = Bp[j];
                acc[i][j] = lns::add(acc[i][j],
                                     lns::mul(Aval, Bval),
                                     base);
            }
        }
    }
}

torch::Tensor matmul_forward(const torch::Tensor& A,
                             const torch::Tensor& B,
                             const torch::Tensor& base_t)
{
    const double base = base_t.item<double>();

    if (A.dim() == 1 && B.dim() == 1) {
        TORCH_CHECK(A.size(0) == B.size(0), "dot product: size mismatch");

        const int64_t K = A.size(0);
        const int64_t* a = A.data_ptr<int64_t>();
        const int64_t* b = B.data_ptr<int64_t>();

        int64_t acc = lns::zero_int;
        for (int64_t k = 0; k < K; ++k)
            acc = lns::add(acc, lns::mul(a[k], b[k]), base);

        return torch::scalar_tensor(acc, A.options());
    }

    bool A_was_1d = A.dim() == 1;
    bool B_was_1d = B.dim() == 1;

    torch::Tensor A_tmp = A_was_1d ? A.unsqueeze(0)  : A;
    torch::Tensor B_tmp = B_was_1d ? B.unsqueeze(-1) : B;

    TORCH_CHECK(A_tmp.size(-1) == B_tmp.size(-2),
                "matmul_forward: inner dimensions mismatch");

    const int a_bdims = A_tmp.dim() - 2;
    const int b_bdims = B_tmp.dim() - 2;

    std::vector<int64_t> a_batch(a_bdims), b_batch(b_bdims);
    for (int i = 0; i < a_bdims; ++i) a_batch[i] = A_tmp.size(i);
    for (int i = 0; i < b_bdims; ++i) b_batch[i] = B_tmp.size(i);

    std::vector<int64_t> batch_shape =
        at::infer_size(at::IntArrayRef(a_batch),
                       at::IntArrayRef(b_batch));

    auto expand_to = [&](const torch::Tensor& t) {
        std::vector<int64_t> s = batch_shape;
        s.push_back(t.size(-2));
        s.push_back(t.size(-1));
        return t.expand(s).contiguous();
    };

    torch::Tensor A_exp = expand_to(A_tmp);
    torch::Tensor B_exp = expand_to(B_tmp);

    int64_t batch_elems = 1;
    for (int64_t v : batch_shape) batch_elems *= v;

    const int64_t M = A_exp.size(-2);
    const int64_t K = A_exp.size(-1);
    const int64_t N = B_exp.size(-1);

    std::vector<int64_t> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);
    torch::Tensor out = at::empty(out_shape, A.options());

    const int64_t* Ap = A_exp.data_ptr<int64_t>();
    const int64_t* Bp = B_exp.data_ptr<int64_t>();
    int64_t*       Cp = out   .data_ptr<int64_t>();

    const int64_t A_bs = M*K;
    const int64_t B_bs = K*N;
    const int64_t C_bs = M*N;

    const int64_t tilesM = (M + TM - 1) / TM;
    const int64_t tilesN = (N + TN - 1) / TN;
    const int64_t total_tiles = batch_elems * tilesM * tilesN;

    const int64_t grain = std::max<int64_t>(1,
                        (total_tiles + 4*at::get_num_threads() - 1) /
                        (4*at::get_num_threads()));

    at::parallel_for(0, total_tiles, grain,
        [&](int64_t t0, int64_t t1)
    {
        int64_t acc_local[TM][TN];

        for (int64_t tid = t0; tid < t1; ++tid)
        {
            int64_t tmp = tid;
            const int64_t jn = tmp % tilesN;   tmp /= tilesN;
            const int64_t im = tmp % tilesM;   tmp /= tilesM;
            const int64_t b  = tmp;

            const int64_t i0 = im * TM;
            const int64_t j0 = jn * TN;
            const int64_t M_rem = std::min<int64_t>(TM, M - i0);
            const int64_t N_rem = std::min<int64_t>(TN, N - j0);

            const int64_t* A_base = Ap + b*A_bs + i0*K;
            const int64_t* B_base = Bp + b*B_bs + j0;
            int64_t*       C_base = Cp + b*C_bs + i0*N + j0;

            for (int64_t i = 0; i < M_rem; ++i)
                for (int64_t j = 0; j < N_rem; ++j)
                    acc_local[i][j] = lns::zero_int;

            for (int64_t k0 = 0; k0 < K; k0 += TK)
            {
                const int64_t K_eff = std::min<int64_t>(TK, K - k0);

                accumulate_K_slice(A_base + k0,
                                   B_base + k0*N,
                                   acc_local,
                                   K,        // strideA
                                   N,        // strideB
                                   M_rem,
                                   N_rem,
                                   K_eff,
                                   base);
            }

            for (int64_t i = 0; i < M_rem; ++i)
                for (int64_t j = 0; j < N_rem; ++j)
                    C_base[i*N + j] = acc_local[i][j];
        }
    });

    if (A_was_1d) out = out.squeeze(-2);
    if (B_was_1d) out = out.squeeze(-1);
    return out;
}
*/


torch::Tensor matmul_forward(const torch::Tensor& A_in,
                             const torch::Tensor& B_in,
                             const torch::Tensor& base_t)
{
    const double base = base_t.item<double>();

    if (A_in.dim() == 1 && B_in.dim() == 1) {

        TORCH_CHECK(A_in.size(0) == B_in.size(0), "dot product: size mismatch");
        // return sum_forward(mul_forward(A_in, B_in), base_t);

        const int64_t K = A_in.size(0);
        const int64_t* a = A_in.data_ptr<int64_t>();
        const int64_t* b = B_in.data_ptr<int64_t>();

        int64_t acc = lns::zero_int;
        for (int64_t k = 0; k < K; ++k)
            acc = lns::add(acc, lns::mul(a[k], b[k]), base);

        return torch::scalar_tensor(acc, A_in.options());
    }

    bool A_was_1d = A_in.dim() == 1;
    bool B_was_1d = B_in.dim() == 1;

    torch::Tensor A = A_was_1d ? A_in.unsqueeze(0) : A_in;
    torch::Tensor B = B_was_1d ? B_in.unsqueeze(-1) : B_in;

    TORCH_CHECK(A.size(-1) == B.size(-2), "matmul_forward: inner dimensions mismatch");

    auto A_batch = A.sizes().slice(0, A.dim() - 2);
    auto B_batch = B.sizes().slice(0, B.dim() - 2);
    std::vector<int64_t> out_batch = at::infer_size(A_batch, B_batch);   // throws if not broadcastable

    auto expand_to = [&](const torch::Tensor& t) {
        std::vector<int64_t> s(out_batch.begin(), out_batch.end());
        s.push_back(t.size(-2));
        s.push_back(t.size(-1));
        return t.expand(s);
    };

    A = expand_to(A);   // no .contiguous(): element-wise kernels tolerate 0-strides
    B = expand_to(B);

    const int64_t M = A.size(-2);
    const int64_t K = A.size(-1);
    const int64_t N = B.size(-1);

    std::vector<int64_t> out_shape(out_batch.begin(), out_batch.end());
    out_shape.push_back(M);
    out_shape.push_back(N);

    torch::Tensor result = torch::full(out_shape,
                                       lns::zero_int,
                                       A.options());

    for (int64_t k = 0; k < K; ++k) {
        auto A_slice = A.select(-1, k).unsqueeze(-1);  // ... × M × 1
        auto B_slice = B.select(-2, k).unsqueeze(-2);  // ... × 1 × N

        torch::Tensor term = mul_forward(A_slice, B_slice);
        result = add_forward(result, term, base_t);
    }

    if (A_was_1d) result = result.squeeze(-2);
    if (B_was_1d) result = result.squeeze(-1);
    return result;
}

static torch::Tensor reduce_like(
    const torch::Tensor& grad_expanded,
    const torch::Tensor& original_view,
    const torch::Tensor& base_t
) {
    const int64_t gdim = grad_expanded.dim();
    const int64_t odim = original_view.dim();
    const int64_t offset = gdim - odim;

    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < grad_expanded.dim(); ++d) {
        int64_t o_d = d - offset;
        int64_t o_size = (o_d >= 0) ? original_view.size(o_d) : 1;

        if (o_size == 1 && grad_expanded.size(d) > 1)
            reduce_dims.push_back(d);

    }

    if (reduce_dims.empty())
        return grad_expanded;

    return sum_forward(grad_expanded, base_t, reduce_dims, /*keepdim=*/true);
}

std::vector<torch::Tensor> matmul_backward(
    const torch::Tensor& grad_out_,
    const torch::Tensor& A_,
    const torch::Tensor& B_,
    const torch::Tensor& base_t
) {

    if (A_.dim() == 1 && B_.dim() == 1) {
        TORCH_CHECK(grad_out_.dim() == 0, "grad_out for dot product must be a scalar");

        const int64_t K = A_.size(0);
        const int64_t go = grad_out_.item<int64_t>();

        torch::Tensor dA = at::empty_like(A_);
        torch::Tensor dB = at::empty_like(B_);

        const int64_t* ap = A_.data_ptr<int64_t>();
        const int64_t* bp = B_.data_ptr<int64_t>();
        int64_t* dap = dA.data_ptr<int64_t>();
        int64_t* dbp = dB.data_ptr<int64_t>();

        at::parallel_for(0, K, 64, [&](int64_t k0, int64_t k1) {
            for (int64_t k = k0; k < k1; ++k) {
                dap[k] = lns::mul(go, bp[k]);
                dbp[k] = lns::mul(go, ap[k]);
            }
        });
        return {dA, dB};
    }

    bool A_was_1d = A_.dim() == 1;
    bool B_was_1d = B_.dim() == 1;

    torch::Tensor A_ref = A_was_1d ? A_.unsqueeze(0)  : A_;
    torch::Tensor B_ref = B_was_1d ? B_.unsqueeze(-1) : B_;

    torch::Tensor grad_out = grad_out_;
    if (A_was_1d) grad_out = grad_out.unsqueeze(-2);
    if (B_was_1d) grad_out = grad_out.unsqueeze(-1);

    auto A_batch = A_ref.sizes().slice(0, A_ref.dim() - 2);
    auto B_batch = B_ref.sizes().slice(0, B_ref.dim() - 2);
    std::vector<int64_t> batch_shape = at::infer_size(A_batch, B_batch);

    auto expand_to = [&](const torch::Tensor& t) {
        std::vector<int64_t> s(batch_shape.begin(), batch_shape.end());
        s.push_back(t.size(-2));
        s.push_back(t.size(-1));
        return t.expand(s);
    };

    torch::Tensor A_exp = expand_to(A_ref);
    torch::Tensor B_exp = expand_to(B_ref);
    torch::Tensor gout_exp = expand_to(grad_out);

    torch::Tensor dA_exp = matmul_forward(gout_exp, B_exp.transpose(-2, -1).contiguous(), base_t);
    torch::Tensor dB_exp = matmul_forward(A_exp.transpose(-2, -1).contiguous(), gout_exp, base_t);

    torch::Tensor dA = reduce_like(dA_exp, A_ref, base_t);
    torch::Tensor dB = reduce_like(dB_exp, B_ref, base_t);

    if (A_was_1d) dA = dA.squeeze(0);
    if (B_was_1d) dB = dB.squeeze(-1);

    return {dA, dB};

}

void init_lns_matmul(py::module& m) {
    m.def(
        "matmul_forward",
        &matmul_forward,
        "LNS matmul forward pass",
        py::arg("A"),
        py::arg("B"),
        py::arg("base_t")
    );
    m.def(
        "matmul_backward",
        &matmul_backward,
        "LNS matmul backward pass",
        py::arg("grad_output"),
        py::arg("A"),
        py::arg("B"),
        py::arg("base_t")
    );
}