#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

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
        [base](long long a, long long b) -> long long {
            return lns::add(a, b, base);
        });

    return out.to(torch::kFloat64);

}

// torch::Tensor add_forward(
//     const torch::Tensor& x,
//     const torch::Tensor& y,
//     const torch::Tensor& base_t
// ) {

//     const double base = base_t.item<double>();
//     torch::ScalarType common_dtype = (x.scalar_type() == torch::kFloat64 || y.scalar_type() == torch::kFloat64)
//         ? torch::kFloat64
//         : torch::kLong;
//     auto result_sizes = at::infer_size(x.sizes(), y.sizes());
//     auto out = torch::empty(result_sizes, x.options().dtype(common_dtype));

//     at::TensorIterator iter = at::TensorIteratorConfig()
//         .add_output(out)
//         .add_input(x)
//         .add_input(y)
//         .promote_inputs_to_common_dtype(true)
//         .build();

//     AT_DISPATCH_FLOATING_TYPES_AND(torch::kLong, common_dtype, "add_forward", ([&] {

//         using scalar_t = scalar_t;

//         auto kernel = [base] (scalar_t a, scalar_t b) -> scalar_t {

//             long long a_packed, b_packed;
//             if constexpr (std::is_same_v<scalar_t,double>) {
//                 // float64 tensors: bits are inside the double
//                 a_packed = static_cast<long long>(a);
//                 b_packed = static_cast<long long>(b);
//             }
//             else {
//                 // int64 tensors: value already is the packed bits
//                 a_packed = a;
//                 b_packed = b;
//             }

//             long long result = lns::add(a_packed, b_packed, base);
//             return static_cast<scalar_t>(result);

//         };

//         at::native::cpu_kernel(iter, kernel);

//     }));

//     return out;

// }

void init_lns_addition(py::module& m) {
    m.def("add_forward", &add_forward, "LNS addition forward pass");
}