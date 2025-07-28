#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include "lns_constants.h"

inline long long sbdb_ideal(long long z, long long s, double base) {
    double power_term = std::pow(base, z);
    double magnitude = std::abs(1.0 - 2.0 * s + power_term);
    double log_term = std::log(magnitude) / std::log(base);
    return std::llround(log_term) << 1;
}

using sbdb_fn_ptr = long long(*)(long long, long long, double);
const std::map<std::string, sbdb_fn_ptr> sbdb_funcs {
    {"ideal", &sbdb_ideal} 
};

torch::Tensor add_forward(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& base,
    const std::string& sbdb_key
) {

    const double double_b = base.item<double>();
    auto it = sbdb_funcs.find(sbdb_key);
    TORCH_CHECK(it != sbdb_funcs.end(), "Unsupported sbdb_func: ", sbdb_key);
    sbdb_fn_ptr sbdb_func = it->second; 

    torch::ScalarType common_dtype = (x.scalar_type() == torch::kFloat64 || y.scalar_type() == torch::kFloat64)
        ? torch::kFloat64
        : torch::kLong;
    auto result_sizes = at::infer_size(x.sizes(), y.sizes());
    auto out = torch::empty(result_sizes, x.options().dtype(common_dtype));

    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .promote_inputs_to_common_dtype(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND(torch::kLong, common_dtype, "add_forward", ([&] {

        using scalar_t = scalar_t;

        auto kernel = [double_b, sbdb_func] (scalar_t a, scalar_t b) -> scalar_t {

            long long a_packed, b_packed;
            if constexpr (std::is_same_v<scalar_t,double>) {
                // float64 tensors: bits are inside the double
                a_packed = static_cast<long long>(a);
                b_packed = static_cast<long long>(b);
            }
            else {
                // int64 tensors: value already is the packed bits
                a_packed = a;
                b_packed = b;
            }

            if ((a_packed | 1LL) == lns::zero_int) {
                return b;
            }
            else if ((b_packed | 1LL) == lns::zero_int) {
                return a;
            }
            else if ((a_packed ^ 1LL) == b_packed) {
                if (std::is_same_v<scalar_t, double>) {
                    return lns::zero;
                }
                else {
                    return lns::zero_int;
                }
            }

            long long max_operand = std::max(a_packed, b_packed);

            long long abs_diff = std::abs((a_packed >> 1) - (b_packed >> 1));
            long long sign_diff = (a_packed ^ b_packed) & 1LL;

            long long result = max_operand + sbdb_func(-abs_diff, sign_diff, double_b);
            return static_cast<scalar_t>(result);

        };

        at::native::cpu_kernel(iter, kernel);

    }));

    return out;

}

void init_lns_addition(py::module& m) {
    m.def("add_forward", &add_forward, "LNS addition forward pass");
}