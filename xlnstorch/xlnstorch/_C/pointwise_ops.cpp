#include <torch/extension.h>
#include <cmath>
#include <string>
#include <map>

#include "lns_constants.h"

namespace sbdb {

    inline long long ideal(long long z, long long s, double base) {

        double power_term = std::pow(base, z);
        double magnitude = std::abs(1.0 - 2.0 * s + power_term);
        double log_term = std::log(magnitude) / std::log(base);

        return std::llround(log_term) << 1;
    }

}

namespace lns {

    using sbdb_fn_ptr = long long(*)(long long, long long, double);
    const std::map<std::string, sbdb_fn_ptr> sbdb_funcs {
        {"ideal", &sbdb::ideal} 
    };

    sbdb_fn_ptr default_sbdb_func = sbdb::ideal;
    void set_default_sbdb_func(std::string sbdb_key) {
        auto it = sbdb_funcs.find(sbdb_key);

        if (it == sbdb_funcs.end())
            default_sbdb_func = sbdb::ideal;

        else
            default_sbdb_func = it->second;
    }

    long long add(long long x, long long y, double base) {

        if ((x | 1LL) == lns::zero_int) return y;
        else if ((y | 1LL) == lns::zero_int) return x;
        else if ((x ^ 1LL) == y) return lns::zero_int;

        long long max_operand = std::max(x, y);
        const long long abs_diff = std::abs((x >> 1) - (y >> 1));
        const long long sign_diff = (x ^ y) & 1LL;

        return max_operand + default_sbdb_func(-abs_diff, sign_diff, base);
    }

    long long neg(long long x) {
        return x ^ 1LL;
    }

    long long sub(long long x, long long y, double base) {
        long long neg_y = lns::neg(y);
        return lns::add(x, neg_y, base);
    }

    long long mul(long long x, long long y) {

        if ((x | 1LL) == lns::zero_int || (y | 1LL) == lns::zero_int)
            return lns::zero_int;

        return (x + y - (y & 1)) ^ (y & 1);
    }

    long long div(long long x, long long y) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        if ((y | 1LL) == lns::zero_int)
            throw std::runtime_error("Division by zero in LNS division operation");

        return (x - y + (y & 1)) ^ (y & 1);
    }

    long long reciprocal(long long x) {
        return lns::div(lns::one_int, x);
    }

    long long square(long long x) {
        return lns::mul(x, x);
    }

    long long sqrt(long long x) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        return ((x & (-2)) / 2) & (-2);
    }

    long long pow(long long x, double n) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        if ((x & 1LL) && n < 0.0)
            throw std::runtime_error("Negative exponent in LNS power operation");

        return (static_cast<long long>((x & (-2)) * n)) & (-2);
    }

    long long pow(long long x, long long n) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        long long abs_result = ((x & (-2)) * n) & (-2);
        return (n & 1LL) ? abs_result | (x & 1) : abs_result;
    }

}

void init_pointwise_ops(py::module& m) {
    m.def("set_default_sbdb_func", &lns::set_default_sbdb_func, "Set the default SBDB function for C++ LNS operations");
}