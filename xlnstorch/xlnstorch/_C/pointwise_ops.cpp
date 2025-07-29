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
    sbdb_fn_ptr default_sbdb_func = sbdb::ideal;
    const std::map<std::string, sbdb_fn_ptr> sbdb_funcs {
        {"ideal", &sbdb::ideal} 
    };

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

}