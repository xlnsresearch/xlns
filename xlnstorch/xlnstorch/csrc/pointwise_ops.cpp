#include <torch/extension.h>
#include <cmath>
#include <string>
#include <map>
#include <cstdint>

#include "lns_constants.h"
#include "pointwise_ops.h"
#include "sbdb.h"

namespace lns {

    int64_t float_to_lns(double value, double inv_log_base) {

        if (value == 0.0)
            return lns::zero_int;

        int64_t exponent = llround(std::log(std::abs(value)) * inv_log_base);
        int64_t sign_bit = (value < 0.0) ? 1LL : 0LL;

        return (exponent << 1) | sign_bit;
    }

    int64_t add(int64_t x, int64_t y, double base) {

        if ((x | 1LL) == lns::zero_int) return y;
        else if ((y | 1LL) == lns::zero_int) return x;
        else if ((x ^ 1LL) == y) return lns::zero_int;

        const int64_t max_operand = std::max(x, y);
        const int64_t abs_diff = std::abs((x >> 1) - (y >> 1));
        const int64_t sign_diff = (x ^ y) & 1LL;

        return max_operand + (&sbdb::default_entry)->scalar(-abs_diff, sign_diff, base);
    }

    int64_t neg(int64_t x) {
        return x ^ 1LL;
    }

    int64_t sub(int64_t x, int64_t y, double base) {
        int64_t neg_y = lns::neg(y);
        return lns::add(x, neg_y, base);
    }

    int64_t mul(int64_t x, int64_t y) {

        // commented out overflow check
        /*
        if ((x | 1LL) == lns::zero_int || (y | 1LL) == lns::zero_int || 
            ((x < 0) && (y < (std::numeric_limits<int64_t>::min() - x))))
            return lns::zero_int;

        return (x + y - (y & 1LL)) ^ (y & 1LL);

        // underflow check
        if (result < lns::zero_int) {
            return lns::zero_int;
        }
        */

        if ((x | 1LL) == lns::zero_int || (y | 1LL) == lns::zero_int)
            return lns::zero_int;

        return (x + y - (y & 1)) ^ (y & 1);
    }

    int64_t div(int64_t x, int64_t y) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        if ((y | 1LL) == lns::zero_int)
            throw std::runtime_error("Division by zero in LNS division operation");

        return (x - y + (y & 1)) ^ (y & 1);
    }

    int64_t reciprocal(int64_t x) {
        return lns::div(lns::one_int, x);
    }

    int64_t square(int64_t x) {
        return lns::mul(x, x);
    }

    int64_t sqrt(int64_t x) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        return ((x & (-2)) / 2) & (-2);
    }

    int64_t pow(int64_t x, double n) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        if ((x & 1LL) && n < 0.0)
            throw std::runtime_error("Negative exponent in LNS power operation");

        return (static_cast<int64_t>((x & (-2)) * n)) & (-2);
    }

    int64_t pow(int64_t x, int64_t n) {

        if ((x | 1LL) == lns::zero_int)
            return lns::zero_int;

        int64_t abs_result = ((x & (-2)) * n) & (-2);
        return (n & 1LL) ? abs_result | (x & 1) : abs_result;
    }

}