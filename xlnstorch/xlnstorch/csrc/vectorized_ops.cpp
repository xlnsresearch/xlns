#include <ATen/cpu/vec/vec.h>

#include "vectorized_ops.h"
#include "lns_constants.h"
#include "sbdb.h"

namespace lns {

        int64_vec_t add_vec(int64_vec_t x, int64_vec_t y, double base) {

        const int64_vec_t mask_x_is_zero = ((x | int64_vec_t(1)) == int64_vec_t(lns::zero_int));
        const int64_vec_t mask_y_is_zero = ((y | int64_vec_t(1)) ==  int64_vec_t(lns::zero_int));
        const int64_vec_t mask_annihilate = ((x ^ int64_vec_t(1)) == y);

        const int64_vec_t max_operand = at::vec::maximum(x, y);
        const int64_vec_t abs_diff = ((x >> int64_vec_t(1)) - (y >> int64_vec_t(1))).abs();
        const int64_vec_t sign_diff = (x ^ y) & int64_vec_t(1);

        int64_vec_t result = max_operand + (&sbdb::default_entry)->vector(abs_diff.neg(), sign_diff, base);
        return int64_vec_t::blendv(
            int64_vec_t::blendv(
                int64_vec_t::blendv(
                    result,
                    int64_vec_t(lns::zero_int),
                    mask_annihilate
                ),
                x,
                mask_y_is_zero
            ),
            y,
            mask_x_is_zero
        );
        // result = select(mask_x_is_zero, y, result);

        // const int64_vec_t mask_use_x = mask_y_is_zero & (~mask_x_is_zero);
        // result = select(mask_use_x, x, result);

        // const int64_vec_t mask_zero_out = mask_annihilate & (~mask_x_is_zero) & (~mask_y_is_zero);
        // result = select(mask_zero_out, int64_vec_t(lns::zero_int), result);

        // return result;
    }

    int64_vec_t mul_vec(int64_vec_t x, int64_vec_t y) {

        // commented out overflow check
        /*
        const int64_vec_t x_or_y_is_zero_or_underflow = (
            (x | int64_vec_t(1)) == int64_vec_t(lns::zero_int)
        ) | (
            (y | int64_vec_t(1)) == int64_vec_t(lns::zero_int)
        ) | (
            (x < int64_vec_t(0)) & (y < (int64_vec_t(std::numeric_limits<int64_t>::min()) - x))
        );
        const int64_vec_t result = (x + y - (y & int64_vec_t(1))) ^ (y & int64_vec_t(1));
        return int64_vec_t::blendv(
            result,
            int64_vec_t(lns::zero_int),
            x_or_y_is_zero_or_underflow
        );

        const int64_vec_t underflow_mask = (result < int64_vec_t(lns::zero_int));
        */

        const int64_vec_t x_or_y_is_zero = (
            (x | int64_vec_t(1)) == int64_vec_t(lns::zero_int)
        ) | (
            (y | int64_vec_t(1)) == int64_vec_t(lns::zero_int));

        const int64_vec_t result = (x + y - (y & int64_vec_t(1))) ^ (y & int64_vec_t(1));
        return int64_vec_t::blendv(
            result,
            int64_vec_t(lns::zero_int),
            x_or_y_is_zero
        );
    }

}