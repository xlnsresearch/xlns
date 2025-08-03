#ifndef VECTORIZED_OPS_H
#define VECTORIZED_OPS_H

#include <ATen/cpu/vec/vec.h>

using int64_vec_t = at::vec::Vectorized<int64_t>;
using double_vec_t = at::vec::Vectorized<double>;

namespace lns {

    int64_vec_t add_vec(int64_vec_t x, int64_vec_t y, double base);
    int64_vec_t mul_vec(int64_vec_t x, int64_vec_t y);

}

#endif // VECTORIZED_OPS_H