#ifndef POINTWISE_OPS_H
#define POINTWISE_OPS_H

namespace lns {

    int64_t float_to_lns(double value, double inv_log_base);

    int64_t add(int64_t x, int64_t y, double base);
    int64_t neg(int64_t x);
    int64_t sub(int64_t x, int64_t y, double base);
    int64_t mul(int64_t x, int64_t y);
    int64_t div(int64_t x, int64_t y);
    int64_t reciprocal(int64_t x);
    int64_t square(int64_t x);
    int64_t sqrt(int64_t x);
    int64_t pow(int64_t x, double n);
    int64_t pow(int64_t x, int64_t n);

}

#endif // POINTWISE_OPS_H