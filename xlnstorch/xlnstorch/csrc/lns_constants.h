#ifndef LNS_CONSTANTS_H
#define LNS_CONSTANTS_H

#include <cstdint>

namespace lns {

    /*
    These are constants used in the LNS implementation that are independent
    of base. This means that they can be pre-computed and reused across
    different bases.
    */

    inline constexpr int64_t zero_int = (-(1LL << 53)) | 1LL;
    inline constexpr double zero = static_cast<double>(zero_int);

    inline constexpr int64_t one_int = 0LL;
    inline constexpr double one = static_cast<double>(one_int);

    inline constexpr int64_t neg_one_int = 1LL;
    inline constexpr double neg_one = static_cast<double>(neg_one_int);

}

#endif // LNS_CONSTANTS_H