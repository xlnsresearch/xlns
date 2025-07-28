#pragma once

namespace lns {

    /*
    These are constants used in the LNS implementation that are independent
    of base. This means that they can be pre-computed and reused across
    different bases.
    */

    inline constexpr long long zero_int = (-(1LL << 53)) | 1LL;
    inline constexpr double zero = static_cast<double>(zero_int);

    inline constexpr long long one_int = 0LL;
    inline constexpr double one = static_cast<double>(one_int);

    inline constexpr long long neg_one_int = 1LL;
    inline constexpr double neg_one = static_cast<double>(neg_one_int);

}