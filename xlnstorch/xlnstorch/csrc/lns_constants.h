#ifndef LNS_CONSTANTS_H
#define LNS_CONSTANTS_H

#include <cstdint>

namespace lns {

    /*
    These are constants used in the LNS implementation that are independent
    of base. This means that they can be pre-computed and reused across
    different bases.
    */

    inline constexpr int64_t zero_int = -9223372036854775807LL;
    inline constexpr int64_t one_int = 0LL;
    inline constexpr int64_t neg_one_int = 1LL;

}

#endif // LNS_CONSTANTS_H