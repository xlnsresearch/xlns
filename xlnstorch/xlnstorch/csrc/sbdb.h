#ifndef XLNSTORCH_SBDB_H
#define XLNSTORCH_SBDB_H

#include <torch/extension.h>
#include <ATen/cpu/vec/vec.h>
#include <cstdint>
#include <map>
#include <string>

using int64_vec_t = at::vec::Vectorized<int64_t>;
using double_vec_t = at::vec::Vectorized<double>;

using sbdb_fn_ptr = int64_t(*)(int64_t, int64_t, double);
using sbdb_vec_fn_ptr = int64_vec_t(*)(int64_vec_t, int64_vec_t, double);

struct SbdbEntry {
    sbdb_fn_ptr scalar;     // element-wise implementation
    sbdb_vec_fn_ptr vector; // vectorized implementation
};

namespace sbdb {

    int64_t ideal(int64_t z, int64_t s, double base);
    int64_vec_t ideal_vec(int64_vec_t z, int64_vec_t s, double base);

    extern SbdbEntry default_entry;
    const std::map<std::string, SbdbEntry> funcs {
        {"ideal", {&ideal, &ideal_vec}}
    };

    void set_default_func(std::string sbdb_key);
}

void init_sbdb(py::module& m);

#endif // XLNSTORCH_SBDB_H