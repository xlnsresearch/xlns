#include <torch/extension.h>
#include <map>
#include <string>

#include "sbdb.h"

namespace sbdb {

    SbdbEntry default_entry = {&ideal, &ideal_vec};

    inline int64_t ideal(int64_t z, int64_t s, double base) {

        double power_term = std::pow(base, z);
        double magnitude = std::abs(1.0 - 2.0 * s + power_term);
        double log_term = std::log(magnitude) / std::log(base);

        return std::llround(log_term) << 1;
    }

    inline int64_vec_t ideal_vec(int64_vec_t z, int64_vec_t s, double base) {
        double_vec_t z_double = at::vec::convert<double>(z);
        double_vec_t s_double = at::vec::convert<double>(s);
        double_vec_t base_vec(base);

        auto power_term = base_vec.pow(z_double);
        auto magnitude = (double_vec_t(1.0) - double_vec_t(2.0) * s_double + power_term).abs();
        auto log_term = magnitude.log() / base_vec.log();

        auto rounded = at::vec::convert<int64_t>(log_term.round());
        return rounded << int64_vec_t(1);
    }

    void set_default_func(std::string sbdb_key) {
        auto it = funcs.find(sbdb_key);

        if (it == funcs.end())
            default_entry = {&ideal, &ideal_vec};

        else
            default_entry = it->second;
    }

}

void init_sbdb(py::module& m) {
    m.def("set_default_sbdb_func", &sbdb::set_default_func, "Set the default SBDB function for C++ LNS operations");
}