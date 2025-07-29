#ifndef POINTWISE_OPS_H
#define POINTWISE_OPS_H

#include <string>
#include <map>

namespace sbdb {

    long long ideal(long long z, long long s, double base);

}

namespace lns {

    using sbdb_fn_ptr = long long(*)(long long, long long, double);
    extern const std::map<std::string, sbdb_fn_ptr> sbdb_funcs;

    extern sbdb_fn_ptr default_sbdb_func;
    void set_default_sbdb_func(std::string sbdb_key);

    long long add(long long x, long long y, double base);
    long long neg(long long x);
    long long sub(long long x, long long y, double base);

}

void init_pointwise_ops(py::module& m);

#endif // POINTWISE_OPS_H