#include <torch/torch.h>
#include <torch/extension.h>
#include <map>
#include <string>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>

#include "sbdb.h"

void get_table(
    torch::Tensor& tab_ez,
    torch::Tensor& tab_sbdb,
    torch::Tensor& tab_base
) {

    tab::base = tab_base.item<double>();
    tab::ez = tab_ez.item<int64_t>();
    tab::cols = tab_sbdb.numel() / 2;
    tab::sbdb = tab_sbdb.data_ptr<int64_t>();
    tab::initialized = true;
}

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

    inline int64_t tab(int64_t z, int64_t s, double base) {

        if (base == tab::base) {
            int64_t idx = (z == 0 ? -1 : z);
            idx = std::max(tab::ez, idx);

            int64_t wrapped = (idx - tab::ez) % tab::cols;
            if (wrapped < 0) wrapped += tab::cols;

            return tab::sbdb[s * tab::cols + wrapped];
        }

        return sbdb::ideal(z, s, base);
    }

    inline int64_vec_t tab_vec(int64_vec_t z, int64_vec_t s, double base) {

        if (base == tab::base) {
            constexpr int LANES = int64_vec_t::size();

            alignas(64) int64_t z_arr[LANES];
            alignas(64) int64_t s_arr[LANES];
            z.store(z_arr);
            s.store(s_arr);

            alignas(64) int64_t out_arr[LANES];

            for (int i = 0; i < LANES; ++i) {
                out_arr[i] = sbdb::tab(z_arr[i], s_arr[i], base);
            }

            return int64_vec_t::loadu(out_arr);

            // vectorized version doesn't currently work
            /*
            int64_vec_t idx = int64_vec_t::blendv(z, int64_vec_t(-1), z == int64_vec_t(0));
            idx = at::vec::maximum(idx, int64_vec_t(tab::ez));
            int64_vec_t diff = idx - int64_vec_t(tab::ez);

            constexpr int LANES = int64_vec_t::size();
            alignas(64) int64_t diff_buf[LANES];
            diff.store(diff_buf);

            alignas(64) int64_t wrap_buf[LANES];
            for (int i = 0; i < LANES; ++i) {
                int64_t w = diff_buf[i] % int64_t(tab::cols);
                if (w < 0) w += int64_t(tab::cols);
                wrap_buf[i] = w;
            }
            int64_vec_t wrapped = int64_vec_t::loadu(wrap_buf);

            alignas(64) int64_t idx_buf[LANES];
            int64_vec_t idx_vec = s * int64_vec_t(int64_t(tab::cols)) + wrapped;
            idx_vec.store(idx_buf);

            return at::vec::gather(tab::sbdb.data(), idx_buf);
            */
        }

        else {
            return sbdb::ideal_vec(z, s, base);
        }
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
    m.def(
        "set_default_sbdb_implementation",
        &sbdb::set_default_func,
        "Set the default SBDB function for C++ LNS operations",
        py::arg("sbdb_key")
    );
    m.def(
        "get_table",
        &get_table,
        "Get the SBDB table for a given base precision",
        py::arg("tab_ez"),
        py::arg("tab_sbdb"),
        py::arg("tab_base")
    );
}