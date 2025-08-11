#include <torch/torch.h>
#include <torch/extension.h>
#include <map>
#include <string>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cnpy.h" // see https://github.com/rogersce/cnpy
#include "sbdb.h"

static std::string decimal_suffix(double d) {
    std::ostringstream oss;
    oss << std::fixed
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << d;
    std::string s = oss.str();

    auto dot = s.find('.');
    if (dot == std::string::npos)
        return {};
    std::string frac = s.substr(dot + 1);

    while (!frac.empty() && frac.back() == '0')
        frac.pop_back();

    return frac;
}

double get_base_from_precision(int prec) {
    return std::pow(2.0, std::pow(2.0, -prec));
}

void get_table(
    const std::string &filestem,
    const double base
) {

    // not used for now
    tab::mismatch = false;
    tab::base = base;

    const std::string filename = "./" + filestem + "_" + decimal_suffix(tab::base) + ".npz";

    if (std::filesystem::exists(filename)) {
        std::cout << "Loading table from " << filename << '\n';
        cnpy::npz_t npzFile = cnpy::npz_load(filename);

        // tab::ez - scalar
        {
            const cnpy::NpyArray& arr = npzFile["tab::ez"];
            tab::ez = *arr.data<int64_t>();
        }

        // tab::sbdb - [2, N]
        {
            const cnpy::NpyArray& arr = npzFile["tab::sbdb"];
            const std::size_t rows = arr.shape[0];
            const std::size_t cols = arr.shape[1];
            const std::size_t numel = rows * cols;
            tab::cols = cols;

            tab::sbdb.resize(numel);
            std::memcpy(tab::sbdb.data(), arr.data<int64_t>(), numel * sizeof(int64_t));
        }

        tab::initialized = true;
        return;
    }

    const double max_base = get_base_from_precision(tab::MAX_PREC);
    if (tab::base >= max_base) {
        std::cout << "Creating ideal table as " << filename << '\n';

        tab::ez = sbdb::ideal(1, 1, tab::base);

        const int64_t first_z = tab::ez;
        const int64_t last_z = -1;
        tab::cols = static_cast<std::size_t>(-first_z);

        tab::sbdb.resize(2 * tab::cols);
        std::size_t col = 0;
        for (int64_t z = first_z; z <= last_z; ++z, ++col) {
            tab::sbdb[0 * tab::cols + col] = sbdb::ideal(z, 0, tab::base);
            tab::sbdb[1 * tab::cols + col] = sbdb::ideal(z, 1, tab::base);
        }

        cnpy::npz_save(filename,
                       "tab::ez",
                       &tab::ez,
                       {static_cast<std::size_t>(1)},
                       "w");

        cnpy::npz_save(filename,
                       "tab::sbdb",
                       tab::sbdb.data(),
                       {static_cast<std::size_t>(2), tab::cols},
                       "a");

        tab::initialized = true;
        return;
    }

    std::cerr << "Warning: Table for base " << tab::base
              << " is too large to create. Max precision is "
              << tab::MAX_PREC << std::endl;
    tab::base = 0.0;
    tab::initialized = false;
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
    m.def("set_default_sbdb_implementation", &sbdb::set_default_func, "Set the default SBDB function for C++ LNS operations");
    m.def("get_table", &get_table, "Get the SBDB table for a given base precision");
}