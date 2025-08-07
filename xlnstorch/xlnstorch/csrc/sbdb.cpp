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
    std::string s = std::to_string(d);
    auto pos = s.find('.');
    return pos == std::string::npos ? s : s.substr(pos + 1);
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
            auto opts = at::TensorOptions(at::kCPU).dtype(torch::kInt64);

            tab::ez = torch::empty({}, opts);
            std::copy_n(arr.data<int64_t>(), 1, tab::ez.data_ptr<int64_t>());
        }

        // tab::sbdb - [2, N]
        {
            const cnpy::NpyArray& arr = npzFile["tab::sbdb"];
            const std::size_t rows = arr.shape[0];
            const std::size_t cols = arr.shape[1];
            const std::size_t numel = rows * cols;

            auto opts = at::TensorOptions(at::kCPU).dtype(torch::kInt64);
            tab::sbdb  = torch::empty({static_cast<long>(rows), static_cast<long>(cols)}, opts);

            std::copy_n(arr.data<int64_t>(), numel, tab::sbdb.data_ptr<int64_t>());
        }

        tab::initialized = true;
        return;
    }

    const double max_base = get_base_from_precision(tab::MAX_PREC);
    if (tab::base >= max_base) {
        std::cout << "Creating ideal table as " << filename << '\n';

        int64_t ez_val = sbdb::ideal(1, 1, tab::base);
        tab::ez = torch::tensor(ez_val, torch::TensorOptions(at::kCPU).dtype(torch::kInt64));

        auto opts = at::TensorOptions(at::kCPU).dtype(torch::kInt64);
        torch::Tensor zrange = torch::arange(ez_val, 0, opts);

        const int64_t N = zrange.numel();
        torch::Tensor sbt = torch::empty({N}, opts);
        torch::Tensor dbt = torch::empty({N}, opts);

        auto z_accessor = zrange.data_ptr<int64_t>();
        auto sb_ptr = sbt.data_ptr<int64_t>();
        auto db_ptr = dbt.data_ptr<int64_t>();

        for (int64_t i = 0; i < N; ++i) {
            const int64_t z = z_accessor[i];
            sb_ptr[i] = sbdb::ideal(z, 0, tab::base);
            db_ptr[i] = sbdb::ideal(z, 1, tab::base);
        }

        tab::sbdb = torch::stack({sbt, dbt}); // [2, N]

        cnpy::npz_save(
            filename,
            "tab::ez",
            tab::ez.cpu().data_ptr<int64_t>(),
            {static_cast<size_t>(tab::ez.numel())},
            "w"
        );
        cnpy::npz_save(
            filename,
            "tab::sbdb",
            tab::sbdb.cpu().data_ptr<int64_t>(),
            {static_cast<size_t>(tab::sbdb.size(0)),
                static_cast<size_t>(tab::sbdb.size(1))},
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
            const int64_t ez = tab::ez.item<int64_t>();

            int64_t idx = (z == 0 ? -1 : z);
            idx = std::max(ez, idx);

            return tab::sbdb.index({s, idx}).item<int64_t>();
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

            // vectorized version not working
            // const int64_t ez_scalar = tab::ez.item<int64_t>();
            // const int64_vec_t ez_vec(ez_scalar);

            // int64_vec_t idx = int64_vec_t::blendv(z, int64_vec_t(-1), z == int64_vec_t(0));
            // idx = at::vec::maximum(idx, ez_vec);

            // const int64_t N = tab::sbdb.size(1);
            // const int64_vec_t Nvec(N);

            // const int64_vec_t neg_mask = (idx < int64_vec_t(0));
            // const int64_vec_t idx_pos = int64_vec_t::blendv(idx, idx + Nvec, neg_mask);
            // int64_vec_t offsets = s * Nvec + idx_pos;

            // const int64_t* base_ptr = tab::sbdb.data_ptr<int64_t>();
            // return at::vec::gather(base_ptr, offsets);
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