#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "lns_utils.h"
#include "sbdb.h"
#include "lns_addition.h"
#include "lns_multiplication.h"
#include "lns_matmul.h"
#include "lns_convolution.h"
#include "lns_pooling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_lns_utils(m);
    init_sbdb(m);
    init_lns_addition(m);
    init_lns_multiplication(m);
    init_lns_matmul(m);
    init_lns_convolution(m);
    init_lns_pool(m);
}