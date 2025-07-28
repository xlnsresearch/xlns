#include <torch/extension.h>
#include "lns_utils.h"
#include "lns_addition.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_lns_utils(m);
    init_lns_addition(m);
}