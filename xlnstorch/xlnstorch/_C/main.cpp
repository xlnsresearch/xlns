#include <torch/extension.h>
#include "lns_utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_lns_utils(m);
}