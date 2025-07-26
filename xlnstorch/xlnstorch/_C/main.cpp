#include <torch/extension.h>
#include "change_base.h"
#include "float_to_lns.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_float_to_lns(m);
    init_change_base(m);
}