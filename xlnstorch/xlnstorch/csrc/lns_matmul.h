#ifndef LNS_MATMUL_H
#define LNS_MATMUL_H

#include <torch/extension.h>

void init_lns_matmul(py::module& m);

#endif // LNS_MATMUL_H