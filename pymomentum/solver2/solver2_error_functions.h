#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

void addErrorFunctions(pybind11::module_& m);

}
