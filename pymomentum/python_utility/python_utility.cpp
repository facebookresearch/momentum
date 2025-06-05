/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/python_utility/python_utility.h"

#include <momentum/math/mesh.h>

#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace pymomentum {

namespace py = pybind11;

std::vector<const momentum::Character*> toCharacterList(
    PyObject* obj,
    int64_t nBatch,
    const char* context,
    bool forceBatchSize) {
  assert(obj != nullptr);
  if (PyList_Check(obj)) {
    std::vector<const momentum::Character*> result;
    for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
      try {
        result.push_back(
            py::cast<const momentum::Character*>(PyList_GetItem(obj, i)));
      } catch (const std::exception& e) {
        MT_THROW(
            "Expected a valid pymomentum.Character at index {} in list in {}: {}",
            i,
            context,
            e.what());
      }
    }

    MT_THROW_IF(
        result.empty(),
        "Expected a valid pymomentum.Character of list of Characters in {}",
        context);

    MT_THROW_IF(
        result.size() != nBatch,
        "Expected a valid pymomentum.Character of list of Characters in {}",
        context);

    for (const auto& c : result) {
      MT_THROW_IF(
          c->parameterTransform.name != result.front()->parameterTransform.name,
          "Mismatch in parameter transforms in {}; when processing a batch of characters, all characters must have the same parameters in the same order.",
          context);

      MT_THROW_IF(
          c->parameterTransform.numJointParameters() !=
              result.front()->parameterTransform.numJointParameters(),
          "Mismatch in parameter transforms in {}; when processing a batch of characters, all characters must have the same number of skeleton bones.",
          context);

      if (result.front()->mesh) {
        MT_THROW_IF(
            !c->mesh ||
                c->mesh->vertices.size() !=
                    result.front()->mesh->vertices.size(),
            "Mismatch in meshes in {}; when processing a batch of characters, all characters must have the same number of vertices.",
            context);
      }

      if (result.front()->skinWeights) {
        MT_THROW_IF(
            !c->skinWeights,
            "Mismatch in meshes in {}; when processing a batch of characters, all characters must have the same number of vertices.",
            context);
      }
    }

    return result;
  } else {
    try {
      const momentum::Character* character =
          py::cast<const momentum::Character*>(obj);
      return std::vector<const momentum::Character*>(
          forceBatchSize ? nBatch : 1, character);
    } catch (const std::exception& e) {
      MT_THROW(
          "Expected a valid pymomentum.Character in {}: {}", context, e.what());
    }
  }
}

// Extracts any single character that can be used to check the parameter counts,
// etc.
const momentum::Character& anyCharacter(PyObject* obj, const char* context) {
  assert(obj != nullptr);
  if (PyList_Check(obj)) {
    for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
      try {
        return *py::cast<const momentum::Character*>(PyList_GetItem(obj, i));
      } catch (std::exception&) {
        MT_THROW(
            "Expected a valid pymomentum.Character of list of Characters in {}",
            context);
      }
    }

    MT_THROW(
        "Expected a valid pymomentum.Character of list of Characters in {}; got an empty list.",
        context);
  }

  return *py::cast<const momentum::Character*>(obj);
}

nlohmann::json from_msgpack(const pybind11::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());
  const uint8_t* data = reinterpret_cast<const uint8_t*>(info.ptr);
  const size_t length = static_cast<size_t>(info.size);

  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.")

  return nlohmann::json::from_msgpack(data, data + length);
}

pybind11::bytes to_msgpack(const nlohmann::json& j) {
  // Use of Vector of char because pybind11::bytes would like to receive
  // const char* for initialization of bytes.
  std::string buffer;
  nlohmann::json::to_msgpack(j, buffer);
  pybind11::bytes pybytes(buffer);
  return pybytes;
}

PyBytesStreamBuffer::PyBytesStreamBuffer(const pybind11::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());

  // C++ does not distinguish between const and non-const streambufs, but we
  // promise to only use this with std::istreams.
  char* data = const_cast<char*>(reinterpret_cast<const char*>(info.ptr));
  const size_t length = static_cast<size_t>(info.size);

  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.")

  this->setg(data, data, data + length);
}

} // namespace pymomentum
