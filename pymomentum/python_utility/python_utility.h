/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>

#include <Python.h>
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <streambuf>
#include <vector>

// Utility functionality for dealing with Python types and momentum.

namespace pymomentum {

// If forceBatchSize is false, then it is allowed to return a list with a single
// element. This is sometimes useful to avoid redundant computation inside the
// Character.
std::vector<const momentum::Character*>
toCharacterList(PyObject* obj, int64_t nBatch, const char* context, bool forceBatchSize = true);

// Utility function to extract a single character from the character list.
// This is useful for performing checks against the character types before
// performing operations.
const momentum::Character& anyCharacter(PyObject* obj, const char* context);

nlohmann::json from_msgpack(const pybind11::bytes& bytes);

pybind11::bytes to_msgpack(const nlohmann::json& j);

class PyBytesStreamBuffer : public std::streambuf {
 public:
  explicit PyBytesStreamBuffer(const pybind11::bytes& bytes);

 private:
  // Store a copy of the bytes object to keep it alive for the lifetime of the streambuf
  pybind11::bytes bytes_;
};

} // namespace pymomentum
