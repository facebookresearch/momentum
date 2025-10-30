/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/mppca_io.h"

#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/math/mppca.h"

#include <fstream>

namespace momentum {

std::shared_ptr<const Mppca> loadMppca(const std::span<const unsigned char> posePriorDataRaw) {
  // Load from passed file data
  std::stringstream instream;
  std::copy(
      posePriorDataRaw.begin(),
      posePriorDataRaw.end(),
      std::ostream_iterator<unsigned char>(instream));

  return loadMppca(instream);
}

std::shared_ptr<const Mppca> loadMppca(const std::string& name) {
  std::ifstream data(name, std::ios::in | std::ios::binary);
  MT_THROW_IF(!data.good(), "Error loading Mppca model: could not open file '{}'.", name);

  return loadMppca(data);
}

std::shared_ptr<const Mppca> loadMppca(std::istream& inputStream) {
  auto result = std::make_shared<Mppca>();

  // try to load file
  try {
    MT_THROW_IF(!inputStream, "Error loading Mppca model: empty input stream.");

    // load dimensions
    uint64_t od = 0;
    uint64_t op = 0;
    inputStream.read((char*)&od, sizeof(od));
    inputStream.read((char*)&op, sizeof(op));

    // set data on result
    result->d = od;
    result->p = op;

    // load names
    result->names.resize(od);
    for (size_t i = 0; i < od; i++) {
      uint64_t count = 0;
      inputStream.read((char*)&count, sizeof(uint64_t));
      result->names[i].resize(count);
      inputStream.read((char*)result->names[i].data(), count);
    }

    // prepare data structures
    result->Rpre = VectorXf(op);
    result->Cinv.resize(op);
    result->L.resize(op);
    result->mu.resize(op, od);

    // load Rpre, Cinv, and mu
    inputStream.read((char*)result->Rpre.data(), sizeof(float) * op);
    for (size_t i = 0; i < op; i++) {
      result->Cinv[i] = MatrixXf(od, od);
      inputStream.read((char*)result->Cinv[i].data(), sizeof(float) * od * od);
      result->L[i] = result->Cinv[i].llt().matrixL().transpose();
    }
    inputStream.read((char*)result->mu.data(), sizeof(float) * op * od);

    return result;
  } catch (std::exception& e) {
    MT_THROW("Error loading Mppca model: {}", e.what());
  }
  return result;
}

void saveMppca(const Mppca& mppca, const std::string& name) {
  // write output file
  std::ofstream data(name, std::ios::out | std::ios::binary);
  if (!data.is_open()) {
    return;
  }

  // write dimensions
  const uint64_t od = gsl::narrow<uint64_t>(mppca.d);
  const uint64_t op = gsl::narrow<uint64_t>(mppca.p);
  data.write((char*)&od, sizeof(od));
  data.write((char*)&op, sizeof(op));

  // write out names
  MT_CHECK(mppca.names.size() >= od);
  for (size_t i = 0; i < od; i++) {
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    const std::string& nm = mppca.names[i];
    const uint64_t ln = nm.size();
    data.write((const char*)&ln, sizeof(uint64_t));
    data.write((const char*)nm.data(), ln);
  }

  // write pi, mu, W, sigma2
  data.write((char*)mppca.Rpre.data(), sizeof(float) * op);
  for (const auto& Cinv : mppca.Cinv) {
    data.write((char*)Cinv.data(), sizeof(float) * od * od);
  }
  data.write((char*)mppca.mu.data(), sizeof(float) * op * od);

  // close file
  data.close();
}

} // namespace momentum
