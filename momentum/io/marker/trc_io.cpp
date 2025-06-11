/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/marker/trc_io.h"

#include "momentum/common/string.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/io/marker/conversions.h"

#include <fstream>

namespace momentum {

MarkerSequence loadTrc(const std::string& filename, UpVector up) {
  MarkerSequence res;
  res.fps = 0.f;

  std::ifstream infile(filename);
  if (!infile.is_open()) {
    return res;
  }

  std::string line;

  // parse header
  GetLineCrossPlatform(infile, line);
  if (line.find("PathFileType\t4\t(X/Y/Z)") == std::string::npos) {
    return res;
  }

  GetLineCrossPlatform(infile, line);
  if (line.find(
          "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames") ==
      std::string::npos) {
    return res;
  }

  GetLineCrossPlatform(infile, line);
  auto tokens = tokenize(line, " \t\r\n");
  if (tokens.size() != 8) {
    return res;
  }
  const double frameRate = std::stod(tokens[0]);
  const size_t numMarkers = std::stoi(tokens[3]);
  res.fps = static_cast<float>(frameRate);

  // get line with marker names and parse it
  GetLineCrossPlatform(infile, line);
  tokens = tokenize(line, " \t\r\n");
  if (tokens.size() != numMarkers + 2) {
    res.fps = 0.0f;
    return res;
  }
  std::vector<std::string> markerNames;
  for (size_t i = 2; i < tokens.size(); i++)
    markerNames.push_back(tokens[i]);

  // ignore next line
  GetLineCrossPlatform(infile, line);

  while (GetLineCrossPlatform(infile, line)) {
    tokens = tokenize(line, "\t", false);
    // check for right size
    if (tokens.size() != numMarkers * 3 + 2) {
      continue;
    }

    // create actor structure for every frame
    std::vector<Marker> markers(numMarkers);

    // go over all markers
    for (size_t i = 0; i < numMarkers; i++) {
      const size_t pos = i * 3 + 2;
      markers[i].name = markerNames[i];
      if (tokens[pos + 0].empty() || tokens[pos + 1].empty() || tokens[pos + 2].empty()) {
        markers[i].occluded = true;
      } else {
        markers[i].occluded = false;
        Vector3d p{
            std::stod(tokens[pos + 0]), std::stod(tokens[pos + 1]), std::stod(tokens[pos + 2])};
        p = toMomentumVector3(p, up, tokens[4]);
        markers[i].pos = p;
      }
    }

    // store frame
    res.frames.push_back(markers);
  }

  return res;
}

} // namespace momentum
