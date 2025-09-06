/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <rerun.hpp>
#include <Eigen/Core>

#include <algorithm>
#include <cstring>
#include <vector>

namespace rerun {

// Rerun Adapter for Eigen::Vector3<T>
template <typename TElement, typename T>
struct CollectionAdapter<TElement, std::vector<Eigen::Vector3<T>>> {
  /// Borrow for non-temporary.
  Collection<TElement> operator()(const std::vector<Eigen::Vector3<T>>& container) {
    return Collection<TElement>::borrow(container.data(), container.size());
  }

  // Do a full copy for temporaries (otherwise the data might be deleted when the temporary is
  // destroyed).
  Collection<TElement> operator()(std::vector<Eigen::Vector3<T>>&& container) {
    std::vector<TElement> positions(container.size());
    std::memcpy(positions.data(), container.data(), container.size() * sizeof(Eigen::Vector3<T>));
    return Collection<TElement>::take_ownership(std::move(positions));
  }
};

// Rerun Adapter for Eigen::Vector4<T>
template <typename TElement, typename T>
struct CollectionAdapter<TElement, std::vector<Eigen::Vector4<T>>> {
  /// Borrow for non-temporary.
  Collection<TElement> operator()(const std::vector<Eigen::Vector4<T>>& container) {
    return Collection<TElement>::borrow(container.data(), container.size());
  }

  // Do a full copy for temporaries (otherwise the data might be deleted when the temporary is
  // destroyed).
  Collection<TElement> operator()(std::vector<Eigen::Vector4<T>>&& container) {
    std::vector<TElement> positions(container.size());
    std::memcpy(positions.data(), container.data(), container.size() * sizeof(Eigen::Vector4<T>));
    return Collection<TElement>::take_ownership(std::move(positions));
  }
};

// Adapter for converting a vector of Eigen::Vector3b to a vector of rerun::Color
template <>
struct CollectionAdapter<Color, std::vector<Eigen::Vector3<uint8_t>>> {
  Collection<Color> operator()(const std::vector<Eigen::Vector3<uint8_t>>& container) {
    std::vector<rerun::Color> colors;
    colors.reserve(container.size());
    std::transform(
        container.cbegin(),
        container.cend(),
        std::back_inserter(colors),
        [](const Eigen::Vector3<uint8_t>& vertexColor) {
          return rerun::Color(vertexColor[0], vertexColor[1], vertexColor[2]);
        });
    return Collection<Color>::take_ownership(std::move(colors));
  }
};

} // namespace rerun
