/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/simd/simd.h>

namespace momentum {

// Forward declarations for camera types
template <typename T>
class CameraT;

template <typename T>
class IntrinsicsModelT;

template <typename T>
class PinholeIntrinsicsModelT;

template <typename T>
class OpenCVIntrinsicsModelT;

template <typename T>
struct OpenCVDistortionParametersT;

template <typename T>
struct OpenCVFisheyeDistortionParametersT;

template <typename T>
class OpenCVFisheyeIntrinsicsModelT;

using Camera = CameraT<float>;
using Camerad = CameraT<double>;
using IntrinsicsModel = IntrinsicsModelT<float>;
using IntrinsicsModeld = IntrinsicsModelT<double>;
using PinholeIntrinsicsModel = PinholeIntrinsicsModelT<float>;
using PinholeIntrinsicsModeld = PinholeIntrinsicsModelT<double>;
using OpenCVIntrinsicsModel = OpenCVIntrinsicsModelT<float>;
using OpenCVIntrinsicsModeld = OpenCVIntrinsicsModelT<double>;
using OpenCVDistortionParameters = OpenCVDistortionParametersT<float>;
using OpenCVDistortionParametersd = OpenCVDistortionParametersT<double>;
using OpenCVFisheyeDistortionParameters = OpenCVFisheyeDistortionParametersT<float>;
using OpenCVFisheyeDistortionParametersd = OpenCVFisheyeDistortionParametersT<double>;
using OpenCVFisheyeIntrinsicsModel = OpenCVFisheyeIntrinsicsModelT<float>;
using OpenCVFisheyeIntrinsicsModeld = OpenCVFisheyeIntrinsicsModelT<double>;

} // namespace momentum
