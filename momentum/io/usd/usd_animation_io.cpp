/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_animation_io.h"

#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/math/utility.h"

#include <pxr/base/gf/half.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/bindingAPI.h>

#include <cmath>
#include <unordered_map>
#include <unordered_set>

PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

void saveSkeletonStatesToUsd(
    const UsdStageRefPtr& stage,
    UsdSkelSkeleton& skelPrim,
    const Skeleton& skeleton,
    std::span<const SkeletonState> skeletonStates,
    float fps) {
  MT_CHECK(!skeletonStates.empty());

  const auto numJoints = skeleton.joints.size();

  // Create animation prim
  auto animPath = skelPrim.GetPath().AppendChild(TfToken("Animation"));
  auto animPrim = UsdSkelAnimation::Define(stage, animPath);

  // Set joint order to match skeleton
  VtArray<TfToken> jointTokens;
  jointTokens.reserve(numJoints);
  for (const auto& joint : skeleton.joints) {
    jointTokens.push_back(TfToken(joint.name));
  }
  animPrim.GetJointsAttr().Set(jointTokens);

  // Bind animation to skeleton
  UsdSkelBindingAPI bindingAPI = UsdSkelBindingAPI::Apply(skelPrim.GetPrim());
  bindingAPI.GetAnimationSourceRel().SetTargets({animPath});

  // Set stage time range
  const double startTime = 0.0;
  const auto endTime = static_cast<double>(skeletonStates.size() - 1);
  stage->SetStartTimeCode(startTime);
  stage->SetEndTimeCode(endTime);
  stage->SetTimeCodesPerSecond(static_cast<double>(fps));

  // Write time-sampled TRS per frame
  VtArray<GfVec3f> translations;
  VtArray<GfQuatf> rotations;
  VtArray<GfVec3h> scales;
  translations.reserve(numJoints);
  rotations.reserve(numJoints);
  scales.reserve(numJoints);

  auto translationsAttr = animPrim.GetTranslationsAttr();
  auto rotationsAttr = animPrim.GetRotationsAttr();
  auto scalesAttr = animPrim.GetScalesAttr();

  for (size_t frame = 0; frame < skeletonStates.size(); ++frame) {
    const auto& state = skeletonStates[frame];
    const auto timeCode = UsdTimeCode(static_cast<double>(frame));

    translations.clear();
    rotations.clear();
    scales.clear();

    for (size_t j = 0; j < numJoints; ++j) {
      const auto& localTransform = state.jointState[j].localTransform;

      const auto& t = localTransform.translation;
      translations.push_back(GfVec3f(t.x(), t.y(), t.z()));

      const auto& q = localTransform.rotation;
      rotations.push_back(GfQuatf(q.w(), q.x(), q.y(), q.z()));

      const GfHalf s = static_cast<GfHalf>(localTransform.scale);
      scales.push_back(GfVec3h(s, s, s));
    }

    translationsAttr.Set(translations, timeCode);
    rotationsAttr.Set(rotations, timeCode);
    scalesAttr.Set(scales, timeCode);
  }
}

std::tuple<std::vector<SkeletonState>, std::vector<float>> loadSkeletonStatesFromUsd(
    const UsdStageRefPtr& stage,
    const Skeleton& skeleton) {
  std::vector<SkeletonState> states;
  std::vector<float> frameTimes;

  const auto numJoints = skeleton.joints.size();

  // Find animation prim
  for (const auto& prim : stage->Traverse()) {
    if (!prim.IsA<UsdSkelAnimation>()) {
      continue;
    }

    UsdSkelAnimation animPrim(prim);

    const double tps = stage->GetTimeCodesPerSecond();
    if (tps <= 0.0) {
      break;
    }

    // Query time samples from the animation
    std::vector<double> timeSamples;
    animPrim.GetTranslationsAttr().GetTimeSamples(&timeSamples);
    if (timeSamples.empty()) {
      break;
    }

    states.reserve(timeSamples.size());
    frameTimes.reserve(timeSamples.size());

    VtArray<GfVec3f> translations;
    VtArray<GfQuatf> rotations;
    VtArray<GfVec3h> scales;

    auto translationsAttr = animPrim.GetTranslationsAttr();
    auto rotationsAttr = animPrim.GetRotationsAttr();
    auto scalesAttr = animPrim.GetScalesAttr();

    for (const double sample : timeSamples) {
      const auto timeCode = UsdTimeCode(sample);

      translationsAttr.Get(&translations, timeCode);
      rotationsAttr.Get(&rotations, timeCode);
      scalesAttr.Get(&scales, timeCode);

      // Reconstruct joint parameters from TRS
      JointParameters jointParams = JointParameters::Zero(numJoints * kParametersPerJoint);

      for (size_t j = 0; j < numJoints; ++j) {
        const size_t base = j * kParametersPerJoint;
        const auto& joint = skeleton.joints[j];

        // Translation: subtract translationOffset to get the parameter delta
        if (j < translations.size()) {
          jointParams[base + 0] = translations[j][0] - joint.translationOffset.x();
          jointParams[base + 1] = translations[j][1] - joint.translationOffset.y();
          jointParams[base + 2] = translations[j][2] - joint.translationOffset.z();
        }

        // Rotation: remove preRotation then convert to euler angles (ZYX convention)
        if (j < rotations.size()) {
          const auto& gfQ = rotations[j];
          Eigen::Quaternionf fullRotation(
              gfQ.GetReal(), gfQ.GetImaginary()[0], gfQ.GetImaginary()[1], gfQ.GetImaginary()[2]);

          // The saved rotation is preRotation * Rz * Ry * Rx, so remove preRotation
          Eigen::Quaternionf jointRotation = joint.preRotation.inverse() * fullRotation;
          Eigen::Vector3f euler = quaternionToEuler(jointRotation);
          jointParams[base + 3] = euler[0];
          jointParams[base + 4] = euler[1];
          jointParams[base + 5] = euler[2];
        }

        // Scale: take log2 to get the parameter (forward: scale = exp2(param))
        if (j < scales.size()) {
          const auto scaleVal = static_cast<float>(scales[j][0]);
          jointParams[base + 6] = scaleVal > 0.0f ? std::log2(scaleVal) : 0.0f;
        }
      }

      states.emplace_back(std::move(jointParams), skeleton, false);
      frameTimes.push_back(static_cast<float>(sample / tps));
    }

    break; // Use first animation found
  }

  return {std::move(states), std::move(frameTimes)};
}

void saveMotionToUsd(
    const UsdPrim& prim,
    float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets) {
  const auto& [paramNames, poses] = motion;
  const auto& [jointNames, offsetValues] = offsets;

  if (paramNames.empty() || poses.cols() == 0) {
    return;
  }

  prim.CreateAttribute(TfToken("momentum:motion:fps"), SdfValueTypeNames->Float).Set(fps);

  // Save parameter names as string array
  VtArray<std::string> paramNamesArray(paramNames.begin(), paramNames.end());
  prim.CreateAttribute(TfToken("momentum:motion:parameterNames"), SdfValueTypeNames->StringArray)
      .Set(paramNamesArray);

  // Save poses matrix as flat float array (column-major: numParams * numFrames)
  const Eigen::Index numParams = poses.rows();
  const Eigen::Index numFrames = poses.cols();
  VtArray<float> posesArray(numParams * numFrames);
  for (Eigen::Index f = 0; f < numFrames; ++f) {
    for (Eigen::Index p = 0; p < numParams; ++p) {
      posesArray[f * numParams + p] = poses(p, f);
    }
  }
  prim.CreateAttribute(TfToken("momentum:motion:poses"), SdfValueTypeNames->FloatArray)
      .Set(posesArray);
  prim.CreateAttribute(TfToken("momentum:motion:numFrames"), SdfValueTypeNames->Int)
      .Set(static_cast<int>(numFrames));
  prim.CreateAttribute(TfToken("momentum:motion:numParams"), SdfValueTypeNames->Int)
      .Set(static_cast<int>(numParams));

  // Save identity offsets
  if (!jointNames.empty() && offsetValues.size() > 0) {
    VtArray<std::string> jointNamesArray(jointNames.begin(), jointNames.end());
    prim.CreateAttribute(TfToken("momentum:motion:jointNames"), SdfValueTypeNames->StringArray)
        .Set(jointNamesArray);

    VtArray<float> offsetsArray(offsetValues.size());
    for (Eigen::Index i = 0; i < offsetValues.size(); ++i) {
      offsetsArray[i] = offsetValues[i];
    }
    prim.CreateAttribute(TfToken("momentum:motion:offsets"), SdfValueTypeNames->FloatArray)
        .Set(offsetsArray);
  }
}

std::tuple<MotionParameters, IdentityParameters, float> loadMotionFromUsd(const UsdPrim& prim) {
  MotionParameters motion;
  IdentityParameters identity;
  float fps = 120.0f;

  auto fpsAttr = prim.GetAttribute(TfToken("momentum:motion:fps"));
  if (!fpsAttr) {
    return {motion, identity, fps};
  }
  fpsAttr.Get(&fps);

  int numFrames = 0;
  int numParams = 0;
  prim.GetAttribute(TfToken("momentum:motion:numFrames")).Get(&numFrames);
  prim.GetAttribute(TfToken("momentum:motion:numParams")).Get(&numParams);

  if (numFrames <= 0 || numParams <= 0) {
    return {motion, identity, fps};
  }

  // Load parameter names
  VtArray<std::string> paramNamesArray;
  if (auto attr = prim.GetAttribute(TfToken("momentum:motion:parameterNames")); attr) {
    attr.Get(&paramNamesArray);
  }
  auto& [paramNames, poses] = motion;
  paramNames.assign(paramNamesArray.begin(), paramNamesArray.end());

  // Load poses matrix
  VtArray<float> posesArray;
  if (auto attr = prim.GetAttribute(TfToken("momentum:motion:poses")); attr) {
    attr.Get(&posesArray);
  }

  const auto expectedSize = static_cast<size_t>(numParams) * static_cast<size_t>(numFrames);
  if (posesArray.size() == expectedSize) {
    poses.resize(numParams, numFrames);
    for (int f = 0; f < numFrames; ++f) {
      for (int p = 0; p < numParams; ++p) {
        poses(p, f) = posesArray[f * numParams + p];
      }
    }
  } else {
    MT_LOGW(
        "Poses array size ({}) does not match numParams * numFrames ({} * {})",
        posesArray.size(),
        numParams,
        numFrames);
  }

  // Load identity offsets
  auto& [jointNames, offsetValues] = identity;

  VtArray<std::string> jointNamesArray;
  if (auto attr = prim.GetAttribute(TfToken("momentum:motion:jointNames")); attr) {
    attr.Get(&jointNamesArray);
  }
  jointNames.assign(jointNamesArray.begin(), jointNamesArray.end());

  VtArray<float> offsetsArray;
  if (auto attr = prim.GetAttribute(TfToken("momentum:motion:offsets")); attr) {
    attr.Get(&offsetsArray);
    Eigen::VectorXf offsetVec(offsetsArray.size());
    for (size_t i = 0; i < offsetsArray.size(); ++i) {
      offsetVec[i] = offsetsArray[i];
    }
    offsetValues = JointParameters(offsetVec);
  }

  return {std::move(motion), std::move(identity), fps};
}

void saveMarkerSequenceToUsd(
    const UsdStageRefPtr& stage,
    const SdfPath& skelRootPath,
    float fps,
    std::span<const std::vector<Marker>> markerSequence) {
  if (markerSequence.empty()) {
    return;
  }

  // Collect unique marker names across all frames
  std::vector<std::string> markerNames;
  markerNames.reserve(markerSequence[0].size());
  std::unordered_set<std::string> seenNames;
  for (const auto& frameMarkers : markerSequence) {
    for (const auto& marker : frameMarkers) {
      if (seenNames.insert(marker.name).second) {
        markerNames.push_back(marker.name);
      }
    }
  }

  if (markerNames.empty()) {
    return;
  }

  // Build per-frame name-to-index maps for O(1) marker lookup
  std::vector<std::unordered_map<std::string_view, size_t>> frameIndices(markerSequence.size());
  MT_CHECK(!frameIndices.empty());
  for (size_t f = 0; f < markerSequence.size(); ++f) {
    frameIndices[f].reserve(markerSequence[f].size());
    for (size_t i = 0; i < markerSequence[f].size(); ++i) {
      frameIndices[f][markerSequence[f][i].name] = i;
    }
  }

  // Create Markers scope
  auto markersPath = skelRootPath.AppendChild(TfToken("Markers"));
  UsdGeomScope::Define(stage, markersPath);

  // Store fps as attribute on the Markers scope
  auto markersPrim = stage->GetPrimAtPath(markersPath);
  markersPrim.CreateAttribute(TfToken("momentum:markers:fps"), SdfValueTypeNames->Float).Set(fps);

  // Track sanitized names to detect collisions
  std::unordered_set<std::string> usedSafeNames;

  // Create an Xform prim for each marker
  for (const auto& name : markerNames) {
    // Sanitize name for USD path (replace spaces/special chars)
    std::string safeName = name;
    for (auto& c : safeName) {
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
        c = '_';
      }
    }

    // Disambiguate if sanitized name collides with an existing one
    std::string baseName = safeName;
    int suffix = 1;
    while (!usedSafeNames.insert(safeName).second) {
      safeName = baseName + "_" + std::to_string(suffix++);
    }

    auto markerPath = markersPath.AppendChild(TfToken(safeName));
    auto xformPrim = UsdGeomXform::Define(stage, markerPath);

    // Store original name as attribute
    xformPrim.GetPrim()
        .CreateAttribute(TfToken("momentum:marker:name"), SdfValueTypeNames->String)
        .Set(name);

    auto xformAPI = UsdGeomXformCommonAPI(xformPrim);

    // Create the occluded attribute once per marker, then reuse inside the loop
    auto occludedAttr = xformPrim.GetPrim().CreateAttribute(
        TfToken("momentum:marker:occluded"), SdfValueTypeNames->Bool);

    // Write time-sampled translations and occlusion
    for (size_t frame = 0; frame < markerSequence.size(); ++frame) {
      const auto timeCode = UsdTimeCode(static_cast<double>(frame));

      // Find this marker in the frame via index map
      auto it = frameIndices[frame].find(name);
      if (it != frameIndices[frame].end()) {
        const auto& marker = markerSequence[frame][it->second];
        xformAPI.SetTranslate(GfVec3d(marker.pos.x(), marker.pos.y(), marker.pos.z()), timeCode);
        occludedAttr.Set(marker.occluded, timeCode);
      } else {
        // Marker not present in this frame — mark as occluded at origin
        xformAPI.SetTranslate(GfVec3d(0.0, 0.0, 0.0), timeCode);
        occludedAttr.Set(true, timeCode);
      }
    }
  }

  // Update stage time range
  const auto endTime = static_cast<double>(markerSequence.size() - 1);
  if (stage->GetEndTimeCode() < endTime) {
    stage->SetEndTimeCode(endTime);
  }
  if (stage->GetStartTimeCode() > 0.0) {
    stage->SetStartTimeCode(0.0);
  }
  stage->SetTimeCodesPerSecond(static_cast<double>(fps));
}

MarkerSequence loadMarkerSequenceFromUsd(const UsdStageRefPtr& stage) {
  MarkerSequence result;

  // Find the Markers scope
  UsdPrim markersScopePrim;
  for (const auto& prim : stage->Traverse()) {
    if (prim.GetName() == TfToken("Markers") && prim.IsA<UsdGeomScope>()) {
      markersScopePrim = prim;
      break;
    }
  }

  if (!markersScopePrim) {
    return result;
  }

  // Read fps
  auto fpsAttr = markersScopePrim.GetAttribute(TfToken("momentum:markers:fps"));
  if (fpsAttr) {
    fpsAttr.Get(&result.fps);
  }
  // TODO: result.name = ...;

  // Collect marker prims
  struct MarkerPrimInfo {
    std::string name;
    UsdGeomXform xform;
  };
  std::vector<MarkerPrimInfo> markerPrims;

  for (const auto& child : markersScopePrim.GetChildren()) {
    if (!child.IsA<UsdGeomXform>()) {
      continue;
    }

    std::string markerName;
    auto nameAttr = child.GetAttribute(TfToken("momentum:marker:name"));
    if (nameAttr) {
      nameAttr.Get(&markerName);
    } else {
      markerName = child.GetName().GetString();
    }

    markerPrims.push_back({markerName, UsdGeomXform(child)});
  }

  if (markerPrims.empty()) {
    return result;
  }

  // Determine frame count from actual time samples on marker data
  std::vector<double> timeSamples;
  auto firstOccludedAttr =
      markerPrims[0].xform.GetPrim().GetAttribute(TfToken("momentum:marker:occluded"));
  if (firstOccludedAttr) {
    firstOccludedAttr.GetTimeSamples(&timeSamples);
  }

  if (timeSamples.empty()) {
    return result;
  }

  const auto numFrames = static_cast<int>(timeSamples.size());

  result.frames.resize(numFrames);

  // Pre-cache per-marker API objects to avoid reconstructing them every frame
  struct CachedMarkerInfo {
    std::string name;
    UsdGeomXformCommonAPI xformAPI;
    UsdAttribute occludedAttr;
  };
  std::vector<CachedMarkerInfo> cachedMarkers;
  cachedMarkers.reserve(markerPrims.size());
  for (const auto& [name, xform] : markerPrims) {
    cachedMarkers.push_back(
        {name,
         UsdGeomXformCommonAPI(xform),
         xform.GetPrim().GetAttribute(TfToken("momentum:marker:occluded"))});
  }

  for (int frame = 0; frame < numFrames; ++frame) {
    const auto timeCode = UsdTimeCode(timeSamples[frame]);
    auto& frameMarkers = result.frames[frame];
    frameMarkers.reserve(cachedMarkers.size());

    for (const auto& cached : cachedMarkers) {
      Marker marker;
      marker.name = cached.name;

      // Read translation
      GfVec3d translate{};
      GfVec3f rotation{};
      GfVec3f scale{};
      GfVec3f pivot{};
      UsdGeomXformCommonAPI::RotationOrder rotOrder{};
      if (cached.xformAPI.GetXformVectors(
              &translate, &rotation, &scale, &pivot, &rotOrder, timeCode)) {
        marker.pos = Vector3d(translate[0], translate[1], translate[2]);
      }

      // Read occlusion
      if (cached.occludedAttr) {
        bool occluded = true;
        cached.occludedAttr.Get(&occluded, timeCode);
        marker.occluded = occluded;
      }

      frameMarkers.push_back(std::move(marker));
    }
  }

  return result;
}

} // namespace momentum
