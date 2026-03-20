/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_animation_io.h"

#include "momentum/common/checks.h"
#include "momentum/math/utility.h"

#include <pxr/base/gf/half.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/bindingAPI.h>

#include <cmath>

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

} // namespace momentum
