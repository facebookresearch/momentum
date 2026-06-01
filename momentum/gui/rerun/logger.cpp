/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/gui/rerun/logger.h"

#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/locator.h"
#include "momentum/character/locator_state.h"
#include "momentum/character/marker.h"
#include "momentum/gui/rerun/eigen_adapters.h"
#include "momentum/gui/rerun/rerun_compat.h"
#include "momentum/math/mesh.h"

#include <axel/Bvh.h>
#include <fmt/format.h>
#include <rerun.hpp>

#include <array>
#include <vector>

namespace momentum {

namespace {

template <typename Derived>
std::array<float, 3> toStdArray3f(const Eigen::MatrixBase<Derived>& vec3) {
  return {static_cast<float>(vec3[0]), static_cast<float>(vec3[1]), static_cast<float>(vec3[2])};
}

template <typename Derived>
rerun::Position3D toRerunPosition3D(const Eigen::MatrixBase<Derived>& vec3) {
  return rerun::Position3D(vec3[0], vec3[1], vec3[2]);
}

template <typename Derived>
rerun::HalfSize3D toRerunHalfSizes3D(const Eigen::MatrixBase<Derived>& vec3) {
  return rerun::HalfSize3D(vec3[0], vec3[1], vec3[2]);
}

} // namespace

namespace detail {

CollisionEllipsoidLogData makeCollisionEllipsoidLogData(
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState) {
  CollisionEllipsoidLogData data;
  data.centers.reserve(collisionGeometry.size());
  data.halfSizes.reserve(collisionGeometry.size());
  data.quaternions.reserve(collisionGeometry.size());

  for (const auto& cg : collisionGeometry) {
    if (cg.type != CollisionPrimitiveType::Ellipsoid) {
      continue;
    }

    const Transform parentTransform = (cg.parent == kInvalidIndex)
        ? Transform()
        : skeletonState.jointState.at(cg.parent).transform;
    const Transform tf = parentTransform * cg.transformation;

    data.centers.push_back(tf.translation);
    // Radii are scaled by the parent joint scale only (matches CollisionGeometryStateT::update).
    data.halfSizes.emplace_back(cg.ellipsoidRadii * parentTransform.scale);
    data.quaternions.push_back(tf.rotation);
  }

  return data;
}

CollisionBoxLogData makeCollisionBoxLogData(
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState) {
  CollisionBoxLogData data;
  data.centers.reserve(collisionGeometry.size());
  data.halfSizes.reserve(collisionGeometry.size());
  data.quaternions.reserve(collisionGeometry.size());

  for (const auto& cg : collisionGeometry) {
    if (cg.type != CollisionPrimitiveType::Box) {
      continue;
    }

    const Transform parentTransform = (cg.parent == kInvalidIndex)
        ? Transform()
        : skeletonState.jointState.at(cg.parent).transform;
    const Transform tf = parentTransform * cg.transformation;

    data.centers.push_back(tf.translation);
    data.halfSizes.emplace_back(cg.boxHalfExtents * parentTransform.scale);
    data.quaternions.push_back(tf.rotation);
  }

  return data;
}

} // namespace detail

void logMesh(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const Mesh& mesh,
    std::optional<rerun::Color> color) {
  auto rerunMesh = rerun::Mesh3D(mesh.vertices).with_triangle_indices(mesh.faces);
  if (color.has_value()) {
    rerunMesh = std::move(rerunMesh).with_albedo_factor(rerun::AlbedoFactor(color.value()));
  } else if (mesh.colors.size() == mesh.vertices.size()) {
    rerunMesh = std::move(rerunMesh).with_vertex_colors(mesh.colors);
  }

  if (mesh.normals.size() == mesh.vertices.size()) {
    rerunMesh = std::move(rerunMesh).with_vertex_normals(mesh.normals);
  }

  safeLog(rec, streamName, rerunMesh);
}

void logJoints(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const Skeleton& skeleton,
    const JointStateList& jointStates) {
  const rerun::Color kGrey(180, 180, 180);
  const auto names = skeleton.getJointNames();
  std::vector<std::string> labels;
  std::vector<std::vector<std::array<float, 3>>> lines;
  labels.reserve(names.size());
  lines.reserve(names.size());
  for (size_t iJoint = 0; iJoint < jointStates.size(); ++iJoint) {
    if (iJoint >= skeleton.joints.size()) {
      break;
    }
    const size_t parentIdx = skeleton.joints[iJoint].parent;
    if (parentIdx != kInvalidIndex && parentIdx < jointStates.size()) {
      lines.push_back(
          {toStdArray3f(jointStates[parentIdx].transform.translation),
           toStdArray3f(jointStates[iJoint].transform.translation)});
      labels.push_back(names[iJoint]);
    }
    safeLog(
        rec,
        streamName + "/" + names[iJoint],
        rerun::Transform3D()
            .with_mat3x3(
                rerun::datatypes::Mat3x3(jointStates[iJoint].transform.toRotationMatrix().data()))
            .with_translation(
                rerun::datatypes::Vec3D(jointStates[iJoint].transform.translation.data()))
#if !defined(RERUN_VERSION_GE) || !RERUN_VERSION_GE(0, 24, 0)
            .with_axis_length(rerun::components::AxisLength(10))
#endif
    );
  }
  safeLog(
      rec,
      streamName,
      rerun::LineStrips3D(lines).with_radii(0.2f).with_colors(kGrey).with_labels(labels));
}

void logMarkers(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    std::span<const Marker> markers) {
  const rerun::Color kPurple(120, 120, 255);
  std::vector<rerun::Position3D> points3d;
  std::vector<std::string> labels;
  points3d.reserve(markers.size());
  labels.reserve(markers.size());

  for (const auto& marker : markers) {
    if (marker.occluded) {
      continue;
    }
    points3d.push_back(toRerunPosition3D(marker.pos));
    labels.push_back(marker.name);
  }

  // TODO: make radius and color configurable
  safeLog(
      rec,
      streamName,
      rerun::Points3D(points3d).with_radii(0.5f).with_colors(kPurple).with_labels(labels));
}

void logLocators(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const LocatorList& locators,
    const LocatorState& locatorState) {
  if (locators.empty()) {
    return;
  }

  const rerun::Color kGreen(100, 255, 100);
  std::vector<std::string> labels;
  std::vector<rerun::Position3D> points3d;
  std::vector<rerun::Color> colors;
  points3d.reserve(locatorState.position.size());
  labels.reserve(locatorState.position.size());
  colors.reserve(locatorState.position.size());

  for (size_t i = 0; i < locatorState.position.size(); ++i) {
    const auto& locator = locatorState.position[i];
    points3d.push_back(toRerunPosition3D(locator));
    labels.push_back(locators[i].name);
    rerun::Color color{};
    if (locators.at(i).name.find("Floor_") != std::string::npos) {
      color = kGreen;
    } else {
      float intensity = 255.0f * locators[i].weight * 0.6f;
      color = rerun::Color(
          static_cast<uint8_t>(intensity),
          static_cast<uint8_t>(intensity),
          static_cast<uint8_t>(intensity * 0.5f));
    }
    colors.push_back(color);
  }

  // TODO: make radius and color configurable
  safeLog(
      rec,
      streamName,
      rerun::Points3D(points3d).with_radii(0.5f).with_colors(colors).with_labels(labels));
}

void logMarkerLocatorCorrespondence(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const std::map<std::string, size_t>& locatorLookup,
    const LocatorState& locatorState,
    std::span<const Marker> markers,
    const float kPositionErrorThreshold) {
  if (locatorLookup.empty()) {
    // No correspondence provided.
    return;
  }
  std::vector<std::string> labels;
  std::vector<std::vector<std::array<float, 3>>> lines;
  std::vector<rerun::components::Color> colors;
  std::vector<rerun::components::Radius> radius;
  lines.reserve(markers.size());
  labels.reserve(markers.size());
  colors.reserve(markers.size());
  radius.reserve(markers.size());

  const rerun::components::Color kGreenColor(50, 255, 128);
  const rerun::components::Color kRedColor(255, 100, 100);
  const rerun::components::Radius kDefaultRadius(0.1f);
  const rerun::components::Radius kLargeRadius(0.5f);

  for (const auto& marker : markers) {
    if (!marker.occluded && (locatorLookup.count(marker.name) != 0)) {
      const auto locatorIdx = locatorLookup.at(marker.name);
      const auto locator = locatorState.position.at(locatorIdx);

      lines.push_back({toStdArray3f(marker.pos), toStdArray3f(locator)});
      labels.push_back(marker.name);

      // TODO: expose marker error computation?
      const auto error = (marker.pos.cast<float>() - locator).squaredNorm();
      if (error > kPositionErrorThreshold) {
        colors.push_back(kRedColor);
        radius.push_back(kLargeRadius);
      } else {
        colors.push_back(kGreenColor);
        radius.push_back(kDefaultRadius);
      }
    }

    safeLog(
        rec,
        streamName,
        rerun::LineStrips3D(lines).with_labels(labels).with_colors(colors).with_radii(radius));
  }
}

void logModelParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::VectorXf& params) {
  // TODO: check names and params have the same size
  const size_t nParams = params.size();

  for (size_t iParam = 0; iParam < nParams; ++iParam) {
    if (names[iParam].find("root") != std::string::npos) {
      safeLog(rec, fmt::format("{}/{}", worldPrefix, names[iParam]), makeScalar(params[iParam]));
    } else {
      safeLog(rec, fmt::format("{}/{}", posePrefix, names[iParam]), makeScalar(params[iParam]));
    }
  }
}

void logJointParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::VectorXf& params) {
  // TODO: check names vs params size

  for (size_t iJoint = 0; iJoint < names.size(); ++iJoint) {
    for (size_t jParam = 0; jParam < kParametersPerJoint; ++jParam) {
      const std::string channelName = names[iJoint] + "_" + kJointParameterNames[jParam];
      const size_t paramIdx = iJoint * kParametersPerJoint + jParam;
      if (names[iJoint].find("world") != std::string::npos ||
          names[iJoint].find("root") != std::string::npos) {
        safeLog(rec, fmt::format("{}/{}", worldPrefix, channelName), makeScalar(params[paramIdx]));
      } else {
        safeLog(rec, fmt::format("{}/{}", posePrefix, channelName), makeScalar(params[paramIdx]));
      }
    }
  }
}

void logModelParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names) {
  for (const auto& name : names) {
    if (name.find("root") != std::string::npos) {
      safeLogStatic(rec, fmt::format("{}/{}", worldPrefix, name), makeSeriesLineWithName(name));
    } else {
      safeLogStatic(rec, fmt::format("{}/{}", posePrefix, name), makeSeriesLineWithName(name));
    }
  }
}

void logJointParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names) {
  for (const auto& name : names) {
    for (const auto& jointParameterName : kJointParameterNames) {
      const std::string channelName = name + "_" + jointParameterName;
      if (name.find("world") != std::string::npos || name.find("root") != std::string::npos) {
        safeLogStatic(
            rec,
            fmt::format("{}/{}", worldPrefix, channelName),
            makeSeriesLineWithName(channelName));
      } else {
        safeLogStatic(
            rec,
            fmt::format("{}/{}", posePrefix, channelName),
            makeSeriesLineWithName(channelName));
      }
    }
  }
}

void logModelParamsColumns(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::MatrixXf& allParams,
    std::span<const int64_t> frameIndices,
    std::span<const double> times) {
  const auto nFrames = static_cast<size_t>(allParams.cols());
  if (nFrames == 0) {
    return;
  }

  // Build time columns (reused for all parameters)
  auto frameIndexColumn = makeSequenceTimeColumn(
      "frame_index",
      rerun::Collection<int64_t>::borrow(frameIndices.data(), frameIndices.size()),
      rerun::SortingStatus::Sorted);
  auto timeColumn = makeDurationSecondsTimeColumn(
      "time",
      rerun::Collection<double>::borrow(times.data(), times.size()),
      rerun::SortingStatus::Sorted);
  const std::vector<rerun::TimeColumn> timeColumns = {
      std::move(frameIndexColumn), std::move(timeColumn)};

  const size_t nParams = std::min(static_cast<size_t>(allParams.rows()), names.size());
  for (size_t iParam = 0; iParam < nParams; ++iParam) {
    // Build scalar values for this parameter across all frames
    std::vector<double> scalarValues(nFrames);
    for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
      scalarValues[iFrame] = static_cast<double>(allParams(iParam, iFrame));
    }

    auto scalarColumns = makeScalarColumns(std::move(scalarValues));

    if (names[iParam].find("root") != std::string::npos) {
      rec.send_columns(
          fmt::format("{}/{}", worldPrefix, names[iParam]), timeColumns, scalarColumns);
    } else {
      rec.send_columns(fmt::format("{}/{}", posePrefix, names[iParam]), timeColumns, scalarColumns);
    }
  }
}

void logJointParamsColumns(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::MatrixXf& allJointParams,
    std::span<const int64_t> frameIndices,
    std::span<const double> times) {
  const auto nFrames = static_cast<size_t>(allJointParams.cols());
  if (nFrames == 0) {
    return;
  }

  // Build time columns (reused for all parameters)
  auto frameIndexColumn = makeSequenceTimeColumn(
      "frame_index",
      rerun::Collection<int64_t>::borrow(frameIndices.data(), frameIndices.size()),
      rerun::SortingStatus::Sorted);
  auto timeColumn = makeDurationSecondsTimeColumn(
      "time",
      rerun::Collection<double>::borrow(times.data(), times.size()),
      rerun::SortingStatus::Sorted);
  const std::vector<rerun::TimeColumn> timeColumns = {
      std::move(frameIndexColumn), std::move(timeColumn)};

  // Bounds check: ensure we don't read beyond allJointParams rows
  const size_t nJoints =
      std::min(names.size(), static_cast<size_t>(allJointParams.rows()) / kParametersPerJoint);
  for (size_t iJoint = 0; iJoint < nJoints; ++iJoint) {
    for (size_t jParam = 0; jParam < kParametersPerJoint; ++jParam) {
      const std::string channelName = names[iJoint] + "_" + kJointParameterNames[jParam];
      const size_t paramIdx = iJoint * kParametersPerJoint + jParam;

      // Build scalar values for this joint parameter across all frames
      std::vector<double> scalarValues(nFrames);
      for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
        scalarValues[iFrame] = static_cast<double>(allJointParams(paramIdx, iFrame));
      }

      auto scalarColumns = makeScalarColumns(std::move(scalarValues));

      if (names[iJoint].find("world") != std::string::npos ||
          names[iJoint].find("root") != std::string::npos) {
        rec.send_columns(
            fmt::format("{}/{}", worldPrefix, channelName), timeColumns, scalarColumns);
      } else {
        rec.send_columns(fmt::format("{}/{}", posePrefix, channelName), timeColumns, scalarColumns);
      }
    }
  }
}

void logBvh(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState) {
  // Compute collision state
  CollisionGeometryState collisionState;
  collisionState.update(skeletonState, collisionGeometry);

  // Compute AABBs
  const auto n = collisionGeometry.size();
  std::vector<axel::BoundingBoxf> aabbs(n);
  for (size_t i = 0; i < n; ++i) {
    auto& aabb = aabbs[i];
    aabb.id = static_cast<axel::Index>(i);
    updateAabb(aabb, collisionState, i);
  }

  // Construct BVH
  axel::Bvhf bvh;
  bvh.setBoundingBoxes(aabbs);
  const auto& bvs = bvh.getPrimitives();

  // Log BVH
  std::vector<rerun::Position3D> centers;
  std::vector<rerun::HalfSize3D> halfSizes;
  centers.reserve(bvs.size());
  halfSizes.reserve(bvs.size());
  for (const auto& bv : bvs) {
    centers.push_back(toRerunPosition3D(bv.center()));
    halfSizes.push_back(toRerunHalfSizes3D((bv.max() - bv.min()) / 2.0f));
  }

  // TODO: make radius configurable
  // TODO: use different colors by the depth of BVH nodes
  safeLog(
      rec,
      streamName,
      rerun::Boxes3D::from_centers_and_half_sizes(centers, halfSizes)
          .with_radii(0.1f)
          .with_colors(rerun::Color(64, 128, 64)));
}

void logCollisionGeometry(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState) {
  std::vector<rerun::Position3D> capsuleTranslations;
  std::vector<rerun::Quaternion> capsuleQuaternions;
  std::vector<float> capsuleLengths;
  std::vector<float> capsuleRadii;

  capsuleTranslations.reserve(collisionGeometry.size());
  capsuleQuaternions.reserve(collisionGeometry.size());
  capsuleLengths.reserve(collisionGeometry.size());
  capsuleRadii.reserve(collisionGeometry.size());

  for (const auto& cg : collisionGeometry) {
    if (cg.type != CollisionPrimitiveType::TaperedCapsule) {
      continue;
    }
    const Transform parentTransform = (cg.parent == kInvalidIndex)
        ? Transform()
        : skeletonState.jointState.at(cg.parent).transform;
    const Transform tf = parentTransform * cg.transformation;

    const Quaternionf& q = tf.rotation * Eigen::AngleAxisf(0.5f * pi(), Vector3f::UnitY());
    capsuleTranslations.push_back(toRerunPosition3D(tf.translation));
    capsuleQuaternions.emplace_back(rerun::Quaternion::from_xyzw(q.x(), q.y(), q.z(), q.w()));
    // Match CollisionGeometryStateT::update: length runs along the composed-transform
    // X axis (scaled by the full transform scale), radius is scaled by the parent joint
    // scale only.
    capsuleLengths.emplace_back(cg.length * tf.scale);
    // TODO: Rerun doesn't support capsules with different radii (i.e. tapered capsule) yet
    capsuleRadii.emplace_back(cg.radius.maxCoeff() * parentTransform.scale);
  }

  if (!capsuleLengths.empty()) {
    safeLog(
        rec,
        streamName + "/capsules",
        rerun::Capsules3D::from_lengths_and_radii(capsuleLengths, capsuleRadii)
            .with_translations(capsuleTranslations)
            .with_quaternions(capsuleQuaternions)
            .with_colors(rerun::Color(128, 64, 64))
            .with_fill_mode(rerun::FillMode::Solid));
  }

  const auto ellipsoidLogData =
      detail::makeCollisionEllipsoidLogData(collisionGeometry, skeletonState);
  if (!ellipsoidLogData.halfSizes.empty()) {
    std::vector<rerun::Position3D> ellipsoidCenters;
    std::vector<rerun::HalfSize3D> ellipsoidHalfSizes;
    std::vector<rerun::Quaternion> ellipsoidQuaternions;
    ellipsoidCenters.reserve(ellipsoidLogData.centers.size());
    ellipsoidHalfSizes.reserve(ellipsoidLogData.halfSizes.size());
    ellipsoidQuaternions.reserve(ellipsoidLogData.quaternions.size());

    for (size_t i = 0; i < ellipsoidLogData.halfSizes.size(); ++i) {
      const Quaternionf& q = ellipsoidLogData.quaternions[i];
      ellipsoidCenters.push_back(toRerunPosition3D(ellipsoidLogData.centers[i]));
      ellipsoidHalfSizes.push_back(toRerunHalfSizes3D(ellipsoidLogData.halfSizes[i]));
      ellipsoidQuaternions.emplace_back(rerun::Quaternion::from_xyzw(q.x(), q.y(), q.z(), q.w()));
    }

    safeLog(
        rec,
        streamName + "/ellipsoids",
        rerun::Ellipsoids3D::from_centers_and_half_sizes(ellipsoidCenters, ellipsoidHalfSizes)
            .with_quaternions(ellipsoidQuaternions)
            .with_colors(rerun::Color(128, 64, 64))
            .with_fill_mode(rerun::FillMode::Solid));
  }

  const auto boxLogData = detail::makeCollisionBoxLogData(collisionGeometry, skeletonState);
  if (!boxLogData.halfSizes.empty()) {
    std::vector<rerun::Position3D> boxCenters;
    std::vector<rerun::HalfSize3D> boxHalfSizes;
    std::vector<rerun::Quaternion> boxQuaternions;
    boxCenters.reserve(boxLogData.centers.size());
    boxHalfSizes.reserve(boxLogData.halfSizes.size());
    boxQuaternions.reserve(boxLogData.quaternions.size());

    for (size_t i = 0; i < boxLogData.halfSizes.size(); ++i) {
      const Quaternionf& q = boxLogData.quaternions[i];
      boxCenters.push_back(toRerunPosition3D(boxLogData.centers[i]));
      boxHalfSizes.push_back(toRerunHalfSizes3D(boxLogData.halfSizes[i]));
      boxQuaternions.emplace_back(rerun::Quaternion::from_xyzw(q.x(), q.y(), q.z(), q.w()));
    }

    safeLog(
        rec,
        streamName + "/boxes",
        rerun::Boxes3D::from_centers_and_half_sizes(boxCenters, boxHalfSizes)
            .with_quaternions(boxQuaternions)
            .with_colors(rerun::Color(128, 64, 64))
            .with_fill_mode(rerun::FillMode::Solid));
  }

  logBvh(rec, streamName + "/bvh", collisionGeometry, skeletonState);
}

void logCharacter(
    const rerun::RecordingStream& rec,
    const std::string& charStreamName,
    const Character& character,
    const CharacterState& characterState,
    const rerun::Color& color) {
  if (characterState.meshState != nullptr) {
    logMesh(rec, charStreamName + "/mesh", *characterState.meshState, color);
  }
  if (!character.locators.empty()) {
    logLocators(rec, charStreamName + "/locators", character.locators, characterState.locatorState);
  }

  logJoints(
      rec, charStreamName + "/joints", character.skeleton, characterState.skeletonState.jointState);

  if (const auto& collision = character.collision) {
    logCollisionGeometry(
        rec, charStreamName + "/collision_geometry", *collision, characterState.skeletonState);
  }
}

} // namespace momentum
