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

  rec.log(streamName, rerunMesh);
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
    rec.log(
        streamName + "/" + names[iJoint],
        rerun::Transform3D()
            .with_mat3x3(
                rerun::datatypes::Mat3x3(jointStates[iJoint].transform.toRotationMatrix().data()))
            .with_translation(
                rerun::datatypes::Vec3D(jointStates[iJoint].transform.translation.data()))
            .with_axis_length(rerun::components::AxisLength(10)));
  }
  rec.log(
      streamName,
      rerun::LineStrips3D(lines).with_radii(0.2f).with_colors(kGrey).with_labels(labels));
}

void logMarkers(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    gsl::span<const Marker> markers) {
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
  rec.log(
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
      float intensity = 255.0 * locators[i].weight * 0.6;
      color = rerun::Color(int8_t(intensity), int8_t(intensity), int8_t(intensity * 0.5));
    }
    colors.push_back(color);
  }

  // TODO: make radius and color configurable
  rec.log(
      streamName,
      rerun::Points3D(points3d).with_radii(0.5f).with_colors(colors).with_labels(labels));
}

void logMarkerLocatorCorrespondence(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const std::map<std::string, size_t>& locatorLookup,
    const LocatorState& locatorState,
    gsl::span<const Marker> markers,
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

    rec.log(
        streamName,
        rerun::LineStrips3D(lines).with_labels(labels).with_colors(colors).with_radii(radius));
  }
}

void logModelParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names,
    const Eigen::VectorXf& params) {
  // TODO: check names and params have the same size
  const size_t nParams = params.size();

  for (size_t iParam = 0; iParam < nParams; ++iParam) {
    if (names[iParam].find("root") != std::string::npos) {
      rec.log(fmt::format("{}/{}", worldPrefix, names[iParam]), rerun::Scalar(params[iParam]));
    } else {
      rec.log(fmt::format("{}/{}", posePrefix, names[iParam]), rerun::Scalar(params[iParam]));
    }
  }
}

void logJointParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names,
    const Eigen::VectorXf& params) {
  // TODO: check names vs params size

  for (size_t iJoint = 0; iJoint < names.size(); ++iJoint) {
    for (size_t jParam = 0; jParam < kParametersPerJoint; ++jParam) {
      const std::string channelName = names[iJoint] + "_" + kJointParameterNames[jParam];
      const size_t paramIdx = iJoint * kParametersPerJoint + jParam;
      if (names[iJoint].find("world") != std::string::npos ||
          names[iJoint].find("root") != std::string::npos) {
        rec.log(fmt::format("{}/{}", worldPrefix, channelName), rerun::Scalar(params[paramIdx]));
      } else {
        rec.log(fmt::format("{}/{}", posePrefix, channelName), rerun::Scalar(params[paramIdx]));
      }
    }
  }
}

void logModelParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names) {
  for (const auto& name : names) {
    if (name.find("root") != std::string::npos) {
      rec.log_static(fmt::format("{}/{}", worldPrefix, name), rerun::SeriesLine().with_name(name));
    } else {
      rec.log_static(fmt::format("{}/{}", posePrefix, name), rerun::SeriesLine().with_name(name));
    }
  }
}

void logJointParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names) {
  for (const auto& name : names) {
    for (const auto& jointParameterName : kJointParameterNames) {
      const std::string channelName = name + "_" + jointParameterName;
      if (name.find("world") != std::string::npos || name.find("root") != std::string::npos) {
        rec.log_static(
            fmt::format("{}/{}", worldPrefix, channelName),
            rerun::SeriesLine().with_name(channelName));
      } else {
        rec.log_static(
            fmt::format("{}/{}", posePrefix, channelName),
            rerun::SeriesLine().with_name(channelName));
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
    aabb.id = i;
    updateAabb(
        aabb, collisionState.origin[i], collisionState.direction[i], collisionState.radius[i]);
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
  rec.log(
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
  // TODO: update collision geometry representation from box to capsule

  std::vector<rerun::Position3D> translations;
  std::vector<rerun::Quaternion> quaternions;
  std::vector<float> lengths;
  std::vector<float> radii;

  translations.reserve(collisionGeometry.size());
  quaternions.reserve(collisionGeometry.size());
  lengths.reserve(collisionGeometry.size());
  radii.reserve(collisionGeometry.size());

  for (const auto& cg : collisionGeometry) {
    const auto& js = skeletonState.jointState.at(cg.parent);

    const Transform tf = js.transform * cg.transformation;
    const Quaternionf& q = tf.rotation * Eigen::AngleAxisf(0.5f * pi(), Vector3f::UnitY());

    translations.push_back(toRerunPosition3D(tf.translation));
    quaternions.emplace_back(rerun::Quaternion::from_xyzw(q.x(), q.y(), q.z(), q.w()));
    lengths.emplace_back(cg.length);
    // TODO: Rerun doesn't support capsules with different radii (i.e. tapered capsule) yet
    radii.emplace_back(cg.radius.maxCoeff());
  }

  // TODO: make radius and color configurable
  rec.log(
      streamName,
      rerun::Capsules3D::from_lengths_and_radii(lengths, radii)
          .with_translations(translations)
          .with_quaternions(quaternions)
          .with_colors(rerun::Color(128, 64, 64)));
  // TODO: Switch to wireframe once available in Rerun

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
