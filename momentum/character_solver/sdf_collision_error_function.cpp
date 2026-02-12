/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/sdf_collision_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/character_solver/skinning_weight_iterator.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include <axel/BoundingBox.h>

#include <algorithm>
#include <map>
#include <unordered_set>

namespace momentum {

namespace {

/// Finds the common ancestor joint between a skinned vertex and an SDF collider parent.
///
/// This function determines the joint where both the mesh skinning hierarchy and the
/// SDF collider hierarchy meet. Gradient/Jacobian contributions above this ancestor
/// will cancel out, so we can stop processing at this point.
///
/// @param skinWeights The character's skin weights
/// @param skeleton The character skeleton
/// @param vertexIndex The index of the vertex being processed
/// @param sdfParent The parent joint index of the SDF collider
/// @return The common ancestor joint index, or kInvalidIndex if none exists
inline size_t findCommonAncestorForVertex(
    const SkinWeights& skinWeights,
    const Skeleton& skeleton,
    size_t vertexIndex,
    size_t sdfParent) {
  if (sdfParent == kInvalidIndex) {
    return kInvalidIndex;
  }

  // Find the minimum (root-most) skinned joint for this vertex.
  // This is where all skinning weights will have merged to 1.0 in the SkinningWeightIterator.
  size_t minSkinnedJoint = kInvalidIndex;
  for (uint32_t k = 0; k < kMaxSkinJoints; ++k) {
    const auto w = skinWeights.weight(vertexIndex, k);
    if (w > 0) {
      const auto joint = skinWeights.index(vertexIndex, k);
      if (minSkinnedJoint == kInvalidIndex || joint < minSkinnedJoint) {
        minSkinnedJoint = joint;
      }
    }
  }

  if (minSkinnedJoint == kInvalidIndex) {
    return kInvalidIndex;
  }

  return skeleton.commonAncestor(minSkinnedJoint, sdfParent);
}

} // namespace

template <typename T>
SDFCollisionErrorFunctionT<T>::SDFCollisionErrorFunctionT(
    const Character& character,
    const SDFCollisionGeometry& sdfColliders,
    const std::vector<int>& participatingVertices,
    const std::vector<T>& vertexWeights,
    bool filterRestPoseIntersections,
    uint8_t maxCollisionsPerVertex)
    : SkeletonErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character),
      sdfColliders_(sdfColliders),
      maxCollisionsPerVertex_(maxCollisionsPerVertex) {
  MT_CHECK_NOTNULL(character_.mesh, "Character must have a mesh for SDF collision detection");
  MT_CHECK_NOTNULL(
      character_.skinWeights, "Character must have skin weights for SDF collision detection");
  MT_CHECK(!sdfColliders_.empty(), "Must provide at least one SDF collider");
  MT_CHECK(maxCollisionsPerVertex_ > 0, "maxCollisionsPerVertex must be positive");
  MT_CHECK(maxCollisionsPerVertex_ <= 255, "maxCollisionsPerVertex must be <= 255");

  // Organize vertices by bone for efficient processing
  organizeVerticesByBone(participatingVertices, vertexWeights);

  // Initialize bone-collider pairs and test rest pose intersections
  initializeBoneCollisionPairs(filterRestPoseIntersections);
}

template <typename T>
void SDFCollisionErrorFunctionT<T>::organizeVerticesByBone(
    const std::vector<int>& participatingVertices,
    const std::vector<T>& weights) {
  // Determine which vertices to include
  std::vector<int> vertices;
  if (participatingVertices.empty()) {
    // Use all vertices
    vertices.reserve(character_.mesh->vertices.size());
    for (int i = 0; i < static_cast<int>(character_.mesh->vertices.size()); ++i) {
      vertices.push_back(i);
    }
  } else {
    vertices = participatingVertices;
  }

  // Validate vertex indices
  for (int vertexIndex : vertices) {
    MT_CHECK(
        vertexIndex >= 0 && vertexIndex < static_cast<int>(character_.mesh->vertices.size()),
        "Invalid vertex index: {} (mesh has {} vertices)",
        vertexIndex,
        character_.mesh->vertices.size());
  }

  // Determine weights
  std::vector<T> vertexWeights;
  if (weights.empty()) {
    // Use uniform weights
    vertexWeights.assign(vertices.size(), T(1));
  } else {
    MT_CHECK(
        weights.size() == vertices.size(),
        "Weight count ({}) must match vertex count ({})",
        weights.size(),
        vertices.size());
    vertexWeights = weights;
  }

  // Create vertex info with primary bone indices
  struct VertexInfo {
    size_t vertexIndex;
    T weight;
    size_t primaryBone;
  };

  std::vector<VertexInfo> vertexInfos;
  vertexInfos.reserve(vertices.size());

  MT_CHECK_NOTNULL(character_.skinWeights);
  const auto& skinWeights = *character_.skinWeights;

  for (size_t i = 0; i < vertices.size(); ++i) {
    const int vertexIndex = vertices[i];
    const T weight = vertexWeights[i];
    const size_t primaryBone = skinWeights.index(vertexIndex, 0);
    MT_CHECK(primaryBone < character_.skeleton.joints.size());

    vertexInfos.push_back({static_cast<size_t>(vertexIndex), weight, primaryBone});
  }

  // Sort by primary bone index for contiguous organization
  std::sort(vertexInfos.begin(), vertexInfos.end(), [](const VertexInfo& a, const VertexInfo& b) {
    return a.primaryBone < b.primaryBone;
  });

  // Initialize bone vertex ranges for all bones in skeleton
  boneVertexRanges_.resize(character_.skeleton.joints.size());

  // Populate contiguous arrays and build ranges
  activeVertexIndices_.reserve(vertices.size());
  vertexWeights_.reserve(vertices.size());

  size_t currentBone = 0;

  for (size_t i = 0; i < vertexInfos.size(); ++i) {
    const auto& info = vertexInfos[i];

    // Check if we've moved to a new bone
    while (currentBone < info.primaryBone) {
      boneVertexRanges_[currentBone].end = i;
      ++currentBone;
      boneVertexRanges_[currentBone].start = i;
    }

    // Add vertex to contiguous arrays
    activeVertexIndices_.push_back(info.vertexIndex);
    vertexWeights_.push_back(info.weight);
  }

  // Finalize the last bone's range
  boneVertexRanges_[currentBone].end = activeVertexIndices_.size();
  ++currentBone;

  // Finalize any remaining bone ranges.
  while (currentBone < boneVertexRanges_.size()) {
    boneVertexRanges_[currentBone++] =
        BoneVertexRange{activeVertexIndices_.size(), activeVertexIndices_.size()};
  }
}

template <typename T>
void SDFCollisionErrorFunctionT<T>::initializeBoneCollisionPairs(bool filterRestPoseIntersections) {
  // Find all unique bone indices that have vertices
  std::unordered_set<size_t> activeBones;
  for (size_t boneIndex = 0; boneIndex < boneVertexRanges_.size(); ++boneIndex) {
    if (!boneVertexRanges_[boneIndex].empty()) {
      activeBones.insert(boneIndex);
    }
  }

  // Create zero skeleton state for rest pose (create once, reuse)
  const SkeletonStateT<T> restState(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);

  // Create bone-collider pairs for all combinations
  boneCollisionPairs_.clear();
  boneCollisionPairs_.reserve(activeBones.size() * sdfColliders_.size());

  for (size_t boneIndex : activeBones) {
    if (boneVertexRanges_[boneIndex].empty()) {
      continue; // No vertices for this bone`
    }

    for (size_t sdfIndex = 0; sdfIndex < sdfColliders_.size(); ++sdfIndex) {
      const auto& collider = sdfColliders_[sdfIndex];
      // Filter out parts of the body that are intersecting in the rest pose.
      if (collider.parent != kInvalidIndex && filterRestPoseIntersections &&
          testRestPoseIntersection(boneIndex, sdfIndex, restState)) {
        continue;
      }

      boneCollisionPairs_.emplace_back(BoneCollisionPairT<T>{boneIndex, sdfIndex});
    }
  }
}

template <typename T>
bool SDFCollisionErrorFunctionT<T>::testRestPoseIntersection(
    size_t boneIndex,
    size_t sdfColliderIndex,
    const SkeletonStateT<T>& restState) const {
  const auto& range = boneVertexRanges_[boneIndex];
  if (range.empty()) {
    return false;
  }

  // Get SDF collider reference
  const auto& sdfCollider = sdfColliders_[sdfColliderIndex];

  // Check if SDF is valid
  if (!sdfCollider.isValid()) {
    return false; // No collision with invalid SDF
  }

  // Compute the world-to-SDF transform once for all vertices
  const auto worldToSdfTransform = computeWorldToSDFTransform(sdfColliderIndex, restState);

  // Test vertices for this bone using the pre-computed transform
  for (size_t i = range.start; i < range.end; ++i) {
    const size_t vertexIndex = activeVertexIndices_[i];

    // Get vertex position in rest pose (world space)
    const auto& restPos = character_.mesh->vertices[vertexIndex].template cast<T>();

    // Sample SDF at rest position using pre-computed transform
    const T distance = sampleSDFDistanceWithTransform(restPos, sdfCollider, worldToSdfTransform);

    // If any vertex penetrates (distance < 0), this pair should be filtered out
    if (distance < T(0)) {
      return true;
    }
  }

  return false;
}

template <typename T>
double SDFCollisionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);

  // Update which collision pairs are currently active (broad-phase)
  computeActivePairs(meshState, state);

  double totalError = 0.0;

  collisionCounts_.assign(activeVertexIndices_.size(), 0);

  // Process each active collision pair
  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    // Skip inactive pairs (no bounding box intersection)
    if (!activePairs_[pairIndex]) {
      continue;
    }

    const auto& pair = boneCollisionPairs_[pairIndex];
    const size_t boneIndex = pair.boneIndex;
    const size_t sdfColliderIndex = pair.sdfColliderIndex;
    const auto& range = boneVertexRanges_[boneIndex];

    // Skip if bone has no vertices
    if (range.empty()) {
      continue;
    }

    // Get SDF collider reference
    const auto& sdfCollider = sdfColliders_[sdfColliderIndex];

    // Skip invalid SDF colliders
    if (!sdfCollider.isValid()) {
      continue;
    }

    // Compute world-to-SDF transform once for all vertices in this bone-SDF pair
    const auto worldToSdfTransform = computeWorldToSDFTransform(sdfColliderIndex, state);

    // Test all vertices for this bone against the SDF collider
    for (size_t i = range.start; i < range.end; ++i) {
      if (collisionCounts_[i] >= maxCollisionsPerVertex_) {
        continue; // Skip this vertex if it has already collided with maxCollisionsPerVertex_ pairs
      }

      const size_t vertexIndex = activeVertexIndices_[i];
      const T vertexWeight = vertexWeights_[i];

      // Get world position from posed mesh
      const auto& worldPos = meshState.posedMesh_->vertices[vertexIndex];

      // Sample SDF distance at vertex position using pre-computed transform
      const T distance = sampleSDFDistanceWithTransform(worldPos, sdfCollider, worldToSdfTransform);

      // Apply collision penalty for penetrating vertices (distance < 0)
      if (distance < T(0)) {
        collisionCounts_[i]++;
        const T penetrationDepth = -distance; // Make positive

        // Accumulate squared error: error += vertexWeight * (penetration^2 * kSDFCollisionWeight *
        // this->weight_)
        totalError += static_cast<double>(
            vertexWeight * penetrationDepth * penetrationDepth * kSDFCollisionWeight *
            this->weight_);
      }
    }
  }

  return totalError;
}

template <typename T>
void SDFCollisionErrorFunctionT<T>::accumulateJointGradient(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  for (size_t d = 0; d < 3; d++) {
    if (this->activeJointParams_[paramIndex + d]) {
      gradient_jointParams_to_modelParams(
          direction.dot(jointState.getTranslationDerivative(d)),
          paramIndex + d,
          this->parameterTransform_,
          gradient);
    }
    if (this->activeJointParams_[paramIndex + 3 + d]) {
      gradient_jointParams_to_modelParams(
          direction.dot(jointState.getRotationDerivative(d, posd)),
          paramIndex + 3 + d,
          this->parameterTransform_,
          gradient);
    }
  }
  if (this->activeJointParams_[paramIndex + 6]) {
    gradient_jointParams_to_modelParams(
        direction.dot(jointState.getScaleDerivative(posd)),
        paramIndex + 6,
        this->parameterTransform_,
        gradient);
  }
}

template <typename T>
void SDFCollisionErrorFunctionT<T>::accumulateJointJacobian(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const {
  for (size_t d = 0; d < 3; d++) {
    if (this->activeJointParams_[paramIndex + d]) {
      jacobian_jointParams_to_modelParams<T>(
          direction.dot(jointState.getTranslationDerivative(d)),
          paramIndex + d,
          this->parameterTransform_,
          jacobianRow);
    }
    if (this->activeJointParams_[paramIndex + 3 + d]) {
      jacobian_jointParams_to_modelParams<T>(
          direction.dot(jointState.getRotationDerivative(d, posd)),
          paramIndex + 3 + d,
          this->parameterTransform_,
          jacobianRow);
    }
  }
  if (this->activeJointParams_[paramIndex + 6]) {
    jacobian_jointParams_to_modelParams<T>(
        direction.dot(jointState.getScaleDerivative(posd)),
        paramIndex + 6,
        this->parameterTransform_,
        jacobianRow);
  }
}

template <typename T>
void SDFCollisionErrorFunctionT<T>::accumulateColliderHierarchyGradient(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  size_t jointIndex = colliderParent;
  while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = sdfSurfacePoint - jointState.translation();

    accumulateJointGradient(jointState, paramIndex, posd, direction, gradient);

    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }
}

template <typename T>
void SDFCollisionErrorFunctionT<T>::accumulateColliderHierarchyJacobian(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const {
  size_t jointIndex = colliderParent;
  while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = sdfSurfacePoint - jointState.translation();

    accumulateJointJacobian(jointState, paramIndex, posd, direction, jacobianRow);

    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }
}

template <typename T>
double SDFCollisionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);

  // Update which collision pairs are currently active (broad-phase)
  computeActivePairs(meshState, state);

  double totalError = 0.0;

  collisionCounts_.assign(activeVertexIndices_.size(), 0);

  // Process each active collision pair
  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    // Skip inactive pairs (no bounding box intersection)
    if (!activePairs_[pairIndex]) {
      continue;
    }

    const auto& pair = boneCollisionPairs_[pairIndex];
    const size_t boneIndex = pair.boneIndex;
    const size_t sdfColliderIndex = pair.sdfColliderIndex;
    const auto& range = boneVertexRanges_[boneIndex];

    // Skip if bone has no vertices (shouldn't happen, but safety check)
    if (range.empty()) {
      continue;
    }

    // Get SDF collider reference
    const auto& sdfCollider = sdfColliders_[sdfColliderIndex];

    // Skip invalid SDF colliders
    if (!sdfCollider.isValid()) {
      continue;
    }

    // Compute world-to-SDF transform once for all vertices in this bone-SDF pair
    const auto worldToSdfTransform = computeWorldToSDFTransform(sdfColliderIndex, state);

    // Test all vertices for this bone against the SDF collider
    for (size_t i = range.start; i < range.end; ++i) {
      if (collisionCounts_[i] >= maxCollisionsPerVertex_) {
        continue; // Skip this vertex if it has already collided with maxCollisionsPerVertex_ pairs
      }

      const size_t vertexIndex = activeVertexIndices_[i];
      const T vertexWeight = vertexWeights_[i];

      // Get world position from posed mesh
      const auto& worldPos = meshState.posedMesh_->vertices[vertexIndex];

      // Sample SDF distance and gradient at vertex position using pre-computed transform
      const auto [distance, sdfGradient] =
          sampleSDFDistanceAndGradientWithTransform(worldPos, sdfCollider, worldToSdfTransform);

      // Apply collision penalty for penetrating vertices (distance < 0)
      if (distance < T(0)) {
        const T penetrationDepth = -distance; // Make positive

        // Accumulate squared error: error += vertexWeight * (penetration^2 * kSDFCollisionWeight *
        // this->weight_)
        totalError += static_cast<double>(
            vertexWeight * penetrationDepth * penetrationDepth * kSDFCollisionWeight *
            this->weight_);
        collisionCounts_[i]++;

        // Compute gradient contribution for this vertex
        // Error function: e = vertexWeight * (penetration^2 * kSDFCollisionWeight * this->weight_)
        // de/d(penetration) = vertexWeight * 2 * penetration * kSDFCollisionWeight * this->weight_
        // Using same pattern as other vertex error functions: factor out weights into wgt
        const T wgt = vertexWeight * 2.0f * kSDFCollisionWeight * this->weight_;
        const Eigen::Vector3<T> diff =
            -penetrationDepth * sdfGradient; // Negative because penetration = -distance

        // Compute the closest point on the SDF surface. This is used as the lever arm
        // for rotation derivatives on the collider side. The SDF gradient points outward
        // from the surface, so moving from the penetrating vertex by penetrationDepth
        // in the gradient direction gives the surface point.
        const Eigen::Vector3<T> sdfSurfacePoint = worldPos + penetrationDepth * sdfGradient;

        // Find the common ancestor between the vertex's skinning hierarchy and the SDF parent.
        // Contributions above this ancestor will cancel out.
        const size_t commonAncestor = findCommonAncestorForVertex(
            *character_.skinWeights, this->skeleton_, vertexIndex, sdfCollider.parent);

        // -----------------------------------
        //  Process mesh vertex (skinning weights)
        // -----------------------------------
        // SkinningWeightIteratorT walks up the tree from leaf-most to root-most
        // skinned bones, accumulating weights along the way. Each call to next()
        // returns the next joint in the walk with accumulated weight and position.
        //
        // Stop at the common ancestor since contributions above it will cancel.
        SkinningWeightIteratorT<T> skinningIter(
            character_, *meshState.restMesh_, state, vertexIndex);
        while (!skinningIter.finished()) {
          const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();

          // Stop at common ancestor
          if (jointIndex == commonAncestor) {
            break;
          }

          // Skip if bone weight is too small
          if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
            continue;
          }

          // Check for valid joint index
          MT_CHECK(jointIndex < this->skeleton_.joints.size());

          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Eigen::Vector3<T> posd = transformedVertex - jointState.translation();

          accumulateJointGradient(jointState, paramIndex, posd, boneWeight * wgt * diff, gradient);
        }

        // -----------------------------------
        //  Process collider hierarchy (walk up from collider parent)
        // -----------------------------------
        if (sdfCollider.parent != kInvalidIndex) {
          accumulateColliderHierarchyGradient(
              state, sdfCollider.parent, commonAncestor, sdfSurfacePoint, -wgt * diff, gradient);
        }
      }
    }
  }

  return totalError;
}

template <typename T>
double SDFCollisionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()),
      "Jacobian column count mismatch");

  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);

  // Update which collision pairs are currently active (broad-phase)
  computeActivePairs(meshState, state);

  double totalError = 0.0;
  int currentRow = 0;

  collisionCounts_.assign(activeVertexIndices_.size(), 0);

  // Process each active collision pair
  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    // Skip inactive pairs (no bounding box intersection)
    if (!activePairs_[pairIndex]) {
      continue;
    }

    const auto& pair = boneCollisionPairs_[pairIndex];
    const size_t boneIndex = pair.boneIndex;
    const size_t sdfColliderIndex = pair.sdfColliderIndex;
    const auto& range = boneVertexRanges_[boneIndex];

    // Skip if bone has no vertices (shouldn't happen, but safety check)
    if (range.empty()) {
      continue;
    }

    // Get SDF collider reference
    const auto& sdfCollider = sdfColliders_[sdfColliderIndex];

    // Skip invalid SDF colliders
    if (!sdfCollider.isValid()) {
      continue;
    }

    // Compute world-to-SDF transform once for all vertices in this bone-SDF pair
    const auto worldToSdfTransform = computeWorldToSDFTransform(sdfColliderIndex, state);

    // Test all vertices for this bone against the SDF collider
    for (size_t i = range.start; i < range.end; ++i) {
      if (collisionCounts_[i] >= maxCollisionsPerVertex_) {
        continue; // Skip this vertex if it has already collided with maxCollisionsPerVertex_ pairs
      }

      const size_t vertexIndex = activeVertexIndices_[i];
      const T vertexWeight = vertexWeights_[i];

      // Get world position from posed mesh
      const auto& worldPos = meshState.posedMesh_->vertices[vertexIndex];

      // Sample SDF distance and gradient at vertex position using pre-computed transform
      const auto [distance, sdfGradient] =
          sampleSDFDistanceAndGradientWithTransform(worldPos, sdfCollider, worldToSdfTransform);

      // Apply collision penalty for penetrating vertices (distance < 0)
      if (distance < T(0)) {
        const T penetrationDepth = -distance; // Make positive

        // Check if we have enough rows in jacobian and residual
        MT_CHECK(
            currentRow < jacobian.rows() && currentRow < residual.rows(),
            "Insufficient Jacobian/residual rows: need {}, have jacobian: {}, residual: {}",
            currentRow + 1,
            jacobian.rows(),
            residual.rows());

        // Compute residual: sqrt(weight * kSDFCollisionWeight) * penetrationDepth
        const T wgt = std::sqrt(vertexWeight * kSDFCollisionWeight * this->weight_);
        residual[currentRow] = wgt * penetrationDepth;

        // Accumulate squared error for return value
        totalError += static_cast<double>(
            vertexWeight * penetrationDepth * penetrationDepth * kSDFCollisionWeight *
            this->weight_);

        ++collisionCounts_[i];

        // Zero out the jacobian row for this constraint
        jacobian.row(currentRow).setZero();

        // Compute jacobian row for this vertex constraint
        // Residual function: r = sqrt(weight * kSDFCollisionWeight) * penetrationDepth
        // dr/d(penetration) = sqrt(weight * kSDFCollisionWeight)
        // dr/d(distance) = -dr/d(penetration) (since penetration = -distance)
        // dr/d(worldPos) = dr/d(distance) * d(distance)/d(worldPos) = dr/d(distance) * sdfGradient
        const T jacobianMagnitude = -wgt; // Negative because penetration = -distance
        const Eigen::Vector3<T> worldJacobianContribution = jacobianMagnitude * sdfGradient;

        // Compute the closest point on the SDF surface. This is used as the lever arm
        // for rotation derivatives on the collider side. The SDF gradient points outward
        // from the surface, so moving from the penetrating vertex by penetrationDepth
        // in the gradient direction gives the surface point.
        const Eigen::Vector3<T> sdfSurfacePoint = worldPos + penetrationDepth * sdfGradient;

        // Find the common ancestor between the vertex's skinning hierarchy and the SDF parent.
        // Contributions above this ancestor will cancel out.
        const size_t commonAncestor = findCommonAncestorForVertex(
            *character_.skinWeights, this->skeleton_, vertexIndex, sdfCollider.parent);

        // -----------------------------------
        //  Process mesh vertex (skinning weights)
        // -----------------------------------
        // SkinningWeightIteratorT walks up the tree from leaf-most to root-most
        // skinned bones, accumulating weights along the way. Each call to next()
        // returns the next joint in the walk with accumulated weight and position.
        //
        // Stop at the common ancestor since contributions above it will cancel.
        SkinningWeightIteratorT<T> skinningIter(
            character_, *meshState.restMesh_, state, vertexIndex);
        while (!skinningIter.finished()) {
          const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();

          // Stop at common ancestor
          if (jointIndex == commonAncestor) {
            break;
          }

          // Skip if bone weight is too small
          if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
            continue;
          }

          // Check for valid joint index
          MT_CHECK(jointIndex < this->skeleton_.joints.size());

          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Eigen::Vector3<T> posd = transformedVertex - jointState.translation();

          accumulateJointJacobian(
              jointState,
              paramIndex,
              posd,
              boneWeight * worldJacobianContribution,
              jacobian.middleRows(currentRow, 1));
        }

        // -----------------------------------
        //  Process collider hierarchy (walk up from collider parent)
        // -----------------------------------
        if (sdfCollider.parent != kInvalidIndex) {
          accumulateColliderHierarchyJacobian(
              state,
              sdfCollider.parent,
              commonAncestor,
              sdfSurfacePoint,
              -worldJacobianContribution,
              jacobian.middleRows(currentRow, 1));
        }

        // Move to next row for the next constraint
        currentRow++;
      }
    }
  }

  usedRows = currentRow;
  return totalError;
}

template <typename T>
size_t SDFCollisionErrorFunctionT<T>::getJacobianSize() const {
  // Conservative estimate: all active vertices could potentially penetrate
  return static_cast<size_t>(maxCollisionsPerVertex_) * activeVertexIndices_.size();
}

template <typename T>
void SDFCollisionErrorFunctionT<T>::computeActivePairs(
    const MeshStateT<T>& meshState,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  // Reset all pairs to inactive
  activePairs_.assign(boneCollisionPairs_.size(), false);

  // Pre-compute bone bounding boxes once to avoid redundant computation
  // when the same bone appears in multiple collision pairs
  boneBBoxCache_.resize(boneVertexRanges_.size());
  boneBBoxValid_.assign(boneVertexRanges_.size(), false);
  for (size_t boneIndex = 0; boneIndex < boneVertexRanges_.size(); ++boneIndex) {
    if (!boneVertexRanges_[boneIndex].empty()) {
      boneBBoxCache_[boneIndex] = computeBoneBoundingBox(boneIndex, *meshState.posedMesh_);
      boneBBoxValid_[boneIndex] = true;
    }
  }

  // Pre-compute SDF bounding boxes once
  sdfBBoxCache_.resize(sdfColliders_.size());
  for (size_t sdfIndex = 0; sdfIndex < sdfColliders_.size(); ++sdfIndex) {
    sdfBBoxCache_[sdfIndex] = computeSDFWorldBoundingBox(sdfIndex, state);
  }

  // Test each bone-collider pair for collision using cached bounding boxes
  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    const auto& pair = boneCollisionPairs_[pairIndex];

    const size_t boneIndex = pair.boneIndex;
    const size_t sdfColliderIndex = pair.sdfColliderIndex;

    // Skip if this bone has no vertices
    if (!boneBBoxValid_[boneIndex]) {
      continue;
    }

    // Broad-phase collision detection using cached bounding boxes
    const bool bboxIntersects =
        boneBBoxCache_[boneIndex].intersects(sdfBBoxCache_[sdfColliderIndex]);

    // Mark pair as active if bounding boxes intersect
    activePairs_[pairIndex] = bboxIntersects;
  }
}

template <typename T>
T SDFCollisionErrorFunctionT<T>::sampleSDFDistance(
    const Eigen::Vector3<T>& worldPos,
    size_t sdfColliderIndex,
    const SkeletonStateT<T>& state) const {
  MT_CHECK(
      sdfColliderIndex < sdfColliders_.size(), "Invalid SDF collider index: {}", sdfColliderIndex);

  const auto& sdfCollider = sdfColliders_[sdfColliderIndex];

  // Check if SDF is valid
  if (!sdfCollider.isValid()) {
    return T(1); // Return positive distance (no collision) for invalid SDF
  }

  // Compute the world-to-SDF transform once
  const auto worldToSdfTransform = computeWorldToSDFTransform(sdfColliderIndex, state);

  // Use the efficient transform-based version
  return sampleSDFDistanceWithTransform(worldPos, sdfCollider, worldToSdfTransform);
}

template <typename T>
T SDFCollisionErrorFunctionT<T>::sampleSDFDistanceWithTransform(
    const Eigen::Vector3<T>& worldPos,
    const SDFCollider& sdfCollider,
    const TransformT<T>& worldToSdfTransform) const {
  // Check if SDF is valid
  if (!sdfCollider.isValid()) {
    return T(1); // Return positive distance (no collision) for invalid SDF
  }

  // Transform world position to SDF local space using pre-computed transform
  const Vector3<T> sdfLocalPos = worldToSdfTransform * worldPos;

  // Sample the SDF at the local position
  const float distance = sdfCollider.sdf->sample(sdfLocalPos);

  return static_cast<T>(distance);
}

template <typename T>
TransformT<T> SDFCollisionErrorFunctionT<T>::computeWorldToSDFTransform(
    size_t sdfColliderIndex,
    const SkeletonStateT<T>& state) const {
  MT_CHECK(
      sdfColliderIndex < sdfColliders_.size(), "Invalid SDF collider index: {}", sdfColliderIndex);

  const auto& sdfCollider = sdfColliders_[sdfColliderIndex];

  TransformT<T> parentTransform;
  if (sdfCollider.parent != kInvalidIndex) {
    // Joint-attached collider: chain joint and collider transformations
    const size_t parentJoint = sdfCollider.parent;
    MT_CHECK(parentJoint < state.jointState.size(), "Invalid parent joint index: {}", parentJoint);

    parentTransform = state.jointState[parentJoint].transform;
  }

  auto sdfWorldTransform = parentTransform * sdfCollider.transformation.cast<T>();

  // Return the inverse transform (world-to-SDF)
  return sdfWorldTransform.inverse();
}

template <typename T>
std::pair<T, Eigen::Vector3<T>>
SDFCollisionErrorFunctionT<T>::sampleSDFDistanceAndGradientWithTransform(
    const Eigen::Vector3<T>& worldPos,
    const SDFCollider& sdfCollider,
    const TransformT<T>& worldToSdfTransform) const {
  // Check if SDF is valid
  if (!sdfCollider.isValid()) {
    return std::make_pair(T(0), Eigen::Vector3<T>::Zero());
  }

  // Transform world position to SDF local space using pre-computed transform
  const Vector3<T> sdfLocalPos = worldToSdfTransform * worldPos;

  // Sample both distance and gradient in SDF local space (most efficient)
  const auto [localDistance, localGradient] = sdfCollider.sdf->sampleWithGradient(sdfLocalPos);

  // Convert distance to template type
  const T distance = static_cast<T>(localDistance);

  // Transform gradient from SDF local space to world space
  // Gradient transforms as: worldGradient = (worldToSdfTransform^-1)^T * localGradient
  const auto sdfWorldTransform = worldToSdfTransform.inverse();
  const Eigen::Vector3<T> worldGradient =
      sdfWorldTransform.rotation * localGradient.template cast<T>();

  return std::make_pair(distance, worldGradient);
}

template <typename T>
axel::BoundingBox<T> SDFCollisionErrorFunctionT<T>::computeBoneBoundingBox(
    size_t boneIndex,
    const MeshT<T>& posedMesh) const {
  const auto& range = boneVertexRanges_[boneIndex];

  // Return empty bounding box if no vertices for this bone
  if (range.empty()) {
    return axel::BoundingBox<T>();
  }

  // Get the first vertex position to initialize the bounding box
  const size_t firstVertexIndex = activeVertexIndices_[range.start];
  const auto& firstPos = posedMesh.vertices[firstVertexIndex];
  axel::BoundingBox<T> bbox(firstPos, T(0));

  // Extend bounding box to include all other vertices for this bone
  for (size_t i = range.start + 1; i < range.end; ++i) {
    const size_t vertexIndex = activeVertexIndices_[i];
    const auto& vertexPos = posedMesh.vertices[vertexIndex];
    bbox.extend(vertexPos);
  }

  return bbox;
}

template <typename T>
axel::BoundingBox<T> SDFCollisionErrorFunctionT<T>::computeSDFWorldBoundingBox(
    size_t sdfColliderIndex,
    const SkeletonStateT<T>& state) const {
  MT_CHECK(
      sdfColliderIndex < sdfColliders_.size(), "Invalid SDF collider index: {}", sdfColliderIndex);

  const auto& sdfCollider = sdfColliders_[sdfColliderIndex];

  // Check if SDF is valid
  if (!sdfCollider.isValid()) {
    return axel::BoundingBox<T>(); // Return empty bounding box for invalid SDF
  }

  TransformT<T> sdfWorldTransform;

  if (sdfCollider.parent == kInvalidIndex) {
    // World-space collider: use collider transformation directly
    sdfWorldTransform = sdfCollider.transformation.cast<T>();
  } else {
    // Joint-attached collider: chain joint and collider transformations
    const size_t parentJoint = sdfCollider.parent;
    MT_CHECK(parentJoint < state.jointState.size(), "Invalid parent joint index: {}", parentJoint);

    // Transform chain: world <- joint <- sdfCollider
    const auto& jointWorldTransform = state.jointState[parentJoint].transform;
    sdfWorldTransform = jointWorldTransform * sdfCollider.transformation.cast<T>();
  }

  // Get the SDF's local bounding box
  const auto& localBounds = sdfCollider.sdf->bounds();

  // Transform the 8 corners of the local bounding box to world space
  const auto& minCorner = localBounds.min();
  const auto& maxCorner = localBounds.max();

  // Generate all 8 corners of the bounding box
  std::array<Eigen::Vector3<T>, 8> corners = {
      {{minCorner.x(), minCorner.y(), minCorner.z()},
       {maxCorner.x(), minCorner.y(), minCorner.z()},
       {minCorner.x(), maxCorner.y(), minCorner.z()},
       {maxCorner.x(), maxCorner.y(), minCorner.z()},
       {minCorner.x(), minCorner.y(), maxCorner.z()},
       {maxCorner.x(), minCorner.y(), maxCorner.z()},
       {minCorner.x(), maxCorner.y(), maxCorner.z()},
       {maxCorner.x(), maxCorner.y(), maxCorner.z()}}};

  // Transform first corner to initialize world bounding box
  const auto firstWorldCorner = sdfWorldTransform * corners[0];
  axel::BoundingBox<T> worldBounds(firstWorldCorner, T(0));

  // Transform remaining corners and extend the bounding box
  for (size_t i = 1; i < corners.size(); ++i) {
    const auto worldCorner = sdfWorldTransform * corners[i];
    worldBounds.extend(worldCorner);
  }

  return worldBounds;
}

// Explicit template instantiations
template class SDFCollisionErrorFunctionT<float>;
template class SDFCollisionErrorFunctionT<double>;

} // namespace momentum
