/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/mesh_state.h>
#include <momentum/character/sdf_collision_geometry.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/sdf_collision_utility.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skinning_weight_iterator.h>
#include <momentum/common/checks.h>
#include <momentum/common/profile.h>
#include <momentum/math/fwd.h>
#include <momentum/math/mesh.h>
#include <momentum/math/utility.h>

#include <axel/BoundingBox.h>

#include <algorithm>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

namespace momentum {

/// Range structure for efficiently storing vertex indices per bone.
/// Provides [start, end) indices into the activeVertexIndices_ array.
struct BoneVertexRange {
  size_t start; ///< Start index in activeVertexIndices_
  size_t end; ///< End index in activeVertexIndices_ (exclusive)

  /// Returns the number of vertices for this bone.
  [[nodiscard]] size_t size() const {
    return end - start;
  }

  /// Returns true if this bone has no vertices.
  [[nodiscard]] bool empty() const {
    return start == end;
  }
};

/// Collision pair information for bone-SDF collider combinations.
template <typename T>
struct BoneCollisionPairT {
  size_t boneIndex; ///< Primary bone index
  size_t sdfColliderIndex; ///< Index into SDFCollisionGeometry
};

/// SDF-based collision error function that penalizes vertex penetration into SDF colliders.
///
/// This error function tests source vertices against SDF collision objects and applies
/// penalties for interpenetration. It uses an efficient contiguous memory layout where
/// vertices are grouped by their primary bone index to improve cache locality and
/// reduce memory usage.
///
/// The SdfColliderType must provide the following duck-typed interface:
/// - `T evaluate(const Vector3<T>& localPos) const` — signed distance
/// - `pair<T, Vector3<T>> evaluateWithGradient(const Vector3<T>& localPos) const`
/// - `BoundingBox<S> bounds() const` — local-frame AABB
/// - `size_t parentJoint() const` — parent joint index
/// - `TransformT<S> localToParentTransform() const` — local-to-parent transform
/// - `void update(const VectorX<T>& modelParams)` — per-iteration update
/// - `bool isValid() const` — whether the collider is usable
template <typename T, typename SdfColliderType = SDFColliderT<float>>
class SDFCollisionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  /// Constructor with vertex index list and uniform weights.
  ///
  /// @param character The character containing skeleton, mesh, and skin weights
  /// @param sdfColliders The SDF collision geometry to test against
  /// @param participatingVertices Vertex indices to include (empty means all vertices)
  /// @param vertexWeights Per-vertex collision weights (empty means uniform weight of 1.0)
  explicit SDFCollisionErrorFunctionT(
      const Character& character,
      const std::vector<SdfColliderType>& sdfColliders,
      const std::vector<int>& participatingVertices = {},
      const std::vector<T>& vertexWeights = {},
      bool filterRestPoseIntersections = true,
      uint8_t maxCollisionsPerVertex = 1);

  SDFCollisionErrorFunctionT(const SDFCollisionErrorFunctionT&) = default;
  SDFCollisionErrorFunctionT& operator=(const SDFCollisionErrorFunctionT&) = delete;
  SDFCollisionErrorFunctionT(SDFCollisionErrorFunctionT&&) noexcept = default;
  SDFCollisionErrorFunctionT& operator=(SDFCollisionErrorFunctionT&&) noexcept = delete;
  ~SDFCollisionErrorFunctionT() override = default;

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) override;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const override;

  [[nodiscard]] bool needsMesh() const override {
    return true;
  }

  /// Returns the character used to update mesh state for SDF collision evaluation.
  [[nodiscard]] const Character* getCharacter() const override {
    return &character_;
  }

  [[nodiscard]] size_t getNumActiveVertices() const {
    return activeVertexIndices_.size();
  }

  [[nodiscard]] size_t getNumCollisionPairs() const {
    return boneCollisionPairs_.size();
  }

  static constexpr T kSDFCollisionWeight = T(5e-3);

 private:
  void organizeVerticesByBone(
      const std::vector<int>& participatingVertices,
      const std::vector<T>& weights);

  void organizeVerticesByBone(const std::vector<std::pair<int, T>>& vertexWeightPairs);

  void initializeBoneCollisionPairs(bool filterRestPoseIntersections);

  [[nodiscard]] bool testRestPoseIntersection(
      size_t boneIndex,
      size_t sdfColliderIndex,
      const SkeletonStateT<T>& restState) const;

  void computeActivePairs(const MeshStateT<T>& meshState, const SkeletonStateT<T>& state);

  void updateColliders(const ModelParametersT<T>& params);

  [[nodiscard]] TransformT<T> computeWorldToColliderTransform(
      size_t sdfColliderIndex,
      const SkeletonStateT<T>& state) const;

  [[nodiscard]] axel::BoundingBox<T> computeBoneBoundingBox(
      size_t boneIndex,
      const MeshT<T>& posedMesh) const;

  [[nodiscard]] axel::BoundingBox<T> computeColliderWorldBoundingBox(
      size_t sdfColliderIndex,
      const SkeletonStateT<T>& state) const;

  void accumulateJointGradient(
      const JointStateT<T>& jointState,
      size_t paramIndex,
      const Eigen::Vector3<T>& posd,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  void accumulateJointJacobian(
      const JointStateT<T>& jointState,
      size_t paramIndex,
      const Eigen::Vector3<T>& posd,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const;

  void accumulateColliderHierarchyGradient(
      const SkeletonStateT<T>& state,
      size_t colliderParent,
      size_t commonAncestor,
      const Eigen::Vector3<T>& sdfSurfacePoint,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  void accumulateColliderHierarchyJacobian(
      const SkeletonStateT<T>& state,
      size_t colliderParent,
      size_t commonAncestor,
      const Eigen::Vector3<T>& sdfSurfacePoint,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const;

  const Character& character_;
  std::vector<SdfColliderType> sdfColliders_;

  std::vector<size_t> activeVertexIndices_;
  std::vector<BoneVertexRange> boneVertexRanges_;
  std::vector<T> vertexWeights_;

  std::vector<BoneCollisionPairT<T>> boneCollisionPairs_;

  uint8_t maxCollisionsPerVertex_ = 1;

  std::vector<bool> activePairs_;
  std::vector<uint8_t> collisionCounts_;
  std::vector<axel::BoundingBox<T>> boneBBoxCache_;
  std::vector<bool> boneBBoxValid_;
  std::vector<axel::BoundingBox<T>> sdfBBoxCache_;
};

using SDFCollisionErrorFunction = SDFCollisionErrorFunctionT<float>;
using SDFCollisionErrorFunctiond = SDFCollisionErrorFunctionT<double>;

// ============================================================================
// Template implementation
// ============================================================================
//
// Shared, stateless helpers (collider transforms, bounding boxes, joint chain-rule accumulation,
// common-ancestor lookup) live in the `detail_sdf_collision` namespace in sdf_collision_utility.h
// so they can be reused by the cross-frame (sequence) collision error functions. The member
// functions below forward to those helpers, supplying this error function's skeleton, parameter
// transform, active joints, and collider list.

template <typename T, typename SdfColliderType>
SDFCollisionErrorFunctionT<T, SdfColliderType>::SDFCollisionErrorFunctionT(
    const Character& character,
    const std::vector<SdfColliderType>& sdfColliders,
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
  organizeVerticesByBone(participatingVertices, vertexWeights);
  initializeBoneCollisionPairs(filterRestPoseIntersections);
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::organizeVerticesByBone(
    const std::vector<int>& participatingVertices,
    const std::vector<T>& weights) {
  std::vector<int> vertices;
  if (participatingVertices.empty()) {
    vertices.reserve(character_.mesh->vertices.size());
    for (int i = 0; i < static_cast<int>(character_.mesh->vertices.size()); ++i) {
      vertices.push_back(i);
    }
  } else {
    vertices = participatingVertices;
  }

  for (int vertexIndex : vertices) {
    MT_CHECK(
        vertexIndex >= 0 && vertexIndex < static_cast<int>(character_.mesh->vertices.size()),
        "Invalid vertex index: {} (mesh has {} vertices)",
        vertexIndex,
        character_.mesh->vertices.size());
  }

  std::vector<T> vertexWeights;
  if (weights.empty()) {
    vertexWeights.assign(vertices.size(), T(1));
  } else {
    MT_CHECK(
        weights.size() == vertices.size(),
        "Weight count ({}) must match vertex count ({})",
        weights.size(),
        vertices.size());
    vertexWeights = weights;
  }

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

  std::sort(vertexInfos.begin(), vertexInfos.end(), [](const VertexInfo& a, const VertexInfo& b) {
    return a.primaryBone < b.primaryBone;
  });

  boneVertexRanges_.resize(character_.skeleton.joints.size());
  activeVertexIndices_.reserve(vertices.size());
  vertexWeights_.reserve(vertices.size());

  size_t currentBone = 0;
  for (size_t i = 0; i < vertexInfos.size(); ++i) {
    const auto& info = vertexInfos[i];
    while (currentBone < info.primaryBone) {
      boneVertexRanges_[currentBone].end = i;
      ++currentBone;
      boneVertexRanges_[currentBone].start = i;
    }
    activeVertexIndices_.push_back(info.vertexIndex);
    vertexWeights_.push_back(info.weight);
  }

  boneVertexRanges_[currentBone].end = activeVertexIndices_.size();
  ++currentBone;
  while (currentBone < boneVertexRanges_.size()) {
    boneVertexRanges_[currentBone++] =
        BoneVertexRange{activeVertexIndices_.size(), activeVertexIndices_.size()};
  }
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::initializeBoneCollisionPairs(
    bool filterRestPoseIntersections) {
  std::unordered_set<size_t> activeBones;
  for (size_t boneIndex = 0; boneIndex < boneVertexRanges_.size(); ++boneIndex) {
    if (!boneVertexRanges_[boneIndex].empty()) {
      activeBones.insert(boneIndex);
    }
  }

  const SkeletonStateT<T> restState(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  const ModelParametersT<T> restParams =
      ModelParametersT<T>::Zero(this->parameterTransform_.numAllModelParameters());
  updateColliders(restParams);

  boneCollisionPairs_.clear();
  boneCollisionPairs_.reserve(activeBones.size() * sdfColliders_.size());

  for (size_t boneIndex : activeBones) {
    if (boneVertexRanges_[boneIndex].empty()) {
      continue;
    }
    for (size_t sdfIndex = 0; sdfIndex < sdfColliders_.size(); ++sdfIndex) {
      const auto& collider = sdfColliders_[sdfIndex];
      if (collider.parentJoint() != kInvalidIndex && filterRestPoseIntersections &&
          testRestPoseIntersection(boneIndex, sdfIndex, restState)) {
        continue;
      }
      boneCollisionPairs_.emplace_back(BoneCollisionPairT<T>{boneIndex, sdfIndex});
    }
  }
}

template <typename T, typename SdfColliderType>
bool SDFCollisionErrorFunctionT<T, SdfColliderType>::testRestPoseIntersection(
    size_t boneIndex,
    size_t sdfColliderIndex,
    const SkeletonStateT<T>& restState) const {
  const auto& range = boneVertexRanges_[boneIndex];
  if (range.empty()) {
    return false;
  }

  const auto& collider = sdfColliders_[sdfColliderIndex];
  if (!collider.isValid()) {
    return false;
  }

  const auto worldToCollider = computeWorldToColliderTransform(sdfColliderIndex, restState);

  for (size_t i = range.start; i < range.end; ++i) {
    const size_t vertexIndex = activeVertexIndices_[i];
    const auto& restPos = character_.mesh->vertices[vertexIndex].template cast<T>();
    const Vector3<T> localPos = worldToCollider * restPos;
    const T distance = collider.evaluate(localPos);
    if (distance < T(0)) {
      return true;
    }
  }
  return false;
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::updateColliders(
    const ModelParametersT<T>& params) {
  for (auto& collider : sdfColliders_) {
    collider.update(params.v);
  }
}

template <typename T, typename SdfColliderType>
TransformT<T> SDFCollisionErrorFunctionT<T, SdfColliderType>::computeWorldToColliderTransform(
    size_t sdfColliderIndex,
    const SkeletonStateT<T>& state) const {
  return detail_sdf_collision::computeWorldToColliderTransform(
      sdfColliders_[sdfColliderIndex], state);
}

template <typename T, typename SdfColliderType>
double SDFCollisionErrorFunctionT<T, SdfColliderType>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();
  MT_CHECK_NOTNULL(meshState.posedMesh_);
  updateColliders(params);
  computeActivePairs(meshState, state);

  double totalError = 0.0;
  collisionCounts_.assign(activeVertexIndices_.size(), 0);

  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    if (!activePairs_[pairIndex]) {
      continue;
    }
    const auto& pair = boneCollisionPairs_[pairIndex];
    const auto& range = boneVertexRanges_[pair.boneIndex];
    if (range.empty()) {
      continue;
    }
    const auto& collider = sdfColliders_[pair.sdfColliderIndex];
    if (!collider.isValid()) {
      continue;
    }
    const auto worldToCollider = computeWorldToColliderTransform(pair.sdfColliderIndex, state);

    for (size_t i = range.start; i < range.end; ++i) {
      if (collisionCounts_[i] >= maxCollisionsPerVertex_) {
        continue;
      }
      const size_t vertexIndex = activeVertexIndices_[i];
      const T vertexWeight = vertexWeights_[i];
      const auto& worldPos = meshState.posedMesh_->vertices[vertexIndex];
      const Vector3<T> localPos = worldToCollider * worldPos;
      const T distance = collider.evaluate(localPos);

      if (distance < T(0)) {
        collisionCounts_[i]++;
        const T penetrationDepth = -distance;
        totalError += static_cast<double>(
            vertexWeight * penetrationDepth * penetrationDepth * kSDFCollisionWeight *
            this->weight_);
      }
    }
  }
  return totalError;
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::accumulateJointGradient(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  detail_sdf_collision::accumulateJointGradient(
      jointState,
      paramIndex,
      posd,
      direction,
      this->activeJointParams_,
      this->parameterTransform_,
      gradient);
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::accumulateJointJacobian(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const {
  detail_sdf_collision::accumulateJointJacobian(
      jointState,
      paramIndex,
      posd,
      direction,
      this->activeJointParams_,
      this->parameterTransform_,
      jacobianRow);
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::accumulateColliderHierarchyGradient(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  detail_sdf_collision::accumulateColliderHierarchyGradient(
      state,
      colliderParent,
      commonAncestor,
      sdfSurfacePoint,
      direction,
      this->skeleton_,
      this->activeJointParams_,
      this->parameterTransform_,
      gradient);
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::accumulateColliderHierarchyJacobian(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const {
  detail_sdf_collision::accumulateColliderHierarchyJacobian(
      state,
      colliderParent,
      commonAncestor,
      sdfSurfacePoint,
      direction,
      this->skeleton_,
      this->activeJointParams_,
      this->parameterTransform_,
      jacobianRow);
}

template <typename T, typename SdfColliderType>
double SDFCollisionErrorFunctionT<T, SdfColliderType>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();
  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);
  updateColliders(params);
  computeActivePairs(meshState, state);

  double totalError = 0.0;
  collisionCounts_.assign(activeVertexIndices_.size(), 0);

  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    if (!activePairs_[pairIndex]) {
      continue;
    }
    const auto& pair = boneCollisionPairs_[pairIndex];
    const auto& range = boneVertexRanges_[pair.boneIndex];
    if (range.empty()) {
      continue;
    }
    const auto& collider = sdfColliders_[pair.sdfColliderIndex];
    if (!collider.isValid()) {
      continue;
    }
    const auto worldToCollider = computeWorldToColliderTransform(pair.sdfColliderIndex, state);
    const auto colliderToWorld = worldToCollider.inverse();

    for (size_t i = range.start; i < range.end; ++i) {
      if (collisionCounts_[i] >= maxCollisionsPerVertex_) {
        continue;
      }
      const size_t vertexIndex = activeVertexIndices_[i];
      const T vertexWeight = vertexWeights_[i];
      const auto& worldPos = meshState.posedMesh_->vertices[vertexIndex];
      const Vector3<T> localPos = worldToCollider * worldPos;
      const auto [distance, localGradient] = collider.evaluateWithGradient(localPos);

      if (distance < T(0)) {
        const T penetrationDepth = -distance;
        totalError += static_cast<double>(
            vertexWeight * penetrationDepth * penetrationDepth * kSDFCollisionWeight *
            this->weight_);
        collisionCounts_[i]++;

        // Transform gradient from collider local space to world space
        const Eigen::Vector3<T> sdfGradient = colliderToWorld.rotation * localGradient;

        const T wgt = vertexWeight * 2.0f * kSDFCollisionWeight * this->weight_;
        const Eigen::Vector3<T> diff = -penetrationDepth * sdfGradient;
        const Eigen::Vector3<T> sdfSurfacePoint = worldPos + penetrationDepth * sdfGradient;

        const size_t commonAncestor = detail_sdf_collision::findCommonAncestorForVertex(
            *character_.skinWeights, this->skeleton_, vertexIndex, collider.parentJoint());

        SkinningWeightIteratorT<T> skinningIter(
            character_, *meshState.restMesh_, state, vertexIndex);
        while (!skinningIter.finished()) {
          const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();
          if (jointIndex == commonAncestor) {
            break;
          }
          if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
            continue;
          }
          MT_CHECK(jointIndex < this->skeleton_.joints.size());
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Eigen::Vector3<T> posd = transformedVertex - jointState.translation();
          accumulateJointGradient(jointState, paramIndex, posd, boneWeight * wgt * diff, gradient);
        }

        if (collider.parentJoint() != kInvalidIndex) {
          accumulateColliderHierarchyGradient(
              state,
              collider.parentJoint(),
              commonAncestor,
              sdfSurfacePoint,
              -wgt * diff,
              gradient);
        }
      }
    }
  }
  return totalError;
}

template <typename T, typename SdfColliderType>
double SDFCollisionErrorFunctionT<T, SdfColliderType>::getJacobian(
    const ModelParametersT<T>& params,
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
  updateColliders(params);
  computeActivePairs(meshState, state);

  double totalError = 0.0;
  int currentRow = 0;
  collisionCounts_.assign(activeVertexIndices_.size(), 0);

  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    if (!activePairs_[pairIndex]) {
      continue;
    }
    const auto& pair = boneCollisionPairs_[pairIndex];
    const auto& range = boneVertexRanges_[pair.boneIndex];
    if (range.empty()) {
      continue;
    }
    const auto& collider = sdfColliders_[pair.sdfColliderIndex];
    if (!collider.isValid()) {
      continue;
    }
    const auto worldToCollider = computeWorldToColliderTransform(pair.sdfColliderIndex, state);
    const auto colliderToWorld = worldToCollider.inverse();

    for (size_t i = range.start; i < range.end; ++i) {
      if (collisionCounts_[i] >= maxCollisionsPerVertex_) {
        continue;
      }
      const size_t vertexIndex = activeVertexIndices_[i];
      const T vertexWeight = vertexWeights_[i];
      const auto& worldPos = meshState.posedMesh_->vertices[vertexIndex];
      const Vector3<T> localPos = worldToCollider * worldPos;
      const auto [distance, localGradient] = collider.evaluateWithGradient(localPos);

      if (distance < T(0)) {
        const T penetrationDepth = -distance;
        MT_CHECK(
            currentRow < jacobian.rows() && currentRow < residual.rows(),
            "Insufficient Jacobian/residual rows");

        const T wgt = std::sqrt(vertexWeight * kSDFCollisionWeight * this->weight_);
        residual[currentRow] = wgt * penetrationDepth;
        totalError += static_cast<double>(
            vertexWeight * penetrationDepth * penetrationDepth * kSDFCollisionWeight *
            this->weight_);
        ++collisionCounts_[i];

        jacobian.row(currentRow).setZero();

        // Transform gradient from collider local space to world space
        const Eigen::Vector3<T> sdfGradient = colliderToWorld.rotation * localGradient;

        const T jacobianMagnitude = -wgt;
        const Eigen::Vector3<T> worldJacobianContribution = jacobianMagnitude * sdfGradient;
        const Eigen::Vector3<T> sdfSurfacePoint = worldPos + penetrationDepth * sdfGradient;

        const size_t commonAncestor = detail_sdf_collision::findCommonAncestorForVertex(
            *character_.skinWeights, this->skeleton_, vertexIndex, collider.parentJoint());

        SkinningWeightIteratorT<T> skinningIter(
            character_, *meshState.restMesh_, state, vertexIndex);
        while (!skinningIter.finished()) {
          const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();
          if (jointIndex == commonAncestor) {
            break;
          }
          if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
            continue;
          }
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

        if (collider.parentJoint() != kInvalidIndex) {
          accumulateColliderHierarchyJacobian(
              state,
              collider.parentJoint(),
              commonAncestor,
              sdfSurfacePoint,
              -worldJacobianContribution,
              jacobian.middleRows(currentRow, 1));
        }
        currentRow++;
      }
    }
  }

  usedRows = currentRow;
  return totalError;
}

template <typename T, typename SdfColliderType>
size_t SDFCollisionErrorFunctionT<T, SdfColliderType>::getJacobianSize() const {
  return static_cast<size_t>(maxCollisionsPerVertex_) * activeVertexIndices_.size();
}

template <typename T, typename SdfColliderType>
void SDFCollisionErrorFunctionT<T, SdfColliderType>::computeActivePairs(
    const MeshStateT<T>& meshState,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();
  activePairs_.assign(boneCollisionPairs_.size(), false);

  boneBBoxCache_.resize(boneVertexRanges_.size());
  boneBBoxValid_.assign(boneVertexRanges_.size(), false);
  for (size_t boneIndex = 0; boneIndex < boneVertexRanges_.size(); ++boneIndex) {
    if (!boneVertexRanges_[boneIndex].empty()) {
      boneBBoxCache_[boneIndex] = computeBoneBoundingBox(boneIndex, *meshState.posedMesh_);
      boneBBoxValid_[boneIndex] = true;
    }
  }

  sdfBBoxCache_.resize(sdfColliders_.size());
  for (size_t sdfIndex = 0; sdfIndex < sdfColliders_.size(); ++sdfIndex) {
    sdfBBoxCache_[sdfIndex] = computeColliderWorldBoundingBox(sdfIndex, state);
  }

  for (size_t pairIndex = 0; pairIndex < boneCollisionPairs_.size(); ++pairIndex) {
    const auto& pair = boneCollisionPairs_[pairIndex];
    if (!boneBBoxValid_[pair.boneIndex]) {
      continue;
    }
    activePairs_[pairIndex] =
        boneBBoxCache_[pair.boneIndex].intersects(sdfBBoxCache_[pair.sdfColliderIndex]);
  }
}

template <typename T, typename SdfColliderType>
axel::BoundingBox<T> SDFCollisionErrorFunctionT<T, SdfColliderType>::computeBoneBoundingBox(
    size_t boneIndex,
    const MeshT<T>& posedMesh) const {
  const auto& range = boneVertexRanges_[boneIndex];
  return detail_sdf_collision::computeBoneBoundingBox<T>(
      std::span<const size_t>(activeVertexIndices_).subspan(range.start, range.size()), posedMesh);
}

template <typename T, typename SdfColliderType>
axel::BoundingBox<T>
SDFCollisionErrorFunctionT<T, SdfColliderType>::computeColliderWorldBoundingBox(
    size_t sdfColliderIndex,
    const SkeletonStateT<T>& state) const {
  return detail_sdf_collision::computeColliderWorldBoundingBox(
      sdfColliders_[sdfColliderIndex], state);
}

} // namespace momentum
