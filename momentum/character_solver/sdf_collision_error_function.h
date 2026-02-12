/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/sdf_collision_geometry.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/fwd.h>

#include <axel/BoundingBox.h>

#include <memory>
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
/// Key features:
/// - Configurable vertex selection for performance with high-resolution meshes
/// - Bone-based collision filtering to exclude rest pose intersections
/// - Efficient contiguous memory organization for cache-friendly processing
template <typename T>
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
      const SDFCollisionGeometry& sdfColliders,
      const std::vector<int>& participatingVertices = {},
      const std::vector<T>& vertexWeights = {},
      bool filterRestPoseIntersections = true,
      uint8_t maxCollisionsPerVertex = 1);

  /// Destructor
  ~SDFCollisionErrorFunctionT() override = default;

  /// Computes the collision error for the given model parameters and skeleton state.
  ///
  /// @param params The model parameters
  /// @param state The skeleton state
  /// @param meshState The mesh state containing posed mesh for collision testing
  /// @return The collision error value
  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) override;

  /// Computes the collision error and its gradient.
  ///
  /// @param params The model parameters
  /// @param state The skeleton state
  /// @param meshState The mesh state containing posed mesh for collision testing
  /// @param gradient Output gradient vector
  /// @return The collision error value
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  /// Computes the collision error Jacobian and residual.
  ///
  /// @param params The model parameters
  /// @param state The skeleton state
  /// @param meshState The mesh state containing posed mesh for collision testing
  /// @param jacobian Output Jacobian matrix
  /// @param residual Output residual vector
  /// @param usedRows Number of rows actually used in jacobian and residual
  /// @return The collision error value
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

  /// Returns the maximum number of Jacobian rows needed.
  [[nodiscard]] size_t getJacobianSize() const override;

  /// Returns true because this error function requires mesh state.
  [[nodiscard]] bool needsMesh() const override {
    return true;
  }

  /// Returns the number of active vertices participating in collisions.
  [[nodiscard]] size_t getNumActiveVertices() const {
    return activeVertexIndices_.size();
  }

  /// Returns the number of bone-collider pairs being tested.
  [[nodiscard]] size_t getNumCollisionPairs() const {
    return boneCollisionPairs_.size();
  }

  /// Weight constant for SDF collision penalties.
  static constexpr T kSDFCollisionWeight = T(5e-3);

 private:
  /// Organizes vertices by bone in contiguous arrays for efficient processing.
  ///
  /// @param participatingVertices Vertex indices to include
  /// @param weights Per-vertex weights
  void organizeVerticesByBone(
      const std::vector<int>& participatingVertices,
      const std::vector<T>& weights);

  /// Organizes vertices by bone from vertex-weight pairs.
  ///
  /// @param vertexWeightPairs Vector of (vertex_index, weight) pairs
  void organizeVerticesByBone(const std::vector<std::pair<int, T>>& vertexWeightPairs);

  /// Initializes bone-collider pairs and tests rest pose intersections.
  void initializeBoneCollisionPairs(bool filterRestPoseIntersections);

  /// Tests if any vertex for a bone intersects the SDF in rest pose.
  ///
  /// @param boneIndex The bone index to test
  /// @param sdfColliderIndex The SDF collider index to test
  /// @param restState The rest pose skeleton state
  /// @return True if any vertex penetrates the SDF in rest pose
  [[nodiscard]] bool testRestPoseIntersection(
      size_t boneIndex,
      size_t sdfColliderIndex,
      const SkeletonStateT<T>& restState) const;

  /// Updates which collision pairs are currently active.
  ///
  /// @param meshState The mesh state for bounding box computation
  /// @param state The current skeleton state
  void computeActivePairs(const MeshStateT<T>& meshState, const SkeletonStateT<T>& state);

  /// Samples the SDF distance at a world position.
  ///
  /// @param worldPos The world position to sample
  /// @param sdfColliderIndex The SDF collider index
  /// @param state The skeleton state
  /// @return The signed distance (negative = inside)
  [[nodiscard]] T sampleSDFDistance(
      const Eigen::Vector3<T>& worldPos,
      size_t sdfColliderIndex,
      const SkeletonStateT<T>& state) const;

  /// Samples the SDF distance using a pre-computed world-to-SDF transform.
  /// This is more efficient when sampling multiple points against the same SDF.
  ///
  /// @param worldPos The world position to sample
  /// @param sdfCollider The SDF collider reference
  /// @param worldToSdfTransform Pre-computed inverse of the SDF world transform
  /// @return The signed distance (negative = inside)
  [[nodiscard]] T sampleSDFDistanceWithTransform(
      const Eigen::Vector3<T>& worldPos,
      const SDFCollider& sdfCollider,
      const TransformT<T>& worldToSdfTransform) const;

  /// Computes the world-to-SDF transform for efficient multiple sampling.
  ///
  /// @param sdfColliderIndex The SDF collider index
  /// @param state The skeleton state
  /// @return Transform from world space to SDF local space
  [[nodiscard]] TransformT<T> computeWorldToSDFTransform(
      size_t sdfColliderIndex,
      const SkeletonStateT<T>& state) const;

  /// Computes both SDF distance and gradient using a pre-computed transform.
  /// Most efficient method for collision gradient computation.
  ///
  /// @param worldPos The world position to sample
  /// @param sdfCollider The SDF collider reference
  /// @param worldToSdfTransform Pre-computed inverse of the SDF world transform
  /// @return Pair of (distance, gradient) in world space
  [[nodiscard]] std::pair<T, Eigen::Vector3<T>> sampleSDFDistanceAndGradientWithTransform(
      const Eigen::Vector3<T>& worldPos,
      const SDFCollider& sdfCollider,
      const TransformT<T>& worldToSdfTransform) const;

  /// Computes the axis-aligned bounding box for all vertices associated with a bone.
  ///
  /// @param boneIndex The bone index
  /// @param posedMesh The posed mesh for vertex positions
  /// @return Bounding box containing all vertices for the bone
  [[nodiscard]] axel::BoundingBox<T> computeBoneBoundingBox(
      size_t boneIndex,
      const MeshT<T>& posedMesh) const;

  /// Computes the world-space bounding box for an SDF collider.
  ///
  /// @param sdfColliderIndex The SDF collider index
  /// @param state The skeleton state for transformation
  /// @return World-space bounding box of the SDF collider
  [[nodiscard]] axel::BoundingBox<T> computeSDFWorldBoundingBox(
      size_t sdfColliderIndex,
      const SkeletonStateT<T>& state) const;

  /// Accumulates gradient contributions for a single joint's 7 parameters
  /// (3 translation + 3 rotation + 1 scale).
  ///
  /// @param jointState The joint state for the current joint
  /// @param paramIndex The parameter index (jointIndex * kParametersPerJoint)
  /// @param posd The lever arm (displacement from joint to the relevant point)
  /// @param direction The weighted direction vector (includes sign and weight)
  /// @param gradient Output gradient vector to accumulate into
  void accumulateJointGradient(
      const JointStateT<T>& jointState,
      size_t paramIndex,
      const Eigen::Vector3<T>& posd,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Accumulates jacobian contributions for a single joint's 7 parameters
  /// (3 translation + 3 rotation + 1 scale).
  ///
  /// @param jointState The joint state for the current joint
  /// @param paramIndex The parameter index (jointIndex * kParametersPerJoint)
  /// @param posd The lever arm (displacement from joint to the relevant point)
  /// @param direction The weighted direction vector (includes sign and weight)
  /// @param jacobianRow Output jacobian row to accumulate into
  void accumulateJointJacobian(
      const JointStateT<T>& jointState,
      size_t paramIndex,
      const Eigen::Vector3<T>& posd,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const;

  /// Walks up the collider hierarchy from colliderParent to commonAncestor,
  /// accumulating gradient contributions at each joint.
  ///
  /// @param state The skeleton state
  /// @param colliderParent The starting joint (SDF collider's parent)
  /// @param commonAncestor The joint to stop at (exclusive)
  /// @param sdfSurfacePoint The closest point on the SDF surface (lever arm)
  /// @param direction The weighted direction vector (includes sign and weight)
  /// @param gradient Output gradient vector to accumulate into
  void accumulateColliderHierarchyGradient(
      const SkeletonStateT<T>& state,
      size_t colliderParent,
      size_t commonAncestor,
      const Eigen::Vector3<T>& sdfSurfacePoint,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Walks up the collider hierarchy from colliderParent to commonAncestor,
  /// accumulating jacobian contributions at each joint.
  ///
  /// @param state The skeleton state
  /// @param colliderParent The starting joint (SDF collider's parent)
  /// @param commonAncestor The joint to stop at (exclusive)
  /// @param sdfSurfacePoint The closest point on the SDF surface (lever arm)
  /// @param direction The weighted direction vector (includes sign and weight)
  /// @param jacobianRow Output jacobian row to accumulate into
  void accumulateColliderHierarchyJacobian(
      const SkeletonStateT<T>& state,
      size_t colliderParent,
      size_t commonAncestor,
      const Eigen::Vector3<T>& sdfSurfacePoint,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const;

  const Character& character_; ///< Reference to the character
  const SDFCollisionGeometry& sdfColliders_; ///< Reference to SDF colliders

  // Contiguous vertex organization for efficient processing
  std::vector<size_t> activeVertexIndices_; ///< All active vertex indices
  std::vector<BoneVertexRange> boneVertexRanges_; ///< [start, end) ranges per bone
  std::vector<T> vertexWeights_; ///< Collision weights per vertex

  // Bone-collider pair filtering
  std::vector<BoneCollisionPairT<T>> boneCollisionPairs_; ///< All bone-collider pairs

  uint8_t maxCollisionsPerVertex_ = 1; ///< Maximum number of collisions per vertex

  // Reusable buffers to avoid repeated heap allocations during optimization
  std::vector<bool> activePairs_; ///< Active collision pairs from broad-phase
  std::vector<uint8_t> collisionCounts_; ///< Per-vertex collision counts
  std::vector<axel::BoundingBox<T>> boneBBoxCache_; ///< Cached bone bounding boxes
  std::vector<bool> boneBBoxValid_; ///< Whether each bone bbox is valid
  std::vector<axel::BoundingBox<T>> sdfBBoxCache_; ///< Cached SDF bounding boxes
};

using SDFCollisionErrorFunction = SDFCollisionErrorFunctionT<float>;
using SDFCollisionErrorFunctiond = SDFCollisionErrorFunctionT<double>;

} // namespace momentum
