/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>

#include <Eigen/Core>

#include <span>
#include <vector>

namespace momentum {

/// Shared utility for derivative computation in error functions.
///
/// This class encapsulates the fundamental derivative primitive used by all
/// skeleton-based error functions: the joint hierarchy walk that accumulates
/// derivatives from a joint up to the root.
///
/// @section Usage
/// @code
/// SkeletonDerivativeT<float> deriv(skeleton, parameterTransform, activeJointParams,
///                                   enabledParameters);
///
/// // For joint-attached vectors (single or batched):
/// deriv.accumulateJointGradient<3>(
///     jointIndex, worldVecs, dfdvs, isPoints, weightedResidual, jointStates, gradient);
///
/// deriv.accumulateJointJacobian<3>(
///     jointIndex, worldVecs, dfdvs, isPoints, scale, jointStates, jacobian, rowIndex);
/// @endcode
///
/// @section Performance
/// The utility preserves all existing optimizations:
/// - Per-DOF activeJointParams_ check
/// - Ancestor-only traversal (not full skeleton)
/// - Batched vectors at same joint (single hierarchy walk, vector loop inside)
/// - Per-vector zero-derivative skip
///
/// @tparam T Scalar type (float or double)
template <typename T>
class SkeletonDerivativeT {
 public:
  /// Constructs the derivative utility.
  ///
  /// @param skeleton The skeleton structure (joint hierarchy)
  /// @param parameterTransform Maps joint parameters to model parameters
  /// @param activeJointParams Bitset indicating which joint DOFs are active
  /// @param enabledParameters Bitset indicating which model parameters are enabled
  SkeletonDerivativeT(
      const Skeleton& skeleton,
      const ParameterTransform& parameterTransform,
      const VectorX<bool>& activeJointParams,
      const ParameterSet& enabledParameters);

  /// Accumulates gradient for vectors attached to a single joint.
  ///
  /// This method walks the joint hierarchy from the specified joint to the root,
  /// accumulating derivatives for all vectors in a single pass. Critical for
  /// performance: the vector loop is INSIDE the hierarchy walk loop.
  ///
  /// @tparam FuncDim Dimension of the residual function (1, 2, 3, or 9)
  /// @param jointIndex Index of the joint the vectors are attached to
  /// @param worldVecs World-space vectors (position or direction)
  /// @param dfdvs Derivatives df/dv for each vector (FuncDim x 3 matrices)
  /// @param isPoints True for each vector that is a point (w=1), false for directions (w=0)
  /// @param weightedResidual Weighted residual (2 * weight * loss_deriv * f)
  /// @param jointStates Current joint states
  /// @param gradient Output gradient vector (accumulated, not overwritten)
  template <size_t FuncDim>
  void accumulateJointGradient(
      size_t jointIndex,
      std::span<const Eigen::Vector3<T>> worldVecs,
      std::span<const Eigen::Matrix<T, FuncDim, 3>> dfdvs,
      std::span<const uint8_t> isPoints,
      const Eigen::Matrix<T, FuncDim, 1>& weightedResidual,
      const JointStateListT<T>& jointStates,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Accumulates Jacobian for vectors attached to a single joint.
  ///
  /// Similar to accumulateJointGradient, but accumulates into a Jacobian matrix
  /// instead of a gradient vector. Uses runtime-sized middleRows to handle
  /// FuncDim=1 Eigen block type issues.
  ///
  /// @tparam FuncDim Dimension of the residual function (1, 2, 3, or 9)
  /// @param jointIndex Index of the joint the vectors are attached to
  /// @param worldVecs World-space vectors (position or direction)
  /// @param dfdvs Derivatives df/dv for each vector (FuncDim x 3 matrices)
  /// @param isPoints True for each vector that is a point (w=1), false for directions (w=0)
  /// @param scale Weight multiplier (typically sqrt of loss derivative * constraint weight)
  /// @param jointStates Current joint states
  /// @param jacobian Output Jacobian matrix (accumulated, not overwritten)
  /// @param rowIndex Starting row in the Jacobian for this constraint
  template <size_t FuncDim>
  void accumulateJointJacobian(
      size_t jointIndex,
      std::span<const Eigen::Vector3<T>> worldVecs,
      std::span<const Eigen::Matrix<T, FuncDim, 3>> dfdvs,
      std::span<const uint8_t> isPoints,
      T scale,
      const JointStateListT<T>& jointStates,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  /// Accumulates the gradient CORRECTION for how a mesh normal rotates with joints.
  ///
  /// When a vertex constraint uses a source mesh normal (Normal/SymmetricNormal types),
  /// the standard accumulateVertexGradient treats the normal as constant. But the source
  /// normal rotates with each bone. This method computes the correction term:
  ///
  ///   For each bone (jointIndex, boneWeight, pos) from SkinningWeightIterator:
  ///     cross = (targetPosition - pos) × sourceNormal
  ///     For each rotation DOF d:
  ///       gradient += boneWeight * scale * rotationAxis[d] · cross
  ///
  /// WHERE pos is the skinned rest position from SkinningWeightIterator::next(),
  /// NOT jointState.translation(). This is derived from the scalar triple product
  /// identity applied to the full normal rotation derivative. See
  /// VERTEX_REFACTORING_PLAN.md "The Normal Rotation Correction" for the complete derivation.
  ///
  /// @param vertexIndex Mesh vertex index
  /// @param sourceNormal World-space source normal (from posed mesh)
  /// @param targetPosition Target position for the constraint (lever arm)
  /// @param scale Scalar multiplier (includes constraint weight, loss, and sourceNormalWeight)
  /// @param skeletonState Current skeleton state
  /// @param meshState Mesh state with restMesh_ for skinning iteration
  /// @param character Character with skinWeights
  /// @param gradient Output gradient vector (accumulated)
  void accumulateVertexNormalCorrectionGradient(
      size_t vertexIndex,
      const Eigen::Vector3<T>& sourceNormal,
      const Eigen::Vector3<T>& targetPosition,
      T scale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Jacobian version of accumulateVertexNormalCorrectionGradient.
  /// Accumulates into a single row of the Jacobian matrix.
  void accumulateVertexNormalCorrectionJacobian(
      size_t vertexIndex,
      const Eigen::Vector3<T>& sourceNormal,
      const Eigen::Vector3<T>& targetPosition,
      T scale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  /// Accumulates the gradient contribution from how mesh normals change with
  /// blend shape weights.
  ///
  /// For each blend shape parameter, this computes:
  ///   1. d(skinnedPos)/d(bsWeight) for each vertex of each adjacent face
  ///   2. d(edgeVectors)/d(bsWeight) from the vertex position derivatives
  ///   3. d(faceNormal)/d(bsWeight) via cross product derivative rule
  ///   4. Sum face normal derivatives over adjacent faces
  ///   5. Apply normalization derivative: (I - n̂n̂^T)/|n|
  ///   6. Accumulate: gradient += scale * leverArm · d(normal)/d(bsWeight)
  ///
  /// @param vertexIndex The vertex whose normal derivative is computed
  /// @param leverArm Vector that d(normal) is dotted with (typically pos - target)
  /// @param scale Weight multiplier (includes constraint weight, loss derivative,
  /// sourceNormalWeight)
  /// @param skeletonState Current skeleton state
  /// @param meshState Mesh state (must have posedMesh_ with vertices and faces)
  /// @param character Character with blend shapes and skin weights
  /// @param gradient Output gradient (accumulated)
  void accumulateVertexNormalBlendShapeGradient(
      size_t vertexIndex,
      const Eigen::Vector3<T>& leverArm,
      T scale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Overload with pre-built adjacency list to avoid O(numFaces) linear scan.
  ///
  /// @param adjacentFaces Pre-computed list of face indices adjacent to vertexIndex
  void accumulateVertexNormalBlendShapeGradient(
      size_t vertexIndex,
      const Eigen::Vector3<T>& leverArm,
      T scale,
      std::span<const size_t> adjacentFaces,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Jacobian version of accumulateVertexNormalBlendShapeGradient.
  /// Accumulates into a single row of the Jacobian matrix.
  void accumulateVertexNormalBlendShapeJacobian(
      size_t vertexIndex,
      const Eigen::Vector3<T>& leverArm,
      T scale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  /// Overload with pre-built adjacency list to avoid O(numFaces) linear scan.
  ///
  /// @param adjacentFaces Pre-computed list of face indices adjacent to vertexIndex
  void accumulateVertexNormalBlendShapeJacobian(
      size_t vertexIndex,
      const Eigen::Vector3<T>& leverArm,
      T scale,
      std::span<const size_t> adjacentFaces,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  /// Combined vertex gradient + normal rotation correction in a single skinning
  /// walk.
  ///
  /// This is EQUIVALENT to calling accumulateVertexGradient +
  /// accumulateVertexNormalCorrectionGradient separately, but does it in ONE pass
  /// through the skinning weights, which is ~2x faster.
  ///
  /// @tparam FuncDim Must be 1 for normal constraints
  /// @param vertexIndex Mesh vertex index
  /// @param worldPos World position of the vertex
  /// @param dfdv Derivative of f w.r.t. vertex position (1x3 for FuncDim=1 = normal^T)
  /// @param weightedResidual Weighted residual for standard vertex gradient
  /// @param sourceNormal Source mesh normal for normal correction
  /// @param targetPosition Target position for normal correction lever arm
  /// @param normalCorrectionScale Scale for normal correction (includes sourceNormalWeight)
  /// @param skeletonState Current skeleton state
  /// @param meshState Mesh state with restMesh_
  /// @param character Character with skinWeights, blendShape, faceExpressionBlendShape
  /// @param gradient Output gradient vector (accumulated)
  template <size_t FuncDim>
  void accumulateVertexGradientWithNormalCorrection(
      size_t vertexIndex,
      const Eigen::Vector3<T>& worldPos,
      const Eigen::Matrix<T, FuncDim, 3>& dfdv,
      const Eigen::Matrix<T, FuncDim, 1>& weightedResidual,
      const Eigen::Vector3<T>& sourceNormal,
      const Eigen::Vector3<T>& targetPosition,
      T normalCorrectionScale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Jacobian version of the combined vertex gradient + normal correction method.
  ///
  /// Equivalent to accumulateVertexJacobian + accumulateVertexNormalCorrectionJacobian
  /// in a single skinning walk.
  ///
  /// @tparam FuncDim Must be 1 for normal constraints
  template <size_t FuncDim>
  void accumulateVertexJacobianWithNormalCorrection(
      size_t vertexIndex,
      const Eigen::Vector3<T>& worldPos,
      const Eigen::Matrix<T, FuncDim, 3>& dfdv,
      T scale,
      const Eigen::Vector3<T>& sourceNormal,
      const Eigen::Vector3<T>& targetPosition,
      T normalCorrectionScale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  /// Scalar-specialized combined vertex gradient + normal rotation correction.
  ///
  /// Uses Vector3<T>.dot() instead of Matrix<1,3> * Vector3 for FuncDim=1
  /// optimization. Takes `normal` as Vector3 and `weightedResidual` as scalar T.
  ///
  /// @param vertexIndex Mesh vertex index
  /// @param worldPos World position of the vertex (unused, kept for API symmetry)
  /// @param normal Combined normal vector (sourceNormalWeight*source + targetNormalWeight*target)
  /// @param weightedResidual Scalar weighted residual for standard vertex gradient
  /// @param sourceNormal Source mesh normal for normal correction
  /// @param targetPosition Target position for normal correction lever arm
  /// @param normalCorrectionScale Scale for normal correction (includes sourceNormalWeight)
  /// @param skeletonState Current skeleton state
  /// @param meshState Mesh state with restMesh_
  /// @param character Character with skinWeights, blendShape, faceExpressionBlendShape
  /// @param gradient Output gradient vector (accumulated)
  void accumulateVertexGradientWithNormalCorrectionDirect(
      size_t vertexIndex,
      const Eigen::Vector3<T>& worldPos,
      const Eigen::Vector3<T>& normal,
      T weightedResidual,
      const Eigen::Vector3<T>& sourceNormal,
      const Eigen::Vector3<T>& targetPosition,
      T normalCorrectionScale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Scalar-specialized combined vertex Jacobian + normal rotation correction.
  ///
  /// Uses Vector3<T>.dot() instead of Matrix<1,3> * Vector3 for FuncDim=1
  /// optimization. Takes `normal` as Vector3 and `scale` as scalar T.
  ///
  /// @param vertexIndex Mesh vertex index
  /// @param worldPos World position of the vertex (unused, kept for API symmetry)
  /// @param normal Combined normal vector (sourceNormalWeight*source + targetNormalWeight*target)
  /// @param scale Weight multiplier (typically sqrt of loss derivative * constraint weight)
  /// @param sourceNormal Source mesh normal for normal correction
  /// @param targetPosition Target position for normal correction lever arm
  /// @param normalCorrectionScale Scale for normal correction (includes sourceNormalWeight)
  /// @param skeletonState Current skeleton state
  /// @param meshState Mesh state with restMesh_
  /// @param character Character with skinWeights, blendShape, faceExpressionBlendShape
  /// @param jacobian Output Jacobian matrix (accumulated)
  /// @param rowIndex Starting row in the Jacobian for this constraint
  void accumulateVertexJacobianWithNormalCorrectionDirect(
      size_t vertexIndex,
      const Eigen::Vector3<T>& worldPos,
      const Eigen::Vector3<T>& normal,
      T scale,
      const Eigen::Vector3<T>& sourceNormal,
      const Eigen::Vector3<T>& targetPosition,
      T normalCorrectionScale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  /// Accumulates gradient for a skinned mesh vertex.
  ///
  /// This method iterates over the skinning weights for a vertex and accumulates
  /// derivatives for each influencing bone. It handles:
  /// - Skinning weight iteration via SkinningWeightIteratorT
  /// - Per-bone hierarchy walk for gradient accumulation
  /// - Blend shape parameter derivatives
  /// - Face expression blend shape derivatives
  ///
  /// @tparam FuncDim Dimension of the residual function (1, 2, 3)
  /// @param vertexIndex Index of the mesh vertex
  /// @param worldPos World-space position of the vertex
  /// @param dfdv Derivative df/dv for the vertex (FuncDim x 3 matrix)
  /// @param weightedResidual Weighted residual (2 * weight * loss_deriv * f)
  /// @param skeletonState Current skeleton state
  /// @param meshState Current mesh state (must have restMesh_ set)
  /// @param character Character with skinWeights, blendShape, faceExpressionBlendShape
  /// @param gradient Output gradient vector (accumulated, not overwritten)
  template <size_t FuncDim>
  void accumulateVertexGradient(
      size_t vertexIndex,
      const Eigen::Vector3<T>& worldPos,
      const Eigen::Matrix<T, FuncDim, 3>& dfdv,
      const Eigen::Matrix<T, FuncDim, 1>& weightedResidual,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Specialized vertex gradient for identity dfdv (FuncDim=3, dfdv=I).
  ///
  /// When dfdv is the 3x3 identity matrix (e.g., VertexPositionConstraint), this
  /// avoids the `dfdv * derivative` matrix multiply in the inner loop, replacing it
  /// with a direct dot product: `weightedResidual.dot(derivative)`.
  ///
  /// @param vertexIndex Index of the mesh vertex
  /// @param weightedResidual Weighted residual (2 * weight * loss_deriv * f)
  /// @param skeletonState Current skeleton state
  /// @param meshState Current mesh state (must have restMesh_ set)
  /// @param character Character with skinWeights, blendShape, faceExpressionBlendShape
  /// @param gradient Output gradient vector (accumulated, not overwritten)
  void accumulateVertexGradientIdentityDfdv(
      size_t vertexIndex,
      const Eigen::Vector3<T>& weightedResidual,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Accumulates Jacobian for a skinned mesh vertex.
  ///
  /// Similar to accumulateVertexGradient, but accumulates into a Jacobian matrix.
  ///
  /// @tparam FuncDim Dimension of the residual function (1, 2, 3)
  /// @param vertexIndex Index of the mesh vertex
  /// @param worldPos World-space position of the vertex
  /// @param dfdv Derivative df/dv for the vertex (FuncDim x 3 matrix)
  /// @param scale Weight multiplier
  /// @param skeletonState Current skeleton state
  /// @param meshState Current mesh state (must have restMesh_ set)
  /// @param character Character with skinWeights, blendShape, faceExpressionBlendShape
  /// @param jacobian Output Jacobian matrix (accumulated, not overwritten)
  /// @param rowIndex Starting row in the Jacobian for this constraint
  template <size_t FuncDim>
  void accumulateVertexJacobian(
      size_t vertexIndex,
      const Eigen::Vector3<T>& worldPos,
      const Eigen::Matrix<T, FuncDim, 3>& dfdv,
      T scale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  /// Specialized vertex Jacobian for identity dfdv (FuncDim=3, dfdv=I).
  ///
  /// When dfdv is the 3x3 identity matrix (e.g., VertexPositionConstraint), this
  /// avoids the `dfdv * derivative` matrix multiply in the inner loop.
  ///
  /// @param vertexIndex Index of the mesh vertex
  /// @param scale Weight multiplier
  /// @param skeletonState Current skeleton state
  /// @param meshState Current mesh state (must have restMesh_ set)
  /// @param character Character with skinWeights, blendShape, faceExpressionBlendShape
  /// @param jacobian Output Jacobian matrix (accumulated, not overwritten)
  /// @param rowIndex Starting row in the Jacobian for this constraint
  void accumulateVertexJacobianIdentityDfdv(
      size_t vertexIndex,
      T scale,
      const SkeletonStateT<T>& skeletonState,
      const MeshStateT<T>& meshState,
      const Character& character,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

 private:
  const Skeleton& skeleton_;
  const ParameterTransform& parameterTransform_;
  const VectorX<bool>& activeJointParams_;
  const ParameterSet& enabledParameters_;
};

} // namespace momentum
