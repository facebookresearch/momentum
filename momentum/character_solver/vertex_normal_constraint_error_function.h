/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/vertex_constraint_error_function.h>

namespace momentum {

/// Constraint data for a vertex normal constraint.
///
/// Constrains a mesh vertex using a point-to-plane distance where the plane normal
/// can be the source (mesh) normal, the target normal, or a mix of both.
template <typename T>
struct VertexNormalConstraintDataT : public VertexConstraintData {
  /// Target position in world space
  Eigen::Vector3<T> targetPosition = Eigen::Vector3<T>::Zero();

  /// Target normal direction (will be flipped if facing away from source normal)
  Eigen::Vector3<T> targetNormal = Eigen::Vector3<T>::Zero();

  VertexNormalConstraintDataT() = default;
  VertexNormalConstraintDataT(
      size_t vIndex,
      const Eigen::Vector3<T>& targetPos,
      const Eigen::Vector3<T>& targetNorm,
      float w,
      const std::string& n = "")
      : VertexConstraintData(vIndex, w, n), targetPosition(targetPos), targetNormal(targetNorm) {}
};

/// Error function for vertex Normal and SymmetricNormal constraints.
///
/// Computes point-to-plane distance using a mix of source (mesh) and target normals:
///   f = (sW * sourceNormal + tW * targetNormal) · (vertexPos - targetPosition)
///
/// The sourceNormalWeight and targetNormalWeight control the mix:
/// - Normal type: sW=1, tW=0 (pure source normal)
/// - SymmetricNormal type: sW=0.5, tW=0.5 (50/50 mix)
///
/// The source normal rotates with the mesh, which requires special derivative handling.
/// This is handled automatically by the normal correction methods in SkeletonDerivativeT.
///
/// @tparam T Scalar type (float or double)
template <typename T>
class VertexNormalConstraintErrorFunctionT
    : public VertexConstraintErrorFunctionT<T, VertexNormalConstraintDataT<T>, 1> {
 public:
  using Base = VertexConstraintErrorFunctionT<T, VertexNormalConstraintDataT<T>, 1>;
  using typename Base::DfdvType;
  using typename Base::FuncType;

  /// Constructor.
  ///
  /// @param character The character (needed for mesh access)
  /// @param parameterTransform Maps joint parameters to model parameters
  /// @param sourceNormalWeight Weight for source (mesh) normal (1.0 for Normal, 0.5 for
  /// SymmetricNormal)
  /// @param targetNormalWeight Weight for target normal (0.0 for Normal, 0.5 for SymmetricNormal)
  /// @param lossAlpha Alpha parameter for the generalized loss function (default: L2)
  /// @param lossC C parameter for the generalized loss function (default: 1)
  /// @param computeAccurateNormalDerivatives If true, compute expensive
  /// d(normal)/d(blendShapeWeight)
  ///   terms in the gradient and Jacobian. Default false for better performance.
  explicit VertexNormalConstraintErrorFunctionT(
      const Character& character,
      const ParameterTransform& parameterTransform,
      T sourceNormalWeight = T(1),
      T targetNormalWeight = T(0),
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1),
      bool computeAccurateNormalDerivatives = false);

  ~VertexNormalConstraintErrorFunctionT() override = default;

  void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const final;

  /// Override to handle normal rotation correction terms via SkeletonDerivativeT.
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  /// Override to handle normal rotation correction terms via SkeletonDerivativeT.
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

  /// Enable or disable the expensive d(normal)/d(blendShapeWeight) computation.
  void setComputeAccurateNormalDerivatives(bool enable) {
    computeAccurateNormalDerivatives_ = enable;
  }

  /// Returns whether accurate normal blend shape derivatives are enabled.
  [[nodiscard]] bool computeAccurateNormalDerivatives() const {
    return computeAccurateNormalDerivatives_;
  }

 private:
  /// Build vertex-to-face adjacency map for all mesh vertices.
  /// Called once and cached for reuse across constraints.
  void buildVertexToFaceAdjacency(const MeshStateT<T>& meshState) const;

  T sourceNormalWeight_;
  T targetNormalWeight_;
  bool computeAccurateNormalDerivatives_ = false;

  /// Cached vertex-to-face adjacency for accurate normal derivatives.
  /// vertexToFaceAdjacency_[v] = list of face indices adjacent to vertex v.
  mutable std::vector<std::vector<size_t>> vertexToFaceAdjacency_;
  mutable bool vertexToFaceAdjacencyValid_ = false;
};

} // namespace momentum
