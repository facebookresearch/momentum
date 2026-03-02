/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_derivative.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/generalized_loss.h>

#include <Eigen/Core>

#include <span>
#include <variant>
#include <vector>

namespace momentum {

/// Describes how a world-space vector contributes to the derivative computation.
///
/// Each contribution represents a world-space vector (point or direction) that participates
/// in the chain rule for gradient/Jacobian computation. The base class computes the world-space
/// position of each contribution and handles the derivative accumulation automatically.
///
/// @section Contribution Types
/// - **JointPoint**: A point attached to a joint (w=1 in homogeneous coords).
///   Translation, rotation, and scale all affect the derivative.
/// - **JointDirection**: A direction attached to a joint (w=0 in homogeneous coords).
///   Only rotation affects the derivative.
/// - **Vertex**: A mesh vertex (position only, blend shapes handled separately).
///
/// @tparam T Scalar type (float or double)
template <typename T>
struct SourceT {
  /// A point attached to a joint, transformed by the joint's full transformation.
  /// Use for constraint positions like end-effector positions.
  struct JointPoint {
    size_t jointIndex;
    Eigen::Vector3<T> localOffset;
  };

  /// A direction attached to a joint, rotated by the joint's rotation only.
  /// Use for constraint directions like orientation axes.
  struct JointDirection {
    size_t jointIndex;
    Eigen::Vector3<T> localDirection;
  };

  /// A mesh vertex, deformed by skinning and blend shapes.
  /// Use for vertex-based constraints like vertex-to-point distance.
  struct Vertex {
    size_t vertexIndex;
  };

  /// A mesh vertex with parameter-dependent normal.
  /// Use for constraints where the mesh normal direction matters
  /// (Normal, SymmetricNormal constraint types).
  /// Superset of Vertex: handles position + blend shapes + face expressions + normal rotation
  /// correction. sourceNormalWeight controls how much normal rotation correction to apply:
  /// - 1.0 for Normal (pure source normal)
  /// - 0.5 for SymmetricNormal (50/50 mix)
  /// - 0.0 is equivalent to Vertex (no normal correction)
  struct VertexNormal {
    size_t vertexIndex;
    Eigen::Vector3<T> targetPosition;
    T sourceNormalWeight;
  };

  std::variant<JointPoint, JointDirection, Vertex, VertexNormal> data;

  /// Returns the joint index if this is a joint-based contribution, or kInvalidIndex for vertices.
  [[nodiscard]] size_t getJointIndex() const;

  /// Returns true if this contribution is a point (affects translation derivatives).
  [[nodiscard]] bool isPoint() const;

  /// Returns true if this contribution is a vertex.
  [[nodiscard]] bool isVertex() const;

  /// Returns true if this contribution is a vertex normal.
  [[nodiscard]] bool isVertexNormal() const;

  /// Creates a JointPoint contribution.
  [[nodiscard]] static SourceT jointPoint(size_t jointIndex, const Eigen::Vector3<T>& offset);

  /// Creates a JointDirection contribution.
  [[nodiscard]] static SourceT jointDirection(
      size_t jointIndex,
      const Eigen::Vector3<T>& direction);

  /// Creates a Vertex contribution.
  [[nodiscard]] static SourceT vertex(size_t vertexIndex);

  /// Creates a VertexNormal contribution.
  [[nodiscard]] static SourceT
  vertexNormal(size_t vertexIndex, const Eigen::Vector3<T>& targetPosition, T sourceNormalWeight);
};

/// Base class for error functions with flexible contribution patterns.
///
/// MultiSourceErrorFunctionT provides a unified framework for error functions that can
/// have arbitrary combinations of joint points, joint directions, and mesh vertices
/// contributing to each constraint's residual.
///
/// @section Architecture
/// This is Level 1 of the three-level hierarchy:
/// - **Level 1**: MultiSourceErrorFunctionT (this class) - handles ANY contribution combination
/// - **Level 2**: Mid-level adapters (JointErrorFunctionT, VertexErrorFunctionT)
/// - **Level 3**: Leaf classes that implement evalFunction()
///
/// @section Usage
/// Derived classes must implement:
/// 1. `getContributions(constrIndex)` - returns the list of contributions for a constraint
/// 2. `evalFunction(constrIndex, state, meshState, worldVecs, f, dfdv)` - computes residual and
/// derivatives
///
/// The base class handles:
/// - Computing world-space vectors from contributions
/// - Batching contributions by joint for efficient hierarchy walks
/// - Accumulating derivatives via SkeletonDerivativeT
///
/// @section Example
/// @code
/// // A simple two-joint distance constraint
/// class JointDistanceErrorT : public MultiSourceErrorFunctionT<T, 1> {
///   std::span<const SourceT<T>> getContributions(size_t i) const override {
///     return contributions_[i];  // 2 JointPoint contributions
///   }
///
///   void evalFunction(
///       size_t i, const SkeletonStateT<T>&, const MeshStateT<T>&,
///       std::span<const Vector3<T>> worldVecs,
///       FuncType& f, std::span<DfdvType> dfdv) const override {
///     Vector3<T> diff = worldVecs[0] - worldVecs[1];
///     f(0) = diff.norm() - targetDistance_[i];
///     Vector3<T> n = diff.normalized();
///     dfdv[0] = n.transpose();
///     dfdv[1] = -n.transpose();
///   }
/// };
/// @endcode
///
/// @tparam T Scalar type (float or double)
/// @tparam FuncDim Dimension of the residual function (1, 2, 3, or 9)
template <typename T, size_t FuncDim>
class MultiSourceErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  using FuncType = Eigen::Matrix<T, FuncDim, 1>;
  using DfdvType = Eigen::Matrix<T, FuncDim, 3>;

  /// Constructs the error function.
  ///
  /// @param skeleton The skeleton structure
  /// @param parameterTransform Maps joint parameters to model parameters
  /// @param lossAlpha Alpha parameter for the generalized loss function (default: L2)
  /// @param lossC C parameter for the generalized loss function (default: 1)
  MultiSourceErrorFunctionT(
      const Skeleton& skeleton,
      const ParameterTransform& parameterTransform,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  ~MultiSourceErrorFunctionT() override = default;

  /// Returns the contributions for a specific constraint.
  ///
  /// Derived classes must implement this to return the list of world-space vectors
  /// that contribute to the constraint's residual. The base class will compute
  /// the world positions and handle derivative accumulation.
  ///
  /// @param constrIndex Index of the constraint
  /// @return Span of contributions for this constraint
  [[nodiscard]] virtual std::span<const SourceT<T>> getContributions(size_t constrIndex) const = 0;

  /// Computes the residual and per-contribution derivatives.
  ///
  /// Derived classes implement this with their constraint-specific logic.
  /// The world-space vectors are pre-computed by the base class.
  ///
  /// @param constrIndex Index of the constraint
  /// @param state Current skeleton state (available if needed for extra data)
  /// @param meshState Current mesh state (available if needed for extra data)
  /// @param worldVecs Pre-computed world-space vectors for each contribution
  /// @param f Output: residual vector (FuncDim x 1)
  /// @param dfdv Output: derivative df/dv for each contribution (must be filled if non-empty)
  virtual void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const = 0;

  /// Returns the number of constraints.
  [[nodiscard]] virtual size_t getNumConstraints() const = 0;

  /// Returns the per-constraint weight for a specific constraint.
  ///
  /// Subclasses implement this to provide per-constraint weighting.
  /// The total weight applied to each constraint is: getConstraintWeight(i) * this->weight_
  ///
  /// @param constrIndex Index of the constraint
  /// @return Per-constraint weight (default implementations should return 1.0)
  [[nodiscard]] virtual T getConstraintWeight(size_t constrIndex) const = 0;

  /// Returns true if any constraint has vertex contributions.
  [[nodiscard]] bool needsMesh() const override;

  /// Returns the Jacobian size (number of residuals).
  [[nodiscard]] size_t getJacobianSize() const override;

  /// Returns the Character, if available (needed for vertex contributions).
  /// Subclasses with vertex contributions should override this.
  [[nodiscard]] const Character* getCharacter() const override {
    return nullptr;
  }

  /// Computes the total weighted error.
  double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) override;

  /// Computes the gradient and returns the weighted error.
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  /// Computes the Jacobian and residual, returns the weighted error.
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

 protected:
  SkeletonDerivativeT<T> skeletonDerivative_;
  /// Generalized loss function (default: L2)
  const GeneralizedLossT<T> loss_;
  /// Intermediate storage for gradient accumulation across constraints
  VectorX<T> jointGrad_;
  /// Maximum threads for parallel processing (0 = single-threaded)
  uint32_t maxThreads_ = 0;

 public:
  /// Set the maximum number of threads for parallel gradient/Jacobian computation.
  /// 0 = single-threaded (default).
  void setMaxThreads(uint32_t maxThreads) {
    maxThreads_ = maxThreads;
  }

 private:
  /// Computes world-space vectors for all contributions of a constraint.
  void computeWorldVectors(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::vector<Eigen::Vector3<T>>& worldVecs) const;

  /// Accumulates gradient for all contributions of a constraint.
  void accumulateGradient(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      std::span<const DfdvType> dfdvs,
      const FuncType& weightedResidual,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Accumulates Jacobian for all contributions of a constraint.
  void accumulateJacobian(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      std::span<const DfdvType> dfdvs,
      T scale,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      size_t rowIndex) const;

  // Mutable for lazy caching in const methods. Requires external synchronization for thread safety.
  mutable bool needsMeshCached_ = false;
  mutable bool needsMeshCacheValid_ = false;
};

} // namespace momentum
