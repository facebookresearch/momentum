/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character_solver/error_function_types.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_derivative.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/generalized_loss.h>

#include <Eigen/Core>

#include <span>
#include <string>
#include <vector>

namespace momentum {

/// Base structure for vertex constraint data.
///
/// Derived constraint types extend this with their specific target data.
struct VertexConstraintData {
  /// Vertex index in the mesh
  size_t vertexIndex = kInvalidIndex;

  /// Weight of the constraint
  float weight = 0.0f;

  /// Name of the constraint (for debugging)
  std::string name;

  VertexConstraintData() = default;
  VertexConstraintData(size_t vIndex, float w, const std::string& n = "")
      : vertexIndex(vIndex), weight(w), name(n) {}
};

/// Mid-level adapter for single-vertex constraints.
///
/// VertexErrorFunctionT handles constraints involving a single mesh vertex.
/// It inherits from SkeletonErrorFunctionT directly and uses SkeletonDerivativeT
/// for derivative computation through skinning weights.
///
/// Derived classes implement evalFunction() with vertex-specific logic.
/// The base class handles:
/// - Computing world-space vertex position from mesh state
/// - Skinning weight-based derivative chain rule via SkeletonDerivativeT
/// - Threading via dispenso
/// - Generalized loss function
///
/// @tparam T Scalar type (float or double)
/// @tparam Data Derived constraint data type (must extend VertexConstraintData)
/// @tparam FuncDim Dimension of the residual (1, 2, 3, or 9)
template <typename T, class Data, size_t FuncDim = 3>
class VertexErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  using FuncType = Eigen::Matrix<T, FuncDim, 1>;
  using DfdvType = Eigen::Matrix<T, FuncDim, 3>;

  static constexpr size_t kFuncDim = FuncDim;

  /// Default constant weight for backwards compatibility with the legacy VertexErrorFunctionT.
  /// Position/Normal/Plane leaf classes set this to 1e-4 (matching
  /// PositionErrorFunctionT::kLegacyWeight). Defaults to 1 (no scaling) for other leaf classes.
  static constexpr T kLegacyWeight = T(1e-4);

  /// Constructor.
  ///
  /// @param character The character (needed for mesh access)
  /// @param parameterTransform Maps joint parameters to model parameters
  /// @param lossAlpha Alpha parameter for the generalized loss function (default: L2)
  /// @param lossC C parameter for the generalized loss function (default: 1)
  explicit VertexErrorFunctionT(
      const Character& character,
      const ParameterTransform& parameterTransform,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  ~VertexErrorFunctionT() override = default;

  /// Returns the character (needed for mesh-based computations).
  [[nodiscard]] const Character* getCharacter() const final;

  /// Returns true (vertex constraints always need mesh).
  [[nodiscard]] bool needsMesh() const final;

  /// Adds a constraint.
  void addConstraint(const Data& constraint);

  /// Appends a list of constraints.
  void addConstraints(std::span<const Data> constraints) {
    for (const auto& c : constraints) {
      addConstraint(c);
    }
  }

  /// Sets all constraints, replacing existing ones.
  void setConstraints(std::span<const Data> constraints);

  /// Clears all constraints.
  void clearConstraints();

  /// Returns the current constraints.
  [[nodiscard]] const std::vector<Data>& getConstraints() const;

  /// Returns the number of constraints.
  [[nodiscard]] size_t getNumConstraints() const {
    return constraints_.size();
  }

  /// Returns the Jacobian size.
  [[nodiscard]] size_t getJacobianSize() const final {
    return FuncDim * constraints_.size();
  }

  /// Sets the maximum number of threads for parallel execution.
  void setMaxThreads(size_t maxThreads) {
    maxThreads_ = maxThreads;
  }

  /// Computes the error.
  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) override;

  /// Computes the gradient.
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  /// Computes the Jacobian.
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

  /// Computes the residual and derivatives.
  ///
  /// Derived classes implement this with their specific constraint logic.
  /// worldVecs[0] is the pre-computed vertex world position.
  ///
  /// @param constrIndex Index of the constraint
  /// @param state Current skeleton state
  /// @param meshState Current mesh state
  /// @param worldVecs Pre-computed world position (size 1 for vertex constraints)
  /// @param f Output: residual (FuncDim x 1)
  /// @param dfdv Output: derivative df/dv (must be filled if non-empty)
  virtual void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const = 0;

 protected:
  const Character& character_;
  std::vector<Data> constraints_;
  GeneralizedLossT<T> loss_;
  size_t maxThreads_ = 0;

  /// Legacy weight multiplied into error/gradient/Jacobian for backwards compatibility.
  /// Leaf classes that replace the old VertexErrorFunctionT set this to kLegacyWeight (1e-4).
  T legacyWeight_ = T(1);
};

} // namespace momentum
