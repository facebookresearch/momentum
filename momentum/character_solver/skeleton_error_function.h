/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>

namespace momentum {

template <typename T>
class SkeletonErrorFunctionT {
 public:
  SkeletonErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt)
      : skeleton_(skel),
        parameterTransform_(pt),
        weight_(1.0),
        activeJointParams_(pt.activeJointParams) {
    enabledParameters_.flip(); // all parameters enabled by default
  }
  virtual ~SkeletonErrorFunctionT() = default;

  [[nodiscard]] const Skeleton& getSkeleton() const {
    return skeleton_;
  }

  [[nodiscard]] const ParameterTransform& getParameterTransform() const {
    return parameterTransform_;
  }

  /// For mesh-based error functions, get access to the underlying character.  Error functions
  /// that depend only on the skeleton can return nullptr.
  [[nodiscard]] virtual const Character* getCharacter() const {
    return nullptr;
  }

  void setWeight(T w) {
    weight_ = w;
  }

  [[nodiscard]] T getWeight() const {
    return weight_;
  }

  void setActiveJoints(const VectorX<bool>& aj) {
    activeJointParams_ = aj;
  }

  void setEnabledParameters(const ParameterSet& ps) {
    enabledParameters_ = ps;
  }

  virtual double getError(
      const ModelParametersT<T>& /* params */,
      const SkeletonStateT<T>& /* state */,
      const MeshStateT<T>& /* meshState */) {
    return 0.0f;
  }

  virtual double getGradient(
      const ModelParametersT<T>& /* params */,
      const SkeletonStateT<T>& /* state */,
      const MeshStateT<T>& /* meshState */,
      Eigen::Ref<Eigen::VectorX<T>> /* gradient */) {
    return 0.0f;
  }

  virtual double getJacobian(
      const ModelParametersT<T>& /* params */,
      const SkeletonStateT<T>& /* state */,
      const MeshStateT<T>& /* meshState */,
      Eigen::Ref<Eigen::MatrixX<T>> /* jacobian */,
      Eigen::Ref<Eigen::VectorX<T>> /* residual */,
      int& usedRows) {
    usedRows = 0;
    return 0.0f;
  }

  virtual void getHessian(
      const ModelParametersT<T>& /* params */,
      const SkeletonStateT<T>& /* state */,
      const MeshStateT<T>& /* meshState */,
      Eigen::Ref<Eigen::MatrixX<T>> /* hessian */) {
    throw;
  }

  virtual double getSolverDerivatives(
      const ModelParametersT<T>& parameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const size_t actualParameters,
      Eigen::MatrixX<T>& hess,
      Eigen::VectorX<T>& grad) {
    const int paramSize = static_cast<int>(parameters.size());
    double error = 0.0;

    if (useHessian_) {
      // get some ice cream, you deserve it
      error = getGradient(parameters, state, meshState, grad);
      getHessian(parameters, state, meshState, hess);
    } else {
      const int residualSize = static_cast<int>(getJacobianSize());

      Eigen::MatrixX<T> jacobian = Eigen::MatrixX<T>::Zero(residualSize, paramSize);
      Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(residualSize);
      int rows = 0;
      error = getJacobian(parameters, state, meshState, jacobian, residual, rows);

      // Update JtJ
      if (rows > 0) {
        // ! In truth, on the the "actualParameters" leftmost block will be used
        // We take advantage of this here and skip the other computations
        const auto JtBlock2 =
            (jacobian.topLeftCorner(rows, actualParameters).transpose() * 2.0).eval();

        // Update Hessian (Hessian += 2.0 * J^T * J) using selfadjointView with rankUpdate,
        // replacing triangularView
        hess.template selfadjointView<Eigen::Lower>().rankUpdate(JtBlock2);

        // Update JtR
        grad.noalias() += JtBlock2 * residual.head(rows);
      }
    }

    return error;
  }

  [[nodiscard]] virtual size_t getJacobianSize() const {
    return 0;
  }

  [[nodiscard]] virtual bool needsMesh() const {
    return false;
  }

 protected:
  const Skeleton& skeleton_;
  const ParameterTransform& parameterTransform_;
  T weight_;
  VectorX<bool> activeJointParams_;
  ParameterSet enabledParameters_; // set to zero by default constr
  bool useHessian_{false}; // this can be true only if the getHessian is implemented
};

} // namespace momentum
