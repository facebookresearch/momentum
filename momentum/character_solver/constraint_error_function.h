/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/generalized_loss.h>
#include <momentum/math/types.h>

#include <optional>

namespace momentum {

/// Base structure of constraint data
struct ConstraintData {
  /// Parent joint index this constraint is under
  size_t parent = kInvalidIndex;
  /// Weight of the constraint
  float weight = 0.0;
  /// Name of the constraint
  std::string name = {};

  ConstraintData(size_t pIndex, float w, const std::string& n = "")
      : parent(pIndex), weight(w), name(n) {}
};

/// A list of ConstraintData
using ConstraintDataList = std::vector<ConstraintData>;

/// An optional of a reference. Because optional<T&> is invalid, we need to use reference_wrapper to
/// make the reference of T as an optional.
template <typename T>
using optional_ref = std::optional<std::reference_wrapper<T>>;

/// The ConstraintErrorFunction is a base class of a general form of constraint errors l = w *
/// loss(f^2), where w is the weight (could be a product of different weighting terms), loss() is
/// the generalized loss function (see math/generalized_loss.h), and f is a difference vector we
/// want to minimize.
///
/// f takes the form of f(v, target), where v = T(q)*source. T is the global transformation of the
/// parent joint of the source, and target is the desired value of source in global space. f
/// computes the differences between v and target.
///
/// Based on the above, we have
/// Jacobian: df/dq = df/dv * dv/dT * dT/dq, and
/// Gradient: dl/dq = dl/df * Jac
/// Both dl/df and dT/dq are boiler plate code that we can implement in the base class, so a derived
/// class only needs to implement f and df/dT. However, dT/dq is not efficient to compute, and
/// Momentum instead implements dv/dq, for any 3-vector v. Therefore, we will compute df/dq = df/dv
/// * dv/dq, and implement dv/dq in the base class. So a derived class now needs to implement f and
/// df/dv.
///
/// This should work for a point (eg. PositionErrorFunction), or an axis (eg.
/// FixedAxisErrorFunction), but it can also work for a rotation matrix, or a 3x4 transformation
/// matrix, by applying the transformation one axis/point at a time.The number of 3-vectors to be
/// transformed in a constraint is NumVec.
template <
    typename T, // float or double
    class Data, // derived types from ConstraintData
    size_t FuncDim = 3, // dimension of f
    size_t NumVec =
        1, // how many 3-vector v in one constraint, eg. a point is 1, and a rotation matrix is 3
    size_t NumPos =
        1> // we assume a constraint can be a function of both points and axes, and points come
           // before axes in the NumVec of "v"s. This specifies how many "v"s are points. For
           // example, it's 1 for a point constraint, and 0 for a rotation matrix.
class ConstraintErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  static constexpr size_t kFuncDim = FuncDim;
  static constexpr size_t kNumVec = NumVec;
  static constexpr size_t kNumPos = NumPos;

  using FuncType = Vector<T, FuncDim>; // vector type for f
  using VType = Vector3<T>; // vector type for v
  using DfdvType = Eigen::Matrix<T, FuncDim, 3>; // type for dfdv - it's effectively a vector if f
                                                 // is a scalar (FuncDim=1, eg. PlaneErrorFunction)
  /// Constructor
  ///
  /// @param[in] skel: character skeleton
  /// @param[in] pt: parameter transformation
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit ConstraintErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  /// A convenience constructor where character contains info of the skeleton and parameter
  /// transform.
  ///
  /// @param[in] character: character definition
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit ConstraintErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>(
            character.skeleton,
            character.parameterTransform,
            lossAlpha,
            lossC) {}

  // The functions below should just work for most constraints. When we have new constraints that
  // don't fit these implementations, we can remove the "final" annotation to allow override.

  /// Computes the error function value l = w * loss(f^2). It gets f from the derived class, and
  /// implements the rest.
  ///
  /// @param[in] params: current model parameters
  /// @param[in] state: curren global skeleton joint states computed from the model parameters
  /// @param[in] meshState: current mesh state (unused by this base class)
  ///
  /// @return the error function value l
  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */) final;

  /// The gradient of the error function: dl/dq = dl/d[f^2] * 2f * df/dv * dv/dq. It gets df/dv from
  /// the derived class, and implements the rest.
  ///
  /// @param[in] params: current model parameters
  /// @param[in] state: curren global skeleton joint states computed from the model parameters
  /// @param[in] meshState: current mesh state (unused by this base class)
  /// @param[out] gradient: the gradient vector to accumulate into
  ///
  /// @return the error function value l
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Ref<VectorX<T>> gradient) final;

  /// For least-square problems, we assume l is the square of a vector function F. The jacobian is
  /// then dF/dq. (A factor 2 is implemented in the solver.) With l2 loss, we have F = sqrt(w) * f,
  /// and the jacobian is sqrt(w) * df/dv * dv/dq. With the generalized loss, the jacobian becomes
  /// sqrt(w * d[loss]/d[f^2]) * df/dv * dv/dq. It gets df/dv from the derived class, and implements
  /// the rest.
  ///
  /// @param[in] params: current model parameters
  /// @param[in] state: curren global skeleton joint states computed from the model parameters
  /// @param[in] meshState: current mesh state (unused by this base class)
  /// @param[out] jacobian: the output jacobian matrix
  /// @param[out] residual: the output function residual (ie. f scaled by the loss gradient)
  /// @param[out] usedRows: number of rows in the jacobian/residual used by this error function
  ///
  /// @return the error function l
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Ref<MatrixX<T>> jacobian,
      Ref<VectorX<T>> residual,
      int& usedRows) final;

  /// The number of rows in the jacobian is the dimension of f, FuncDim, times the number of
  /// constraints.
  ///
  /// @return number of rows in the jacobian
  [[nodiscard]] size_t getJacobianSize() const final;

  /// Adds a constraint to the list
  ///
  /// @param[in] constr: the constraint to be added
  void addConstraint(const Data& constr) {
    constraints_.push_back(constr);
  }

  /// Appends a list of constraints
  ///
  /// @param[in] constrs: a list of constraints to be added
  void addConstraints(std::span<const Data> constrs) {
    constraints_.insert(constraints_.end(), constrs.begin(), constrs.end());
  }

  /// Replace the current list of constraints with the input
  ///
  /// @param[in] constrs: the new list of constraints
  void setConstraints(std::span<const Data> constrs) {
    constraints_.assign(constrs.begin(), constrs.end());
  }

  /// @return the current list of constraints immutable
  [[nodiscard]] const std::vector<Data>& getConstraints() const {
    return constraints_;
  }

  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

  /// Clear the current list of constraints
  void clearConstraints() {
    constraints_.clear();
  }

 protected:
  /// List of constraints
  std::vector<Data> constraints_;
  /// The generalized loss function that transforms f^2
  const GeneralizedLossT<T> loss_;
  /// Intermediate storage of the gradient from this error function. We can allocate the space in
  /// the constructor to save some dynamic allocation.
  VectorX<T> jointGrad_;

  /// The only function a derived class needs to implement.
  /// f is needed both for errors and derivatives, v and dfdv are needed only for computing
  /// derivatives, and therefore optional. An implementation should check if they are provided
  /// before computing and setting their values. And don't assume they are zero-initialized. Each
  /// returned v and df/dv corresponds to a source 3-vector (eg. a position constraint, or each of
  /// the three axis of a rotation constraint).
  ///
  /// @param[in] constrIndex: index of the constraint to evaluate
  /// @param[in] state: JointState of the parent joint with transformation T
  /// @param[out] f: output the value of f of dimension FuncDim
  /// @param[out] v: if valid, output the vector v=T*source; there could be NumVec of vs
  /// @param[out] dfdv: if valid, output the matrix df/dv of dimension FuncDim x 3 per v
  virtual void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      FuncType& f,
      optional_ref<std::array<VType, NumVec>> v = {},
      optional_ref<std::array<DfdvType, NumVec>> dfdv = {}) const = 0;

 private:
  double getJacobianForSingleConstraint(
      const JointStateListT<T>& jointStates,
      size_t iConstr,
      Ref<Eigen::MatrixX<T>> jacobian,
      Ref<Eigen::VectorX<T>> residual);

  double getGradientForSingleConstraint(
      const JointStateListT<T>& jointStates,
      size_t iConstr,
      Ref<VectorX<T>> gradient);
};

} // namespace momentum

#include "momentum/character_solver/constraint_error_function-inl.h"
