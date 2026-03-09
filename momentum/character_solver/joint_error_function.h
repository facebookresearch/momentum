/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/skeleton_derivative.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/generalized_loss.h>
#include <momentum/math/types.h>

#include <optional>
#include <span>

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

/// The JointErrorFunction is a base class for constraint errors of the form l = w * loss(f^2),
/// where w is the weight (could be a product of different weighting terms), loss() is the
/// generalized loss function (see math/generalized_loss.h), and f is a difference vector we want to
/// minimize.
///
/// f takes the form of f(v, target), where v = T(q)*source. T is the global transformation of the
/// parent joint of the source, and target is the desired value of source in global space. f
/// computes the differences between v and target.
///
/// Based on the above, we have
/// Jacobian: df/dq = df/dv * dv/dT * dT/dq, and
/// Gradient: dl/dq = dl/df * Jac
///
/// The derivative computation (hierarchy walk from joint to root) is delegated to
/// SkeletonDerivativeT, keeping the per-constraint logic minimal. Derived classes implement
/// evalFunction() to define the constraint residual and its derivatives.
///
/// This should work for a point (eg. PositionErrorFunction), or an axis (eg.
/// FixedAxisErrorFunction), but it can also work for a rotation matrix, or a 3x4 transformation
/// matrix, by applying the transformation one axis/point at a time. The number of 3-vectors to be
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
class JointErrorFunctionT : public SkeletonErrorFunctionT<T> {
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
  explicit JointErrorFunctionT(
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
  explicit JointErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>(
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
  /// @param[in] state: current global skeleton joint states computed from the model parameters
  /// @param[in] meshState: current mesh state (unused by this base class)
  ///
  /// @return the error function value l
  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  /// The gradient of the error function: dl/dq = dl/d[f^2] * 2f * df/dv * dv/dq. It gets df/dv
  /// from the derived class, and implements the rest.
  ///
  /// @param[in] params: current model parameters
  /// @param[in] state: current global skeleton joint states computed from the model parameters
  /// @param[in] meshState: current mesh state (unused by this base class)
  /// @param[out] gradient: the gradient vector to accumulate into
  ///
  /// @return the error function value l
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<VectorX<T>> gradient) final;

  /// For least-square problems, we assume l is the square of a vector function F. The jacobian is
  /// then dF/dq. (A factor 2 is implemented in the solver.) With l2 loss, we have F = sqrt(w) * f,
  /// and the jacobian is sqrt(w) * df/dv * dv/dq. With the generalized loss, the jacobian becomes
  /// sqrt(w * d[loss]/d[f^2]) * df/dv * dv/dq. It gets df/dv from the derived class, and
  /// implements the rest.
  ///
  /// @param[in] params: current model parameters
  /// @param[in] state: current global skeleton joint states computed from the model parameters
  /// @param[in] meshState: current mesh state (unused by this base class)
  /// @param[out] jacobian: the output jacobian matrix
  /// @param[out] residual: the output function residual (ie. f scaled by the loss gradient)
  /// @param[out] usedRows: number of rows in the jacobian/residual used by this error function
  ///
  /// @return the error function l
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
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

  /// Evaluate the constraint residual and derivatives for a single constraint.
  ///
  /// The leaf class computes the world-space vectors v, the residual f, and the
  /// derivatives dfdv from the constraint data and the parent joint state.
  /// The first NumPos entries of v are points (w=1), the rest are directions (w=0).
  ///
  /// @param[in] constrIndex: index of the constraint
  /// @param[in] state: the joint state of the constraint's parent joint
  /// @param[out] f: output residual vector of dimension FuncDim
  /// @param[out] v: output world-space vectors (NumVec entries)
  /// @param[out] dfdv: output df/dv matrices (one per world vector)
  virtual void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      FuncType& f,
      std::array<VType, NumVec>& v,
      std::array<DfdvType, NumVec>& dfdv) const = 0;
};

} // namespace momentum

#include "momentum/character_solver/joint_error_function-inl.h"
