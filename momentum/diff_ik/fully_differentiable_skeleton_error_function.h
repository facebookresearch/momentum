/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/diff_ik/fwd.h>

#include <Eigen/SparseCore>
#include <vector>

namespace momentum {

// An error function which can compute the gradients wrt all its inputs.
// For example, a "normal" position error function can compute the gradient
// of the error wrt the solved modelParameters.  A "FullyDifferentiable"
// error function also knows how to compute the derivative of the gradient
// wrt the per-constraint weights, the per-constraint targets, etc.
//
// To make this work, we need a mechanism to specify which inputs we can
// differentiate wrt.  Thus the inputs()/getInput()/setInput() stuff below.
// This design is a bit "string-heavy" but ensures that we can use the various
// derivatives in code that doesn't know a priori which inputs a given
// constraint has.
//
// Note that the resulting inputs are all specified as Eigen::VectorXf.  This
// tends to mean that the resulting error functions are going to operate on
// big homogenous Tensor types instead of the mixed-struct types like
// MarkerErrorFunction uses.
template <typename T>
class FullyDifferentiableSkeletonErrorFunctionT {
 public:
  virtual ~FullyDifferentiableSkeletonErrorFunctionT() = default;

  [[nodiscard]] virtual const char* name() const = 0;

  // Get a list of all the differentiable inputs for this error function
  // (excluding the modelParameters).
  [[nodiscard]] virtual std::vector<std::string> inputs() const = 0;

  // Get the current input values.  A given input will be unrolled into a flat vector,
  // so, for example, if the input was an n x 3 vector of constraint targets, getInput
  // would return a (3n x 1) VectorXf with the targets packed (t1x t1y t1z t2x t2y t2z ...)
  [[nodiscard]] Eigen::VectorX<T> getInput(const std::string& name) const;
  void getInput(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> value) const;

  // Efficiently get the size of the input (e.g., getInput(name).size()).
  [[nodiscard]] virtual Eigen::Index getInputSize(const std::string& name) const = 0;

  // Set the current input values:
  void setInput(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value);

  // Compute the quantity
  //  d/dInput(dE/dTheta) . v
  // where dE/dTheta is the gradient of the energy computed in getGradient.
  // By applying the dot product inside this function we avoid dealing
  // with tensor types, and it is possible to compute very efficiently
  // in reverse-mode differentiation such as used by PyTorch.
  //
  // Note that because v is a constant here, we can rewrite this as
  //   d/dInput(dE/dTheta . v)
  // Now it's a lot easier to understand (it's the derivative wrt the given input of a scalar,
  // no tensors required) and very amenable to autodiff as described below.
  virtual Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) = 0;

 protected:
  virtual void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> value) const = 0;
  virtual void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) = 0;

  // Notes on implementing d_gradient_d_input_dot: After some experimentation, I have settled on the
  // following procedure for implementing the d_gradient_d_input_dot() function.  By using ceres
  // autodiff, we can compute the derivatives for all the inputs without having to hand-implement
  // them all.
  //
  // Note that I spent some time trying to convince ceres autodiff to compute both getGradient() and
  // d_gradient_d_input_dot() straight from the error function; while I still think that might be
  // possible it gets really nasty.  This is partly because Ceres isn't really engineered to deal
  // with second derivatives (you can do it with jets of jets but the tooling gets pretty poor) and
  // partly because getting it to respect the right sparsity patterns gets really complicated.
  //
  // 1. For a typical constraint like a projection constraint, the inputs are sparse; that is,
  //    a given input only affects a few rows of the residual.  For example: a projection constraint
  //    returns a (2n x 1) residual, but a given block of two is only affected by a single weight
  //    w_i, a single offset o_i, and a single target t_i.
  // 2. Split the gradient function out into a separate function for the ith term of the residual.
  // Example:
  //      void getGradient(...) {
  //        for (int i = 0; i < constraints.size(); ++i) {
  //          getConstraintGradient(constraints[i], gradient);
  //        }
  //      }
  //      void getContraintGradient(const Constraint& cons, Eigen::Ref<Eigen::VectorX<T>> gradient)
  //      {
  //         ...
  //      }
  //    See e.g. MarkerErrorFunction for an example.
  // 3. Copy the getConstraintGradient function and make the following changes:
  //      a. Make it template <typename T>, where T will be a ceres::Jet<float, N>.
  //      b. Turn it into getConstraintGradient_dot(..., Eigen::Ref<const Eigen::VectorX<T>> vec).
  //         You will be applying the dot of the computed gradient against the vec as you go, and
  //         the ceres::Jet types will accumulate the derivatives.
  //      c. Pass all the constraint inputs in explicitly, templated on T.  That is, the
  //         getConstraintGradient() function above becomes:
  //         template <typename T>
  //         void getConstraintGradient_dot(const T cons_weight,
  //                                        const Vector3<T> cons_target,
  //                                        Eigen::Ref<const Eigen::VectorX<T>> vec)
  //      d. Replace all the Vector3f/float with Vector3<T>/T or just use auto and .eval() a lot.
  //         Be wary of Eigen's use of temporaries.
  //              https://eigen.tuxfamily.org/dox/TopicPitfalls.html
  //         This is bad due to Eigen temporaries:
  //           auto foo = bar.cross(baz).normalized();
  //         This is okay:
  //           auto foo = bar.cross(baz).normalized().eval();
  //      e. Replace jointState.getRotationDerivative(...) with getRotationDerivative(jointState,
  //      ...) and
  //         jointState.getScaleDerivative(...) with getScaleDerivative(jointState, ...) from
  //         ceres_utility.h.
  //      f. Add the (grad . vec) dot product in the appropriate spot.  There is a function
  //         times_parameterTransform_times_v() in CeresUtility too apply the dot product if your
  //         function computes the gradient wrt the joint parameters instead of the model
  //         parameters.
  //             auto result = T();
  //             for (size_t d = 0; d < 3; d++) {
  //             {
  //                 const float grad_jointParam = ...
  //                 result += times_parametersTransform_times_v(grad_jointParam, jointIndex *
  //                 kParametersPerJoint + d, vec)
  //             }
  //
  // 4. Now, implement your d_gradient_d_input_dot() function.
  //       if(input == "target") {
  //           Eigen::VectorX<T> result(constraints.size()*3);
  //           for (int iCons = 0; iCons < constraints.size(); ++iCons)
  //           {
  //               typedef ceres::Jet<float,3> JetType;
  //               const auto& cons = constraints[iCons];
  //               // We don't actually care about the value of the dot product here, we only
  //               // care about the derivative, which is stored in the Jet dual number .v.
  //               auto dotProd = getConstraintGradient_dot<JetType>(JetType(cons.weight),
  //               buildJetVec(cons.target), vec); result.segment<3>(3*iCons) = dotProd.v;
  //           }
  //       }
  //    The tricky part is deciding what you're differentiating wrt to.  The variables that are
  //    constant can get converted to jets in this way:
  //        v.cast<JetType>(), JetType(c)
  //    The variables that we _are_ differentiating wrt need to be converted to unit jets, like
  //    this:
  //        buildJetVec(v), JetType(c, 0)
};

} // namespace momentum
