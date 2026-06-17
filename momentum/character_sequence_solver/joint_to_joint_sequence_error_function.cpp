/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/joint_to_joint_sequence_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"

namespace momentum {

namespace {

// Express the source joint in the reference joint's frame (T_ref^{-1} * T_src).
// transformAtoB combines transforms along the common-ancestor path, which is
// more numerically stable than computing the inverse-and-multiply directly.
template <typename T>
TransformT<T> relativeTransform(
    const Skeleton& skeleton,
    const SkeletonStateT<T>& state,
    size_t sourceJoint,
    size_t referenceJoint) {
  return transformAtoB(sourceJoint, referenceJoint, skeleton, state);
}

// Read-only references shared by every chain-walk helper below. Bundling them
// keeps the helper signatures (and the call sites in getGradient/getJacobian)
// short, which also keeps the per-function cyclomatic complexity in check.
template <typename T>
struct ChainContext {
  const Skeleton& skeleton;
  const VectorX<bool>& activeJointParams;
  const ParameterTransform& parameterTransform;
};

// ---------------------------------------------------------------------------
// Gradient chain-walk helpers. Each walks the joint parent chain from
// leafJoint up to (but not including) stopJoint, accumulating into the
// model-parameter gradient segment [gradOffset, gradOffset + nParam).
// ---------------------------------------------------------------------------

// Accumulate gradient for the translation column (col 3 of T_rel).
// Handles translation, rotation, and scale derivatives.
template <typename T>
void accumTransGrad(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const SkeletonStateT<T>& state,
    const Vector3<T>& worldPos,
    const Vector3<T>& wg,
    T wgt,
    Eigen::Ref<Eigen::VectorX<T>> gradient,
    Eigen::Index gradOffset,
    Eigen::Index nParam) {
  Eigen::Ref<Eigen::VectorX<T>> gradientSegment = gradient.segment(gradOffset, nParam);
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const auto& js = state.jointState[cur];
    const size_t pi = cur * kParametersPerJoint;
    const Vector3<T> posd = worldPos - js.translation();

    for (size_t d = 0; d < 3; d++) {
      if (ctx.activeJointParams[pi + d]) {
        const T val = wg.dot(js.getTranslationDerivative(d)) * wgt;
        gradient_jointParams_to_modelParams<T>(
            val, pi + d, ctx.parameterTransform, gradientSegment);
      }
      if (ctx.activeJointParams[pi + 3 + d]) {
        const T val = wg.dot(js.getRotationDerivative(d, posd)) * wgt;
        gradient_jointParams_to_modelParams<T>(
            val, pi + 3 + d, ctx.parameterTransform, gradientSegment);
      }
    }
    if (ctx.activeJointParams[pi + 6]) {
      const T val = wg.dot(js.getScaleDerivative(posd)) * wgt;
      gradient_jointParams_to_modelParams<T>(val, pi + 6, ctx.parameterTransform, gradientSegment);
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

// Accumulate gradient for the R_ref^T rotation effect on translation.
template <typename T>
void accumRefRotTransGrad(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const SkeletonStateT<T>& state,
    const Vector3<T>& d_vec,
    const Vector3<T>& wg,
    T wgt,
    Eigen::Ref<Eigen::VectorX<T>> gradient,
    Eigen::Index gradOffset,
    Eigen::Index nParam) {
  Eigen::Ref<Eigen::VectorX<T>> gradientSegment = gradient.segment(gradOffset, nParam);
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const auto& js = state.jointState[cur];
    const size_t pi = cur * kParametersPerJoint;

    for (size_t d = 0; d < 3; d++) {
      if (ctx.activeJointParams[pi + 3 + d]) {
        const T val = -wg.dot(js.getRotationDerivative(d, d_vec)) * wgt;
        gradient_jointParams_to_modelParams<T>(
            val, pi + 3 + d, ctx.parameterTransform, gradientSegment);
      }
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

// Accumulate gradient for the reference-joint inverse-scale effect on
// translation. The per-parameter value is constant along the chain, so it is
// computed once outside the walk.
template <typename T>
void accumRefInvScaleGrad(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const Vector3<T>& relTranslation,
    const Vector3<T>& errCol,
    T wgt,
    Eigen::Ref<Eigen::VectorX<T>> gradient,
    Eigen::Index gradOffset,
    Eigen::Index nParam) {
  Eigen::Ref<Eigen::VectorX<T>> gradientSegment = gradient.segment(gradOffset, nParam);
  const T val = -errCol.dot(relTranslation) * ln2<T>() * wgt;
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const size_t pi = cur * kParametersPerJoint;
    if (ctx.activeJointParams[pi + 6]) {
      gradient_jointParams_to_modelParams<T>(val, pi + 6, ctx.parameterTransform, gradientSegment);
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

// Accumulate gradient for the rotation columns (cols 0-2 of T_rel).
// Rotation columns are scale-independent, so only rotation derivatives apply.
template <typename T>
void accumRotColGrad(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const SkeletonStateT<T>& state,
    const Vector3<T>& srcCol,
    const Vector3<T>& wg,
    T wgt,
    Eigen::Ref<Eigen::VectorX<T>> gradient,
    Eigen::Index gradOffset,
    Eigen::Index nParam) {
  Eigen::Ref<Eigen::VectorX<T>> gradientSegment = gradient.segment(gradOffset, nParam);
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const auto& js = state.jointState[cur];
    const size_t pi = cur * kParametersPerJoint;

    for (size_t d = 0; d < 3; d++) {
      if (ctx.activeJointParams[pi + 3 + d]) {
        const T val = wg.dot(js.getRotationDerivative(d, srcCol)) * wgt;
        gradient_jointParams_to_modelParams<T>(
            val, pi + 3 + d, ctx.parameterTransform, gradientSegment);
      }
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

// ---------------------------------------------------------------------------
// Jacobian chain-walk helpers. These mirror the gradient helpers but write a
// 3xN Jacobian block (3 residual rows of one transform column, at rows
// [rowBase, rowBase + 3) and columns [colOffset, colOffset + nParam)) instead
// of a scalar gradient contribution.
// ---------------------------------------------------------------------------

// Accumulate the translation-column Jacobian from a joint chain.
template <typename T>
void accumTransJac(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const SkeletonStateT<T>& state,
    const Vector3<T>& worldPos,
    const Matrix3<T>& refRotT,
    T sign,
    T f,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Index colOffset,
    Eigen::Index rowBase,
    Eigen::Index nParam) {
  auto jacobianBlock = jacobian.block(rowBase, colOffset, 3, nParam);
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const auto& js = state.jointState[cur];
    const size_t pi = cur * kParametersPerJoint;
    const Vector3<T> posd = worldPos - js.translation();

    for (size_t d = 0; d < 3; d++) {
      if (ctx.activeJointParams[pi + d]) {
        const Vector3<T> jc = sign * f * refRotT * js.getTranslationDerivative(d);
        jacobian_jointParams_to_modelParams<T>(jc, pi + d, ctx.parameterTransform, jacobianBlock);
      }
      if (ctx.activeJointParams[pi + 3 + d]) {
        const Vector3<T> jc = sign * f * refRotT * js.getRotationDerivative(d, posd);
        jacobian_jointParams_to_modelParams<T>(
            jc, pi + 3 + d, ctx.parameterTransform, jacobianBlock);
      }
    }
    if (ctx.activeJointParams[pi + 6]) {
      const Vector3<T> jc = sign * f * refRotT * js.getScaleDerivative(posd);
      jacobian_jointParams_to_modelParams<T>(jc, pi + 6, ctx.parameterTransform, jacobianBlock);
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

// Accumulate the R_ref^T rotation effect on the translation column.
template <typename T>
void accumRefRotTransJac(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const SkeletonStateT<T>& state,
    const Vector3<T>& d_vec,
    const Matrix3<T>& refRotT,
    T sign,
    T f,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Index colOffset,
    Eigen::Index rowBase,
    Eigen::Index nParam) {
  auto jacobianBlock = jacobian.block(rowBase, colOffset, 3, nParam);
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const auto& js = state.jointState[cur];
    const size_t pi = cur * kParametersPerJoint;

    for (size_t d = 0; d < 3; d++) {
      if (ctx.activeJointParams[pi + 3 + d]) {
        const Vector3<T> jc = -sign * f * refRotT * js.getRotationDerivative(d, d_vec);
        jacobian_jointParams_to_modelParams<T>(
            jc, pi + 3 + d, ctx.parameterTransform, jacobianBlock);
      }
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

// Accumulate the reference-joint inverse-scale effect on the translation
// column. The per-parameter contribution is constant along the chain.
template <typename T>
void accumRefInvScaleJac(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const Vector3<T>& relTranslation,
    T sign,
    T f,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Index colOffset,
    Eigen::Index rowBase,
    Eigen::Index nParam) {
  auto jacobianBlock = jacobian.block(rowBase, colOffset, 3, nParam);
  const Vector3<T> jc = -sign * f * ln2<T>() * relTranslation;
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const size_t pi = cur * kParametersPerJoint;
    if (ctx.activeJointParams[pi + 6]) {
      jacobian_jointParams_to_modelParams<T>(jc, pi + 6, ctx.parameterTransform, jacobianBlock);
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

// Accumulate the rotation-column Jacobian (cols 0-2 of T_rel: R_ref^T * R_src[:, c]).
// Rotation columns are scale-independent, so only rotation derivatives apply.
template <typename T>
void accumRotColJac(
    const ChainContext<T>& ctx,
    size_t leafJoint,
    size_t stopJoint,
    const SkeletonStateT<T>& state,
    const Vector3<T>& srcCol,
    const Matrix3<T>& refRotT,
    T sign,
    T f,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Index colOffset,
    Eigen::Index rowBase,
    Eigen::Index nParam) {
  auto jacobianBlock = jacobian.block(rowBase, colOffset, 3, nParam);
  size_t cur = leafJoint;
  while (cur != kInvalidIndex && cur != stopJoint) {
    const auto& js = state.jointState[cur];
    const size_t pi = cur * kParametersPerJoint;

    for (size_t d = 0; d < 3; d++) {
      if (ctx.activeJointParams[pi + 3 + d]) {
        const Vector3<T> jc = sign * f * refRotT * js.getRotationDerivative(d, srcCol);
        jacobian_jointParams_to_modelParams<T>(
            jc, pi + 3 + d, ctx.parameterTransform, jacobianBlock);
      }
    }
    cur = ctx.skeleton.joints[cur].parent;
  }
}

} // namespace

template <typename T>
JointToJointSequenceErrorFunctionT<T>::JointToJointSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SequenceErrorFunctionT<T>(skel, pt) {}

template <typename T>
JointToJointSequenceErrorFunctionT<T>::JointToJointSequenceErrorFunctionT(
    const Character& character)
    : JointToJointSequenceErrorFunctionT(character.skeleton, character.parameterTransform) {}

template <typename T>
void JointToJointSequenceErrorFunctionT<T>::addConstraint(
    const JointToJointSequenceConstraintT<T>& constraint) {
  MT_CHECK(
      constraint.sourceJoint < this->skeleton_.joints.size(),
      "Invalid sourceJoint index: {}",
      constraint.sourceJoint);
  MT_CHECK(
      constraint.referenceJoint < this->skeleton_.joints.size(),
      "Invalid referenceJoint index: {}",
      constraint.referenceJoint);
  constraints_.push_back(constraint);
}

template <typename T>
void JointToJointSequenceErrorFunctionT<T>::addConstraint(
    size_t sourceJoint,
    size_t referenceJoint,
    T weight,
    const std::string& name) {
  JointToJointSequenceConstraintT<T> c;
  c.sourceJoint = sourceJoint;
  c.referenceJoint = referenceJoint;
  c.weight = weight;
  c.name = name;
  addConstraint(c);
}

template <typename T>
void JointToJointSequenceErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
double JointToJointSequenceErrorFunctionT<T>::getError(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */) const {
  MT_PROFILE_FUNCTION();

  if (constraints_.empty()) {
    return 0.0;
  }

  MT_CHECK(skelStates.size() == 2);

  double error = 0.0;

  for (const auto& constr : constraints_) {
    const TransformT<T> prev = relativeTransform(
        this->skeleton_, skelStates[0], constr.sourceJoint, constr.referenceJoint);
    const TransformT<T> next = relativeTransform(
        this->skeleton_, skelStates[1], constr.sourceJoint, constr.referenceJoint);

    const Matrix3<T> prevRelRot = prev.toRotationMatrix();
    const Matrix3<T> nextRelRot = next.toRotationMatrix();
    const Matrix3<T> rotDiff = nextRelRot - prevRelRot;
    const Vector3<T> posDiff = next.translation - prev.translation;
    const T wBase = this->weight_ * constr.weight;
    error += rotDiff.squaredNorm() * wBase * rotWgt_;
    error += posDiff.squaredNorm() * wBase * posWgt_;
  }

  return error;
}

template <typename T>
double JointToJointSequenceErrorFunctionT<T>::getGradient(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  if (constraints_.empty()) {
    return 0.0;
  }

  MT_CHECK(skelStates.size() == 2);
  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(gradient.size() == 2 * nParam);

  const ChainContext<T> ctx{this->skeleton_, this->activeJointParams_, this->parameterTransform_};

  double error = 0.0;

  for (const auto& constr : constraints_) {
    const auto& prevSrc = prevState.jointState[constr.sourceJoint];
    const auto& prevRef = prevState.jointState[constr.referenceJoint];
    const auto& nextSrc = nextState.jointState[constr.sourceJoint];
    const auto& nextRef = nextState.jointState[constr.referenceJoint];
    const TransformT<T> prev =
        relativeTransform(this->skeleton_, prevState, constr.sourceJoint, constr.referenceJoint);
    const TransformT<T> next =
        relativeTransform(this->skeleton_, nextState, constr.sourceJoint, constr.referenceJoint);

    const Matrix3<T> prevRelRot = prev.toRotationMatrix();
    const Matrix3<T> nextRelRot = next.toRotationMatrix();
    const Matrix3<T> rotDiff = nextRelRot - prevRelRot;
    const Vector3<T> posDiff = next.translation - prev.translation;
    const T wBase = this->weight_ * constr.weight;
    error += rotDiff.squaredNorm() * wBase * rotWgt_;
    error += posDiff.squaredNorm() * wBase * posWgt_;

    const size_t lca = this->skeleton_.commonAncestor(constr.sourceJoint, constr.referenceJoint);
    const Matrix3<T> prevRefRot = prevRef.transform.toRotationMatrix();
    const Matrix3<T> nextRefRot = nextRef.transform.toRotationMatrix();
    const T prevRefInvScale = T(1) / prevRef.transform.scale;
    const T nextRefInvScale = T(1) / nextRef.transform.scale;
    const Vector3<T> prevWorldOffset = prevSrc.translation() - prevRef.translation();
    const Vector3<T> nextWorldOffset = nextSrc.translation() - nextRef.translation();

    // First three residuals are rotation matrix columns; the fourth is relative translation.
    for (int col = 0; col < 4; col++) {
      const T colWgt = (col < 3) ? rotWgt_ : posWgt_;
      const T wgt = T(2) * wBase * colWgt;
      const Vector3<T> errCol = (col < 3) ? rotDiff.col(col) : posDiff;

      if (col < 3) {
        const Vector3<T> nextWg = nextRefRot * errCol;
        const Vector3<T> nextWorldSrcCol = nextRefRot * nextRelRot.col(col);
        accumRotColGrad(
            ctx,
            constr.sourceJoint,
            lca,
            nextState,
            nextWorldSrcCol,
            nextWg,
            wgt,
            gradient,
            nParam,
            nParam);
        accumRotColGrad(
            ctx,
            constr.referenceJoint,
            lca,
            nextState,
            nextWorldSrcCol,
            nextWg,
            -wgt,
            gradient,
            nParam,
            nParam);

        const Vector3<T> prevWg = prevRefRot * errCol;
        const Vector3<T> prevWorldSrcCol = prevRefRot * prevRelRot.col(col);
        accumRotColGrad(
            ctx,
            constr.sourceJoint,
            lca,
            prevState,
            prevWorldSrcCol,
            prevWg,
            -wgt,
            gradient,
            0,
            nParam);
        accumRotColGrad(
            ctx,
            constr.referenceJoint,
            lca,
            prevState,
            prevWorldSrcCol,
            prevWg,
            wgt,
            gradient,
            0,
            nParam);
      } else {
        const Vector3<T> nextWg = nextRefInvScale * (nextRefRot * errCol);
        accumTransGrad(
            ctx,
            constr.sourceJoint,
            lca,
            nextState,
            nextSrc.translation(),
            nextWg,
            wgt,
            gradient,
            nParam,
            nParam);
        accumTransGrad(
            ctx,
            constr.referenceJoint,
            lca,
            nextState,
            nextRef.translation(),
            nextWg,
            -wgt,
            gradient,
            nParam,
            nParam);
        accumRefRotTransGrad(
            ctx,
            constr.referenceJoint,
            lca,
            nextState,
            nextWorldOffset,
            nextWg,
            wgt,
            gradient,
            nParam,
            nParam);
        accumRefInvScaleGrad(
            ctx,
            constr.referenceJoint,
            lca,
            next.translation,
            errCol,
            wgt,
            gradient,
            nParam,
            nParam);

        const Vector3<T> prevWg = prevRefInvScale * (prevRefRot * errCol);
        accumTransGrad(
            ctx,
            constr.sourceJoint,
            lca,
            prevState,
            prevSrc.translation(),
            prevWg,
            -wgt,
            gradient,
            0,
            nParam);
        accumTransGrad(
            ctx,
            constr.referenceJoint,
            lca,
            prevState,
            prevRef.translation(),
            prevWg,
            wgt,
            gradient,
            0,
            nParam);
        accumRefRotTransGrad(
            ctx,
            constr.referenceJoint,
            lca,
            prevState,
            prevWorldOffset,
            prevWg,
            -wgt,
            gradient,
            0,
            nParam);
        accumRefInvScaleGrad(
            ctx, constr.referenceJoint, lca, prev.translation, errCol, -wgt, gradient, 0, nParam);
      }
    }
  }

  return error;
}

template <typename T>
size_t JointToJointSequenceErrorFunctionT<T>::getJacobianSize() const {
  return constraints_.size() * 12;
}

template <typename T>
double JointToJointSequenceErrorFunctionT<T>::getJacobian(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian_full,
    Eigen::Ref<Eigen::VectorX<T>> residual_full,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();

  if (constraints_.empty()) {
    usedRows = 0;
    return 0.0;
  }

  MT_CHECK(skelStates.size() == 2);
  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian_full.cols() == 2 * nParam);

  const ChainContext<T> ctx{this->skeleton_, this->activeJointParams_, this->parameterTransform_};

  double error = 0.0;
  Eigen::Index offset = 0;

  for (const auto& constr : constraints_) {
    const auto& prevSrc = prevState.jointState[constr.sourceJoint];
    const auto& prevRef = prevState.jointState[constr.referenceJoint];
    const auto& nextSrc = nextState.jointState[constr.sourceJoint];
    const auto& nextRef = nextState.jointState[constr.referenceJoint];
    const TransformT<T> prev =
        relativeTransform(this->skeleton_, prevState, constr.sourceJoint, constr.referenceJoint);
    const TransformT<T> next =
        relativeTransform(this->skeleton_, nextState, constr.sourceJoint, constr.referenceJoint);

    const Matrix3<T> prevRelRot = prev.toRotationMatrix();
    const Matrix3<T> nextRelRot = next.toRotationMatrix();
    const Matrix3<T> rotDiff = nextRelRot - prevRelRot;
    const Vector3<T> posDiff = next.translation - prev.translation;
    const T wBase = this->weight_ * constr.weight;
    error += rotDiff.squaredNorm() * wBase * rotWgt_;
    error += posDiff.squaredNorm() * wBase * posWgt_;

    const T rotFac = std::sqrt(wBase * rotWgt_);
    for (int col = 0; col < 3; col++) {
      residual_full.template segment<3>(offset + col * 3) = rotDiff.col(col) * rotFac;
    }
    const T posFac = std::sqrt(wBase * posWgt_);
    residual_full.template segment<3>(offset + 9) = posDiff * posFac;

    const size_t lca = this->skeleton_.commonAncestor(constr.sourceJoint, constr.referenceJoint);
    const Matrix3<T> prevRefRot = prevRef.transform.toRotationMatrix();
    const Matrix3<T> nextRefRot = nextRef.transform.toRotationMatrix();
    const Matrix3<T> prevRefRotT = prevRefRot.transpose();
    const Matrix3<T> nextRefRotT = nextRefRot.transpose();
    const T prevRefInvScale = T(1) / prevRef.transform.scale;
    const T nextRefInvScale = T(1) / nextRef.transform.scale;
    const Vector3<T> prevWorldOffset = prevSrc.translation() - prevRef.translation();
    const Vector3<T> nextWorldOffset = nextSrc.translation() - nextRef.translation();
    const Matrix3<T> prevInvScaleRefRotT = prevRefInvScale * prevRefRotT;
    const Matrix3<T> nextInvScaleRefRotT = nextRefInvScale * nextRefRotT;

    // First three row groups are rotation matrix columns; the fourth is relative translation.
    for (int col = 0; col < 4; col++) {
      const Eigen::Index rowBase = offset + col * 3;
      const T colWgt = (col < 3) ? rotWgt_ : posWgt_;
      const T fac = std::sqrt(wBase * colWgt);

      if (col < 3) {
        const Vector3<T> nextWorldSrcCol = nextRefRot * nextRelRot.col(col);
        accumRotColJac(
            ctx,
            constr.sourceJoint,
            lca,
            nextState,
            nextWorldSrcCol,
            nextRefRotT,
            T(1),
            fac,
            jacobian_full,
            nParam,
            rowBase,
            nParam);
        accumRotColJac(
            ctx,
            constr.referenceJoint,
            lca,
            nextState,
            nextWorldSrcCol,
            nextRefRotT,
            T(-1),
            fac,
            jacobian_full,
            nParam,
            rowBase,
            nParam);

        const Vector3<T> prevWorldSrcCol = prevRefRot * prevRelRot.col(col);
        accumRotColJac(
            ctx,
            constr.sourceJoint,
            lca,
            prevState,
            prevWorldSrcCol,
            prevRefRotT,
            T(-1),
            fac,
            jacobian_full,
            0,
            rowBase,
            nParam);
        accumRotColJac(
            ctx,
            constr.referenceJoint,
            lca,
            prevState,
            prevWorldSrcCol,
            prevRefRotT,
            T(1),
            fac,
            jacobian_full,
            0,
            rowBase,
            nParam);
      } else {
        accumTransJac(
            ctx,
            constr.sourceJoint,
            lca,
            nextState,
            nextSrc.translation(),
            nextInvScaleRefRotT,
            T(1),
            fac,
            jacobian_full,
            nParam,
            rowBase,
            nParam);
        accumTransJac(
            ctx,
            constr.referenceJoint,
            lca,
            nextState,
            nextRef.translation(),
            nextInvScaleRefRotT,
            T(-1),
            fac,
            jacobian_full,
            nParam,
            rowBase,
            nParam);
        accumRefRotTransJac(
            ctx,
            constr.referenceJoint,
            lca,
            nextState,
            nextWorldOffset,
            nextInvScaleRefRotT,
            T(1),
            fac,
            jacobian_full,
            nParam,
            rowBase,
            nParam);
        accumRefInvScaleJac(
            ctx,
            constr.referenceJoint,
            lca,
            next.translation,
            T(1),
            fac,
            jacobian_full,
            nParam,
            rowBase,
            nParam);

        accumTransJac(
            ctx,
            constr.sourceJoint,
            lca,
            prevState,
            prevSrc.translation(),
            prevInvScaleRefRotT,
            T(-1),
            fac,
            jacobian_full,
            0,
            rowBase,
            nParam);
        accumTransJac(
            ctx,
            constr.referenceJoint,
            lca,
            prevState,
            prevRef.translation(),
            prevInvScaleRefRotT,
            T(1),
            fac,
            jacobian_full,
            0,
            rowBase,
            nParam);
        accumRefRotTransJac(
            ctx,
            constr.referenceJoint,
            lca,
            prevState,
            prevWorldOffset,
            prevInvScaleRefRotT,
            T(-1),
            fac,
            jacobian_full,
            0,
            rowBase,
            nParam);
        accumRefInvScaleJac(
            ctx,
            constr.referenceJoint,
            lca,
            prev.translation,
            T(-1),
            fac,
            jacobian_full,
            0,
            rowBase,
            nParam);
      }
    }

    offset += 12;
  }

  usedRows = gsl::narrow_cast<int>(offset);
  return error;
}

template class JointToJointSequenceErrorFunctionT<float>;
template class JointToJointSequenceErrorFunctionT<double>;

} // namespace momentum
