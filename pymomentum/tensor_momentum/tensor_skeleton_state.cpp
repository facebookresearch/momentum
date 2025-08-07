/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_skeleton_state.h"

#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_momentum/tensor_quaternion.h"
#include "pymomentum/tensor_momentum/tensor_transforms.h"
#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>

#include <ceres/jet.h>
#include <dispenso/parallel_for.h> // @manual
#include <torch/csrc/jit/python/python_ivalue.h>
#include <Eigen/Core>

namespace py = pybind11;
namespace mm = momentum;

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace pymomentum {

namespace {

template <typename T>
momentum::TransformT<T> computeLocalTransform(
    const momentum::Joint& joint,
    Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>> parameters) {
  momentum::TransformT<T> result;

  result.translation =
      joint.translationOffset.cast<T>() + parameters.template segment<3>(0);

  // apply pre-rotation
  result.rotation = joint.preRotation.cast<T>();

  // do the rotations
  for (int index = 2; index >= 0; --index) {
    result.rotation *= Eigen::Quaternion<T>(Eigen::AngleAxis<T>(
        parameters[3 + index], Eigen::Vector3f::Unit(index).cast<T>()));
  }

  // perform scale if necessary
  result.scale = pow(T(2.0), parameters[6]);

  return result;
}

// Compute
//    dLoss_dModelParameters = dSkeletonState_dModelParameters .
//    dLoss_dSkeletonState
template <typename T>
void computeSkelStateBackward(
    const momentum::Skeleton& skeleton,
    Eigen::Ref<const Eigen::VectorX<T>> jointParameters,
    Eigen::Ref<const Eigen::VectorX<T>> dLoss_dSkeletonState,
    Eigen::Ref<Eigen::VectorX<T>> dLoss_dJointParameters) {
  using momentum::kInvalidIndex;
  using momentum::kParametersPerJoint;

  const momentum::SkeletonStateT<T> skelState(
      momentum::JointParametersT<T>(jointParameters), skeleton);

  for (size_t iJoint = 0; iJoint < skeleton.joints.size(); ++iJoint) {
    using JetType = ceres::Jet<T, 1>;

    // If the dot product with dLoss_dSkelState would be zero, we can skip.
    // This is definitely worth doing because there are many cases where we'll
    // be using a subset of the skeleton states.
    Eigen::Ref<const Eigen::VectorX<T>> dLoss_dSkelState_cur =
        dLoss_dSkeletonState.template segment<8>(8 * iJoint);
    if (dLoss_dSkelState_cur.isZero()) {
      continue;
    }

    // For each joint i, we want to find all the parameters k that affect
    // the skeleton state T_i (which transforms from the local space of joint
    // i to world space).  For each such parameter, we add in the derivatives,
    //   dLoss/dTheta_k += dLoss/dT_i * dT_i/dTheta_k
    // where Theta_k is the kth joint parameter.
    // Note that each Theta_k only affects a single _local_ transform (by
    // abuse of notation, call it Q_k(theta_k)).
    //
    // To figure out how the joint parameter k affects the _world_ transform
    // T_i, we look at the stack of transforms that make up T_i:
    //    T_i =  ... Q_k(theta_k) ... Q_i
    // Since only Q_k actually depends on theta_k, we can compute the derivative
    // of the whole stack by augmenting Q_k with a ceres::Jet and computing the
    // product.  However, we can speed this computation up significantly by
    // reusing the transform from k->world (this is just the skeleton state of
    // k's parent) as well as the accumulated transform from i to k (which we
    // will store in accumTransform).
    momentum::TransformT<T> accumTransform;
    size_t curJoint = iJoint;
    while (curJoint != momentum::kInvalidIndex) {
      const auto& joint = skeleton.joints[curJoint];
      const momentum::TransformT<T> parentXF = (joint.parent == kInvalidIndex)
          ? momentum::TransformT<T>()
          : skelState.jointState[joint.parent].transform;

      for (int d = 0; d < momentum::kParametersPerJoint; ++d) {
        Eigen::VectorX<JetType> jointParams_cur =
            jointParameters
                .segment(kParametersPerJoint * curJoint, kParametersPerJoint)
                .template cast<JetType>();
        jointParams_cur(d).v[0] = 1;

        const momentum::TransformT<JetType> joint_fullXF =
            momentum::TransformT<JetType>(parentXF) *
            computeLocalTransform<JetType>(joint, jointParams_cur) *
            momentum::TransformT<JetType>(accumTransform);

        dLoss_dJointParameters(curJoint * momentum::kParametersPerJoint + d) +=
            joint_fullXF.translation
                .dot(dLoss_dSkeletonState.template segment<3>(8 * iJoint))
                .v[0];
        dLoss_dJointParameters(curJoint * momentum::kParametersPerJoint + d) +=
            joint_fullXF.rotation.coeffs()
                .dot(dLoss_dSkeletonState.template segment<4>(8 * iJoint + 3))
                .v[0];
        dLoss_dJointParameters(curJoint * momentum::kParametersPerJoint + d) +=
            (joint_fullXF.scale * dLoss_dSkeletonState(8 * iJoint + 7)).v[0];
      }

      accumTransform =
          skelState.jointState[curJoint].localTransform * accumTransform;
      curJoint = joint.parent;
    }
  }
}

// Compute
//    dLoss_dJointParameters = dSkeletonState_dJointParameters .
//    dLoss_dSkeletonState
template <typename T>
void computeLocalSkelStateBackward(
    const momentum::Skeleton& skeleton,
    Eigen::Ref<const Eigen::VectorX<T>> jointParameters,
    Eigen::Ref<const Eigen::VectorX<T>> dLoss_dLocalSkeletonState,
    Eigen::Ref<Eigen::VectorX<T>> dLoss_dJointParameters) {
  using momentum::kParametersPerJoint;

  const momentum::SkeletonStateT<T> skelState(
      momentum::JointParametersT<T>(jointParameters), skeleton);

  for (size_t iJoint = 0; iJoint < skeleton.joints.size(); ++iJoint) {
    using JetType = ceres::Jet<T, 1>;

    // If the dot product with dLoss_dSkelState would be zero, we can skip.
    // This is definitely worth doing because there are many cases where we'll
    // be using a subset of the skeleton states.
    Eigen::Ref<const Eigen::VectorX<T>> dLoss_dLocalSkelState_cur =
        dLoss_dLocalSkeletonState.template segment<8>(8 * iJoint);
    if (dLoss_dLocalSkelState_cur.isZero()) {
      continue;
    }

    // For each joint i, we want to find all the parameters k that affect
    // the skeleton state T_i (which transforms from the local space of joint
    // i to world space).  For each such parameter, we add in the derivatives,
    //   dLoss/dTheta_k += dLoss/dT_i * dT_i/dTheta_k
    // where Theta_k is the kth joint parameter.
    // Note that each Theta_k only affects a single _local_ transform (by
    // abuse of notation, call it Q_k(theta_k)).
    //
    // To figure out how the joint parameter k affects the _world_ transform
    // T_i, we look at the stack of transforms that make up T_i:
    //    T_i =  ... Q_k(theta_k) ... Q_i
    // Since only Q_k actually depends on theta_k, we can compute the derivative
    // of the whole stack by augmenting Q_k with a ceres::Jet and computing the
    // product.  However, we can speed this computation up significantly by
    // reusing the transform from k->world (this is just the skeleton state of
    // k's parent) as well as the accumulated transform from i to k (which we
    // will store in accumTransform).
    const auto& joint = skeleton.joints[iJoint];

    for (int d = 0; d < momentum::kParametersPerJoint; ++d) {
      Eigen::VectorX<JetType> jointParams_cur =
          jointParameters
              .segment(kParametersPerJoint * iJoint, kParametersPerJoint)
              .template cast<JetType>();
      jointParams_cur(d).v[0] = 1;

      const momentum::TransformT<JetType> joint_fullXF =
          computeLocalTransform<JetType>(joint, jointParams_cur);

      dLoss_dJointParameters(iJoint * momentum::kParametersPerJoint + d) +=
          joint_fullXF.translation
              .dot(dLoss_dLocalSkeletonState.template segment<3>(8 * iJoint))
              .v[0];
      dLoss_dJointParameters(iJoint * momentum::kParametersPerJoint + d) +=
          joint_fullXF.rotation.coeffs()
              .dot(
                  dLoss_dLocalSkeletonState.template segment<4>(8 * iJoint + 3))
              .v[0];
      dLoss_dJointParameters(iJoint * momentum::kParametersPerJoint + d) +=
          (joint_fullXF.scale * dLoss_dLocalSkeletonState(8 * iJoint + 7)).v[0];
    }
  }
}

template <typename T>
struct JointParametersToSkeletonStateFunction
    : public torch::autograd::Function<
          JointParametersToSkeletonStateFunction<T>> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      PyObject* characters_in,
      at::Tensor modelParameters);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_jointParameters);
};

template <typename T>
variable_list JointParametersToSkeletonStateFunction<T>::forward(
    AutogradContext* ctx,
    PyObject* characters_in,
    at::Tensor jointParameters) {
  const int nJoints =
      (int)anyCharacter(characters_in, "jointParametersToSkeletonState()")
          .skeleton.joints.size();
  const int nJointParams = nJoints * momentum::kParametersPerJoint;

  TensorChecker checker("jointParametersToSkeletonState");
  bool squeeze;
  const auto input_device =
      jointParameters[0]
          .device(); // Save the input device, reused for the returned grad

  jointParameters = flattenJointParameters(
      anyCharacter(characters_in, "jointParametersToLocalSkeletonState()"),
      jointParameters);

  jointParameters = checker.validateAndFixTensor(
      jointParameters,
      "jointParameters",
      {nJointParams},
      {"nJointParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);
  const auto nBatch = checker.getBatchSize();

  ctx->saved_data["character"] =
      c10::ivalue::ConcretePyObjectHolder::create(characters_in);
  ctx->save_for_backward({jointParameters});

  const auto characters = toCharacterList(
      characters_in, nBatch, "jointParametersToSkeletonState()");

  auto result = at::zeros({nBatch, nJoints, 8}, toScalarType<T>());
  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    const momentum::Character* character = characters[iBatch];
    const momentum::SkeletonStateT<T> skelState(
        toEigenMap<T>(jointParameters.select(0, iBatch)), character->skeleton);
    auto result_cur = toEigenMap<T>(result.select(0, iBatch));
    for (int64_t iJoint = 0; iJoint < nJoints; ++iJoint) {
      result_cur.template segment<3>(8 * iJoint + 0) =
          skelState.jointState[iJoint].translation();
      result_cur.template segment<4>(8 * iJoint + 3) =
          skelState.jointState[iJoint].rotation().coeffs();
      result_cur(8 * iJoint + 7) = skelState.jointState[iJoint].scale();
    }
  });

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result.to(input_device)};
}

template <typename T>
variable_list JointParametersToSkeletonStateFunction<T>::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in JointParametersToSkeletonStateFunction::backward");

  // Restore variables:
  const auto saved = ctx->get_saved_variables();
  auto savedItr = std::begin(saved);
  auto jointParameters = *savedItr++;
  MT_THROW_IF(
      savedItr != std::end(saved), "Mismatch in saved variable counts.");

  const auto nJoints = anyCharacter(
                           ctx->saved_data["character"].toPyObject(),
                           "modelParametersToPositions()")
                           .skeleton.joints.size();
  const int nJointParams = nJoints * momentum::kParametersPerJoint;

  TensorChecker checker("jointParametersToSkeletonState");
  const auto input_device =
      grad_outputs[0]
          .device(); // Save the input device, reused for the returned grad

  bool squeeze_jointParams;
  jointParameters = checker.validateAndFixTensor(
      jointParameters,
      "jointParameters",
      {(int)nJointParams},
      {"nJointParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_jointParams);

  bool squeeze_dLoss = false;
  auto dLoss_dSkeletonState = checker.validateAndFixTensor(
      grad_outputs[0],
      "dLoss_dSkeletonState",
      {(int)nJoints, 8},
      {"nJoints", "trans/rot/scale"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_dLoss);

  const auto nBatch = checker.getBatchSize();

  const auto characters = toCharacterList(
      ctx->saved_data["character"].toPyObject(),
      nBatch,
      "jointParametersToSkeletonState()");

  auto result = at::zeros({nBatch, nJointParams}, toScalarType<T>());
  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    const momentum::Character* character = characters[iBatch];
    computeSkelStateBackward<T>(
        character->skeleton,
        toEigenMap<T>(jointParameters.select(0, iBatch)),
        toEigenMap<T>(dLoss_dSkeletonState.select(0, iBatch)),
        toEigenMap<T>(result.select(0, iBatch)));
  });

  if (squeeze_jointParams) {
    result = result.sum(0);
  }

  return {at::Tensor(), result.to(input_device)};
}

template <typename T>
struct JointParametersToLocalSkeletonStateFunction
    : public torch::autograd::Function<
          JointParametersToLocalSkeletonStateFunction<T>> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      PyObject* characters_in,
      at::Tensor modelParameters);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_jointParameters);
};

template <typename T>
variable_list JointParametersToLocalSkeletonStateFunction<T>::forward(
    AutogradContext* ctx,
    PyObject* characters_in,
    at::Tensor jointParameters) {
  const int nJoints =
      (int)anyCharacter(characters_in, "jointParametersToLocalSkeletonState()")
          .skeleton.joints.size();
  const int nJointParams = nJoints * momentum::kParametersPerJoint;

  TensorChecker checker("jointParametersToLocalSkeletonState");
  bool squeeze;
  const auto input_device = jointParameters.device();

  jointParameters = flattenJointParameters(
      anyCharacter(characters_in, "jointParametersToLocalSkeletonState()"),
      jointParameters);
  jointParameters = checker.validateAndFixTensor(
      jointParameters,
      "jointParameters",
      {nJointParams},
      {"nJointParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);
  const auto nBatch = checker.getBatchSize();

  ctx->saved_data["character"] =
      c10::ivalue::ConcretePyObjectHolder::create(characters_in);
  ctx->save_for_backward({jointParameters});

  const auto characters = toCharacterList(
      characters_in, nBatch, "jointParametersToLocalSkeletonState()");

  auto result = at::zeros({nBatch, nJoints, 8}, toScalarType<T>());
  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    const momentum::Character* character = characters[iBatch];
    const momentum::SkeletonStateT<T> skelState(
        toEigenMap<T>(jointParameters.select(0, iBatch)), character->skeleton);
    auto result_cur = toEigenMap<T>(result.select(0, iBatch));
    for (int64_t iJoint = 0; iJoint < nJoints; ++iJoint) {
      result_cur.template segment<3>(8 * iJoint + 0) =
          skelState.jointState[iJoint].localTranslation();
      result_cur.template segment<4>(8 * iJoint + 3) =
          skelState.jointState[iJoint].localRotation().coeffs();
      result_cur(8 * iJoint + 7) = skelState.jointState[iJoint].localScale();
    }
  });

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result.to(input_device)};
}

template <typename T>
variable_list JointParametersToLocalSkeletonStateFunction<T>::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in JointParametersToLocalSkeletonStateFunction::backward");

  // Restore variables:
  const auto saved = ctx->get_saved_variables();
  auto savedItr = std::begin(saved);
  auto jointParameters = *savedItr++;
  MT_THROW_IF(
      savedItr != std::end(saved), "Mismatch in saved variable counts.");

  const auto nJoints = anyCharacter(
                           ctx->saved_data["character"].toPyObject(),
                           "modelParametersToPositions()")
                           .skeleton.joints.size();
  const int nJointParams = nJoints * momentum::kParametersPerJoint;

  TensorChecker checker("jointParametersToLocalSkeletonState");
  const auto input_device =
      grad_outputs[0]
          .device(); // Save the input device, reused for the returned grad

  bool squeeze_jointParams;
  jointParameters = checker.validateAndFixTensor(
      jointParameters,
      "jointParameters",
      {(int)nJointParams},
      {"nJointParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_jointParams);

  bool squeeze_dLoss = false;
  auto dLoss_dLocalSkeletonState = checker.validateAndFixTensor(
      grad_outputs[0],
      "dLoss_LocalSkeletonState",
      {(int)nJoints, 8},
      {"nJoints", "trans/rot/scale"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_dLoss);

  const auto nBatch = checker.getBatchSize();

  const auto characters = toCharacterList(
      ctx->saved_data["character"].toPyObject(),
      nBatch,
      "jointParametersToLocalSkeletonState()");

  auto result = at::zeros({nBatch, nJointParams}, toScalarType<T>());
  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    const momentum::Character* character = characters[iBatch];
    computeLocalSkelStateBackward<T>(
        character->skeleton,
        toEigenMap<T>(jointParameters.select(0, iBatch)),
        toEigenMap<T>(dLoss_dLocalSkeletonState.select(0, iBatch)),
        toEigenMap<T>(result.select(0, iBatch)));
  });

  if (squeeze_jointParams) {
    result = result.sum(0);
  }

  return {at::Tensor(), result.to(input_device)};
}

} // anonymous namespace

at::Tensor jointParametersToSkeletonState(
    pybind11::object characters,
    at::Tensor jointParams) {
  return applyTemplatedAutogradFunction<JointParametersToSkeletonStateFunction>(
      characters.ptr(), jointParams)[0];
}

at::Tensor jointParametersToLocalSkeletonState(
    pybind11::object characters,
    at::Tensor jointParams) {
  return applyTemplatedAutogradFunction<
      JointParametersToLocalSkeletonStateFunction>(
      characters.ptr(), jointParams)[0];
}

at::Tensor modelParametersToSkeletonState(
    pybind11::object characters,
    at::Tensor modelParams) {
  return jointParametersToSkeletonState(
      characters, applyParamTransform(characters, modelParams));
}

at::Tensor modelParametersToLocalSkeletonState(
    pybind11::object characters,
    at::Tensor modelParams) {
  return jointParametersToLocalSkeletonState(
      characters, applyParamTransform(characters, modelParams));
}

at::Tensor matricesToSkeletonStates(at::Tensor matrices) {
  MT_THROW_IF(
      matrices.dim() < 2 || matrices.size(-1) != 4 || matrices.size(-2) != 4,
      "Expected a tensor of 4x4 matrices");

  /*
      linear = matrices[..., 0:3, 0:3]
    translations = matrices[..., 0:3, 3]
    assert translations.shape == torch.Size(list(matrices.shape[0:-2]) + [3])
    # singular_values dim: batch_size x 3
    # each row is the 3 singular values for the linear
    _, singular_values, _ = torch.linalg.svd(linear)
    # scales shape: batch_size x 1
    scales = singular_values[..., 0].unsqueeze(-1)
    rotations = linear / scales.unsqueeze(-1)
    quaternions = matrix_to_quaternion(rotations)
    assert quaternions.shape == torch.Size(list(matrices.shape[0:-2]) + [4])
    skel_states = torch.cat([translations, quaternions, scales], dim=-1)
  */
  // SVD, bmm need the tensor flattened into [nMat, 4, 4]; do that here and
  // then unflatten later:
  const auto initialShape = matrices.sizes();
  if (matrices.dim() == 2) {
    matrices = matrices.unsqueeze(0);
  } else {
    matrices = matrices.flatten(0, -3);
  }

  const at::Tensor linear = matrices.narrow(-1, 0, 3).narrow(-2, 0, 3);
  const at::Tensor translations = matrices.narrow(-2, 0, 3).select(-1, 3);
  // Torch SVD actually returns the transpose of V:
  const auto [U, S, Vt] = at::linalg_svd(linear);

  // assume the largest singular value is the scale:
  const at::Tensor scales = S.narrow(-1, 0, 1);
  const at::Tensor rotation_matrices = at::bmm(U, Vt);
  const at::Tensor quaternions = rotationMatrixToQuaternion(rotation_matrices);
  at::Tensor result = at::cat({translations, quaternions, scales}, -1);
  std::vector<int64_t> resultShape(
      initialShape.begin(), initialShape.end() - 2);
  resultShape.push_back(8);
  return result.reshape(resultShape);
}

at::Tensor parentsTensor(const momentum::Character& character) {
  std::vector<int64_t> parentsVec;
  parentsVec.reserve(character.skeleton.joints.size());
  for (const auto& j : character.skeleton.joints) {
    parentsVec.push_back(
        j.parent == momentum::kInvalidIndex ? static_cast<int64_t>(-1)
                                            : static_cast<int64_t>(j.parent));
  }
  return to1DTensor(parentsVec);
}

at::Tensor preRotationsTensor(const momentum::Character& character) {
  const int64_t nJoints = character.skeleton.joints.size();
  Eigen::VectorXf data(character.skeleton.joints.size() * 4);
  for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
    data.segment<4>(4 * i) = character.skeleton.joints[i].preRotation.coeffs();
  }

  return torch::from_blob(
             (void*)data.data(),
             {nJoints, 4},
             torch::TensorOptions().dtype(toScalarType<float>()))
      .clone();
}

at::Tensor translationOffsetsTensor(const momentum::Character& character) {
  const int64_t nJoints = character.skeleton.joints.size();
  Eigen::VectorXf data(character.skeleton.joints.size() * 3);
  for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
    data.segment<3>(3 * i) = character.skeleton.joints[i].translationOffset;
  }

  return torch::from_blob(
             (void*)data.data(),
             {nJoints, 3},
             torch::TensorOptions().dtype(toScalarType<float>()))
      .clone();
}

at::Tensor getParentSkeletonState(
    const momentum::Character& character,
    at::Tensor skelState) {
  // Attach the identity to the beginning of the skel_state.  Then we'll offset
  // the parents by such that the root node gets its world-space transform from
  // the 0th element.
  at::Tensor augmentedSkelState = [&]() {
    at::Tensor identitySkelState = identitySkeletonState();
    // Reshape the identity so we can stack it:
    while (identitySkelState.ndimension() < skelState.ndimension()) {
      identitySkelState = identitySkelState.unsqueeze(0);
    }
    const auto sizes = skelState.sizes();
    std::vector<int64_t> identityTensorShape(sizes.begin(), sizes.end());
    identityTensorShape.at(identityTensorShape.size() - 2) = 1;
    identitySkelState = identitySkelState.expand(identityTensorShape, true);

    return at::cat(std::vector<at::Tensor>{identitySkelState, skelState}, -2);
  }();

  at::Tensor parents = parentsTensor(character) + 1;
  return augmentedSkelState.index_select(-2, parents);
}

at::Tensor localSkeletonStateToJointParameters(
    const momentum::Character& character,
    at::Tensor localSkelState) {
  MT_THROW_IF(
      localSkelState.ndimension() < 2 ||
          localSkelState.size(-2) != character.skeleton.joints.size() ||
          localSkelState.size(-1) != 8,
      "Expected skel_state with dimensions [nBatch x nJoints={} x 8]; got {}.",
      character.skeleton.joints.size(),
      formatTensorSizes(localSkelState));

  auto [localTranslation, localRotation, localScale] =
      splitSkeletonState(localSkelState);

  // For translation, just need to subtract off the per-joint offset:
  at::Tensor translationOffsets =
      translationOffsetsTensor(character).type_as(localSkelState);
  while (translationOffsets.ndimension() < localSkelState.ndimension()) {
    translationOffsets = translationOffsets.unsqueeze(0);
  }
  translationOffsets = translationOffsets.expand_as(localTranslation);
  at::Tensor translationJointParams = localTranslation - translationOffsets;

  // For rotation joint parameters, we need to first remove the
  // pre-rotation, and then convert to Euler angles.
  //    local_skel_state =  prerot * local_rot
  //    local_rot = pre_rot.inverse() * local_skel_state
  at::Tensor preRotations =
      preRotationsTensor(character).type_as(localSkelState);
  while (preRotations.ndimension() < localSkelState.ndimension()) {
    preRotations = preRotations.unsqueeze(0);
  }
  preRotations = preRotations.expand_as(localRotation);
  at::Tensor rotationJointParams = quaternionToXYZEuler(
      quaternionMultiply(quaternionInverse(preRotations), localRotation));

  // skel state scale is exp2 of the joint parameter scale:
  at::Tensor scaleJointParams = at::log2(localScale);

  return at::cat(
      {translationJointParams, rotationJointParams, scaleJointParams}, -1);
}

at::Tensor skeletonStateToJointParameters(
    const momentum::Character& character,
    at::Tensor skelState) {
  MT_THROW_IF(
      skelState.ndimension() < 2 ||
          skelState.size(-2) != character.skeleton.joints.size() ||
          skelState.size(-1) != 8,
      "Expected skel_state with dimensions [nBatch x nJoints={} x 8]; got {}.",
      character.skeleton.joints.size(),
      formatTensorSizes(skelState));

  at::Tensor parentSkelState = getParentSkeletonState(character, skelState);

  // Compute joint-to-parent transform.
  // T_parentToWorld * T_jointToParent = T_jointToWorld
  // T_jointToParent = inv(T_parentToWorld) * T_jointToWorld
  at::Tensor localSkelState =
      multiplySkeletonStates(inverseSkeletonStates(parentSkelState), skelState);
  return localSkeletonStateToJointParameters(character, localSkelState);
}

} // namespace pymomentum
