/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//------------------------------------------------------------------------------
// Momentum specific Eigen defines
//------------------------------------------------------------------------------

#include <momentum/character/fwd.h>

#include <Eigen/Dense>

#include <array>
#include <string>
#include <tuple>
#include <vector>

namespace momentum {

inline constexpr size_t kParametersPerJoint = 7;
enum JointParameterNames { TX = 0, TY, TZ, RX, RY, RZ, SC };
inline constexpr std::array<const char*, kParametersPerJoint>
    kJointParameterNames{"tx", "ty", "tz", "rx", "ry", "rz", "sc"};

template <typename T>
using JointVectorT = Eigen::Matrix<T, kParametersPerJoint, 1>;
using JointVector = JointVectorT<float>;

template <template <typename> typename Derived, typename EigenType>
struct EigenStrongType {
  using Scalar = typename EigenType::Scalar;

  EigenType v;

  EigenStrongType() : v() {}

  explicit EigenStrongType(::Eigen::Index size) : v(size) {}

  template <typename Other>
  /* implicit */ EigenStrongType(const ::Eigen::EigenBase<Other>& o) : v(o) {}

  template <typename Other>
  /* implicit */ EigenStrongType(::Eigen::EigenBase<Other>&& o) : v(std::move(o)) {}

  template <typename Other>
  EigenStrongType& operator=(const ::Eigen::EigenBase<Other>& o) {
    v = o;
    return *this;
  }

  template <typename Other>
  EigenStrongType& operator=(::Eigen::EigenBase<Other>&& o) {
    v = std::move(o);
    return *this;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  auto&& operator()(::Eigen::Index i) {
    return v(i);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  auto&& operator()(::Eigen::Index i) const {
    return v(i);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  auto&& operator[](::Eigen::Index i) {
    return v[i];
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  auto&& operator[](::Eigen::Index i) const {
    return v[i];
  }

  template <typename NewScalar>
  [[nodiscard]] EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived<NewScalar> cast() const {
    return v.template cast<NewScalar>();
  }

  [[nodiscard]] EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE auto size() const {
    return v.size();
  }

  EIGEN_DEVICE_FUNC
  static Derived<Scalar> Zero(::Eigen::Index size) {
    return Derived<Scalar>(EigenType::Zero(size));
  }
};

// a vector describing model parameters
template <typename T>
struct ModelParametersT : public EigenStrongType<ModelParametersT, ::Eigen::VectorX<T>> {
  using t = ::Eigen::VectorX<T>;
  using EigenStrongType<ModelParametersT, t>::EigenStrongType;
  using EigenStrongType<ModelParametersT, t>::operator=;
};

using ModelParameters = ModelParametersT<float>;
using ModelParametersd = ModelParametersT<double>;

template <typename T>
struct BlendWeightsT : public EigenStrongType<BlendWeightsT, ::Eigen::VectorX<T>> {
  using t = ::Eigen::VectorX<T>;
  using EigenStrongType<BlendWeightsT, t>::EigenStrongType;
  using EigenStrongType<BlendWeightsT, t>::operator=;
};

using BlendWeights = BlendWeightsT<float>;
using BlendWeightsd = BlendWeightsT<double>;

static_assert(
    sizeof(BlendWeights) == sizeof(Eigen::VectorXf),
    "Missing Empty Base Class Optimization");
static_assert(
    sizeof(BlendWeightsd) == sizeof(Eigen::VectorXd),
    "Missing Empty Base Class Optimization");

// A vector describing joint parameters, the vector has size numSkeletonJoints * kParametersPerJoint
// e.g. in case of a character body model with 159 joints, the joint parameters vector will have the
// size 159 * 7 = 1113
template <typename T>
struct JointParametersT : public EigenStrongType<JointParametersT, ::Eigen::VectorX<T>> {
  using t = ::Eigen::VectorX<T>;
  using EigenStrongType<JointParametersT, t>::EigenStrongType;
  using EigenStrongType<JointParametersT, t>::operator=;

  [[nodiscard]] static ::Eigen::Vector3<T> fromRotationMatrix(const ::Eigen::Matrix3<T>& m) {
    // From JointState::set(), we can see that localRotation = rz * ry * rx, but the order in
    // JointParameters is [rx, ry, rz]. Therefore, the conversion should be
    // rotationMatrixToEulerZYX.reverse.
    return rotationMatrixToEulerZYX(m).reverse();
  }

  [[nodiscard]] static ::Eigen::Vector3<T> fromQuaternion(const ::Eigen::Quaternion<T>& q) {
    return fromRotationMatrix(q.toRotationMatrix());
  }
};

using JointParameters = JointParametersT<float>;
using JointParametersd = JointParametersT<double>;

static_assert(
    sizeof(JointParameters) == sizeof(Eigen::VectorXf),
    "Missing Empty Base Class Optimization");
static_assert(
    sizeof(JointParametersd) == sizeof(Eigen::VectorXd),
    "Missing Empty Base Class Optimization");

template <typename T>
using JointStateListT = std::vector<JointStateT<T>>;

using JointStateList = JointStateListT<float>;
using JointStateListd = JointStateListT<double>;

/// A struct that encapsulates both pose and identity parameters for a character.
///
/// @note This structure implies, but does not enforce, that the pose vector should contain only
/// pose information with identity-related elements set to zero. Similarly, the identity vector
/// should exclusively contain bone length information, excluding any pose data.
template <typename T>
struct CharacterParametersT {
  /// The model parameter vector representing the pose of the character. This vector should have a
  /// size of numModelParams.

  ModelParametersT<T> pose;
  /// The joint parameter vector representing the unique bone lengths of the character, defining
  /// the character's identity. This vector should have a size of numSkeletonJoints *
  /// kParametersPerJoint.
  JointParametersT<T> offsets;
};

using CharacterParameters = CharacterParametersT<float>;
using CharacterParametersd = CharacterParametersT<double>;

// The tuple of model parameter names and corresponding matrix representing the pose in a sequence
// of frames The poses are ordered in columns and the expected shape of motion matrix is
// (numModelParams, numFrames)

using MotionParameters = std::tuple<std::vector<std::string>, Eigen::MatrixXf>;

// The tuple containing the skeleton joint names and identity parameters. The identity parameters
// represent bone offsets and bone scales that are added to the joint states (local transform for
// each joint wrt parent joint) during FK step The identity parameters are expressed as a vector of
// size (numSkeletonJoints * momentum::kParametersPerJoint)

using IdentityParameters = std::tuple<std::vector<std::string>, Eigen::VectorXf>;

// define static kInvalidIndex for size_t
inline constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();

} // namespace momentum
