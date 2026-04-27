/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/random.h>
#include <momentum/math/types.h>

namespace momentum {

namespace detail {

template <typename RealType>
using UniformRealDist = std::uniform_real_distribution<RealType>;

template <typename IntType>
using UniformIntDist = std::uniform_int_distribution<IntType>;

template <typename RealType>
using NormalRealDist = std::normal_distribution<RealType>;

// Whitelists the integer types that std::uniform_int_distribution<T> permits.
// The standard restricts the IntType template parameter to (un)signed
// short/int/long/long long; using e.g. char or int8_t is undefined behavior.
// Ref: https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
template <typename T>
struct is_compatible_to_uniform_int_distribution
    : std::disjunction<
          std::is_same<std::remove_cv_t<T>, short>,
          std::is_same<std::remove_cv_t<T>, int>,
          std::is_same<std::remove_cv_t<T>, long>,
          std::is_same<std::remove_cv_t<T>, long long>,
          std::is_same<std::remove_cv_t<T>, unsigned short>,
          std::is_same<std::remove_cv_t<T>, unsigned int>,
          std::is_same<std::remove_cv_t<T>, unsigned long>,
          std::is_same<std::remove_cv_t<T>, unsigned long long>> {};

template <typename T>
inline constexpr bool is_compatible_to_uniform_int_distribution_v =
    is_compatible_to_uniform_int_distribution<T>::value;

template <template <typename...> class C, typename... Ts>
std::true_type is_base_of_template_impl(const C<Ts...>*);

template <template <typename...> class C>
std::false_type is_base_of_template_impl(...);

template <template <typename...> class C, typename T>
using is_base_of_template = decltype(is_base_of_template_impl<C>(std::declval<T*>()));

template <typename T>
using is_base_of_matrix = is_base_of_template<Eigen::MatrixBase, T>;

template <typename T>
inline constexpr bool is_base_of_matrix_v = is_base_of_matrix<T>::value;

// Note: when T is neither floating-point nor a whitelisted integer type
// (e.g. bool, char, int8_t, uint8_t), no branch is taken and the function
// returns a value-initialized T. TODO: consider a static_assert to fail at
// compile time instead of silently returning T{}.
template <typename T, typename Generator>
[[nodiscard]] T generateScalarUniform(const T& min, const T& max, Generator& generator) {
  if constexpr (std::is_floating_point_v<T>) {
    // std::uniform_real_distribution generates on the half-open interval
    // [min, max); some implementations may return max due to rounding.
    UniformRealDist<T> dist(min, max);
    return dist(generator);
  } else if constexpr (is_compatible_to_uniform_int_distribution_v<T>) {
    // Closed interval [min, max] for integer distribution.
    UniformIntDist<T> dist(min, max);
    return dist(generator);
  }
}

// Generates a random vector/matrix element-wise. The four `if constexpr`
// branches dispatch to the matching Eigen NullaryExpr overload (matrix vs
// vector, dynamic vs fixed) since each takes a different signature for the
// size arguments and the generator lambda.
template <typename Derived, typename Generator>
[[nodiscard]] typename Derived::PlainObject generateMatrixUniform(
    const Eigen::MatrixBase<Derived>& min,
    const Eigen::MatrixBase<Derived>& max,
    Generator& generator) {
  if constexpr (!Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr(
        min.rows(), min.cols(), [&](const Eigen::Index i, const Eigen::Index j) {
          return generateScalarUniform<typename Derived::Scalar>(min(i, j), max(i, j), generator);
        });
  } else if constexpr (
      !Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime != Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr([&](const Eigen::Index i, const Eigen::Index j) {
      return generateScalarUniform<typename Derived::Scalar>(min(i, j), max(i, j), generator);
    });
  } else if constexpr (
      Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr(min.size(), [&](const Eigen::Index i) {
      return generateScalarUniform<typename Derived::Scalar>(min[i], max[i], generator);
    });
  } else if constexpr (
      Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime != Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr([&](const Eigen::Index i) {
      return generateScalarUniform<typename Derived::Scalar>(min[i], max[i], generator);
    });
  }
};

template <typename T, typename Generator>
[[nodiscard]] T generateUniform(const T& min, const T& max, Generator& generator) {
  if constexpr (std::is_arithmetic_v<T>) {
    return generateScalarUniform(min, max, generator);
  } else if constexpr (is_base_of_matrix_v<T>) {
    return generateMatrixUniform(min, max, generator);
  }
}

template <typename T, typename Generator>
[[nodiscard]] T generateScalarNormal(const T& mean, const T& sigma, Generator& generator) {
  if constexpr (std::is_floating_point_v<T>) {
    NormalRealDist<T> dist(mean, sigma);
    return dist(generator);
  } else if constexpr (is_compatible_to_uniform_int_distribution_v<T>) {
    // No discrete normal distribution exists in <random>; sample from a float
    // normal then round to the nearest integer. TODO: this can overflow T for
    // tail samples — `std::round` returns a float that is then implicitly
    // narrowed without bounds checking. Also, using float (not double) limits
    // precision for 64-bit integer T.
    const float realNumber = NormalRealDist<float>(mean, sigma)(generator);
    return std::round(realNumber);
  }
}

// Generates a random vector/matrix element-wise. Each element is drawn
// independently — this is NOT a multivariate normal (no covariance structure);
// `sigma` supplies a per-element standard deviation.
template <typename Derived, typename Generator>
[[nodiscard]] typename Derived::PlainObject generateMatrixNormal(
    const Eigen::MatrixBase<Derived>& mean,
    const Eigen::MatrixBase<Derived>& sigma,
    Generator& generator) {
  if constexpr (!Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr(
        mean.rows(), mean.cols(), [&](const Eigen::Index i, const Eigen::Index j) {
          return generateScalarNormal<typename Derived::Scalar>(mean(i, j), sigma(i, j), generator);
        });
  } else if constexpr (
      !Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime != Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr([&](const Eigen::Index i, const Eigen::Index j) {
      return generateScalarNormal<typename Derived::Scalar>(mean(i, j), sigma(i, j), generator);
    });
  } else if constexpr (
      Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr(mean.size(), [&](const Eigen::Index i) {
      return generateScalarNormal<typename Derived::Scalar>(mean[i], sigma[i], generator);
    });
  } else if constexpr (
      Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime != Eigen::Dynamic) {
    return Derived::PlainObject::NullaryExpr([&](const Eigen::Index i) {
      return generateScalarNormal<typename Derived::Scalar>(mean[i], sigma[i], generator);
    });
  }
};

template <typename T, typename Generator>
[[nodiscard]] T generateNormal(const T& mean, const T& sigma, Generator& generator) {
  if constexpr (std::is_arithmetic_v<T>) {
    return generateScalarNormal(mean, sigma, generator);
  } else if constexpr (is_base_of_matrix_v<T>) {
    return generateMatrixNormal(mean, sigma, generator);
  }
}

} // namespace detail

template <typename Generator>
Random<Generator>& Random<Generator>::GetSingleton() {
  static Random<Generator> singleton;
  return singleton;
}

template <typename Generator>
Random<Generator>::Random(uint32_t seed) : seed_(seed), generator_(seed_) {}

template <typename Generator>
template <typename T>
T Random<Generator>::uniform(const T& min, const T& max) {
  return detail::generateUniform(min, max, generator_);
}

template <typename Generator>
template <typename FixedSizeT>
FixedSizeT Random<Generator>::uniform(
    typename FixedSizeT::Scalar min,
    typename FixedSizeT::Scalar max) {
  return detail::generateMatrixUniform(
      FixedSizeT::Constant(min), FixedSizeT::Constant(max), generator_);
}

template <typename Generator>
template <typename DynamicVector>
DynamicVector Random<Generator>::uniform(
    Eigen::Index size,
    typename DynamicVector::Scalar min,
    typename DynamicVector::Scalar max) {
  return detail::generateMatrixUniform(
      DynamicVector::Constant(size, min), DynamicVector::Constant(size, max), generator_);
}

template <typename Generator>
template <typename DynamicMatrix>
DynamicMatrix Random<Generator>::uniform(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar min,
    typename DynamicMatrix::Scalar max) {
  return detail::generateMatrixUniform(
      DynamicMatrix::Constant(rows, cols, min),
      DynamicMatrix::Constant(rows, cols, max),
      generator_);
}

template <typename Generator>
template <typename T>
Quaternion<T> Random<Generator>::uniformQuaternion() {
  return Quaternion<T>::UnitRandom();
}

template <typename Generator>
template <typename T>
Matrix3<T> Random<Generator>::uniformRotationMatrix() {
  return uniformQuaternion<T>().toRotationMatrix();
}

template <typename Generator>
template <typename T>
Isometry3<T> Random<Generator>::uniformIsometry3(const Vector3<T>& min, const Vector3<T>& max) {
  Isometry3<T> out = Isometry3<T>::Identity();
  out.linear() = uniformRotationMatrix<T>();
  out.translation() = uniform<Vector3<T>>(min, max);
  return out;
}

template <typename Generator>
template <typename T>
Affine3<T> Random<Generator>::uniformAffine3(
    T scaleMin,
    T scaleMax,
    const Vector3<T>& min,
    const Vector3<T>& max) {
  Affine3<T> out = Affine3<T>::Identity();
  // Linear part is rotation * isotropic_scale; this samples a single scale
  // factor applied to all three axes, not a full anisotropic affine.
  // TODO: the public name `uniformAffine3` is misleading — it does not
  // sample uniformly over the space of affine transforms (which would need
  // anisotropic scale and shear); consider renaming or extending.
  out.linear() = uniformRotationMatrix<T>() * uniform<T>(scaleMin, scaleMax);
  out.translation().noalias() = uniform<Vector3<T>>(min, max);
  return out;
}

template <typename Generator>
template <typename T>
T Random<Generator>::normal(const T& mean, const T& sigma) {
  return detail::generateNormal(mean, sigma, generator_);
}

template <typename Generator>
template <typename FixedSizeT>
FixedSizeT Random<Generator>::normal(
    typename FixedSizeT::Scalar mean,
    typename FixedSizeT::Scalar sigma) {
  return detail::generateMatrixNormal(
      FixedSizeT::Constant(mean), FixedSizeT::Constant(sigma), generator_);
}

template <typename Generator>
template <typename DynamicVector>
DynamicVector Random<Generator>::normal(
    Eigen::Index size,
    typename DynamicVector::Scalar mean,
    typename DynamicVector::Scalar sigma) {
  return detail::generateMatrixNormal(
      DynamicVector::Constant(size, mean), DynamicVector::Constant(size, sigma), generator_);
}

template <typename Generator>
template <typename DynamicMatrix>
DynamicMatrix Random<Generator>::normal(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar mean,
    typename DynamicMatrix::Scalar sigma) {
  return detail::generateMatrixNormal(
      DynamicMatrix::Constant(rows, cols, mean),
      DynamicMatrix::Constant(rows, cols, sigma),
      generator_);
}

template <typename Generator>
uint32_t Random<Generator>::getSeed() const {
  return seed_;
}

template <typename Generator>
void Random<Generator>::setSeed(uint32_t seed) {
  if (seed == seed_) {
    return;
  }
  seed_ = seed;
  generator_.seed(seed_);
}

template <typename T>
T uniform(const T& min, const T& max) {
  auto& rand = Random<>::GetSingleton();
  return rand.uniform(min, max);
}

template <typename FixedSizeT>
FixedSizeT uniform(typename FixedSizeT::Scalar min, typename FixedSizeT::Scalar max) {
  auto& rand = Random<>::GetSingleton();
  return rand.uniform<FixedSizeT>(min, max);
}

template <typename DynamicVector>
DynamicVector
uniform(Eigen::Index size, typename DynamicVector::Scalar min, typename DynamicVector::Scalar max) {
  auto& rand = Random<>::GetSingleton();
  return rand.uniform<DynamicVector>(size, min, max);
}

template <typename DynamicMatrix>
DynamicMatrix uniform(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar min,
    typename DynamicMatrix::Scalar max) {
  auto& rand = Random<>::GetSingleton();
  return rand.uniform<DynamicMatrix>(rows, cols, min, max);
}

template <typename T>
Quaternion<T> uniformQuaternion() {
  auto& rand = Random<>::GetSingleton();
  return rand.uniformQuaternion<T>();
}

template <typename T>
Matrix3<T> uniformRotationMatrix() {
  return uniformQuaternion<T>().toRotationMatrix();
}

template <typename T>
Isometry3<T> uniformIsometry3(const Vector3<T>& min, const Vector3<T>& max) {
  auto& rand = Random<>::GetSingleton();
  return rand.uniformIsometry3<T>(min, max);
}

template <typename T>
Affine3<T> uniformAffine3(T scaleMin, T scaleMax, const Vector3<T>& min, const Vector3<T>& max) {
  auto& rand = Random<>::GetSingleton();
  return rand.uniformAffine3<T>(scaleMin, scaleMax, min, max);
}

template <typename T>
T normal(const T& mean, const T& sigma) {
  auto& rand = Random<>::GetSingleton();
  return rand.normal(mean, sigma);
}

template <typename FixedSizeT>
FixedSizeT normal(typename FixedSizeT::Scalar mean, typename FixedSizeT::Scalar sigma) {
  auto& rand = Random<>::GetSingleton();
  return rand.normal<FixedSizeT>(mean, sigma);
}

template <typename DynamicVector>
DynamicVector normal(
    Eigen::Index size,
    typename DynamicVector::Scalar mean,
    typename DynamicVector::Scalar sigma) {
  auto& rand = Random<>::GetSingleton();
  return rand.normal<DynamicVector>(size, mean, sigma);
}

template <typename DynamicMatrix>
DynamicMatrix normal(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar mean,
    typename DynamicMatrix::Scalar sigma) {
  auto& rand = Random<>::GetSingleton();
  return rand.normal<DynamicMatrix>(rows, cols, mean, sigma);
}

} // namespace momentum
