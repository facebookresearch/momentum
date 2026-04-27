/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

#include <random>

namespace momentum {

/// A consolidated random number generator providing convenient APIs wrapping the C++ standard
/// library `<random>`.
///
/// The default engine is `std::mt19937`; any engine satisfying the C++ UniformRandomBitGenerator
/// requirements can be substituted via the template parameter (see:
/// https://en.cppreference.com/w/cpp/numeric/random).
///
/// @note Reproducibility: given the same seed and the same Generator type, the sequence of
/// engine outputs is deterministic. However, the values produced by `uniform()` / `normal()` are
/// **not** guaranteed to be portable across platforms or standard library implementations because
/// `std::uniform_int_distribution`, `std::uniform_real_distribution`, and
/// `std::normal_distribution` are implementation-defined.
///
/// @warning Thread safety: instances are **not** thread-safe. The internal generator holds mutable
/// state that is mutated on every call. Concurrent use of the same instance (including the
/// singleton returned by GetSingleton()) requires external synchronization.
template <typename Generator_ = std::mt19937>
class Random final {
 public:
  using Generator = Generator_;

  /// Returns the process-wide singleton instance.
  ///
  /// @warning The returned instance is shared across all callers and is **not** thread-safe.
  /// See class-level thread-safety note.
  // TODO: Document or enforce the seeding policy of the singleton (currently seeded once via
  // std::random_device on first call, which makes test reproducibility difficult without an
  // explicit setSeed()).
  [[nodiscard]] static Random& GetSingleton();

  /// Constructs a generator seeded with the given value.
  ///
  /// @param seed Seed for the underlying engine. Defaults to a non-deterministic value drawn from
  /// `std::random_device`, which makes default-constructed instances non-reproducible.
  explicit Random(uint32_t seed = std::random_device{}());

  /// Generates a random scalar/vector/matrix from the uniform distribution.
  ///
  /// Supported element types are integer types accepted by `std::uniform_int_distribution<T>`
  /// (see: https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution) and any type
  /// for which `std::is_floating_point<T>::value` is true (see:
  /// https://en.cppreference.com/w/cpp/types/is_floating_point). T may be a scalar, a fixed-size
  /// Eigen vector/matrix, or a dynamic-size Eigen vector/matrix (in which case the result has the
  /// same dimensions as `min`).
  ///
  /// @pre `min <= max` element-wise.
  /// @pre For dynamic-size types, `min` and `max` must have matching dimensions.
  ///
  /// The result lies in `[min, max]` for integer types and `[min, max)` for real types.
  ///
  /// @warning While the upper bound is mathematically exclusive for real types according to
  /// `std::uniform_real_distribution`, there are instances where the sampled value can equal the
  /// upper bound:
  ///   - When built with `-ffast-math`, rounding can yield exactly `max`.
  ///   - Some standard library implementations have known edge cases producing the upper bound.
  ///
  /// For vector/matrix types, supplying vector/matrix bounds applies a per-element range; use the
  /// scalar-bounds overload to apply the same range to every element.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform(0.0, 1.0); // random double in [0, 1)
  /// auto r1 = rand.uniform(0.f, 1.f); // random float in [0, 1)
  /// auto r3 = rand.uniform(0, 10);    // random int in [0, 10]
  /// auto r4 = rand.uniform(Vector2f(0, 1), Vector2f(2, 3)); // random Vector2f in ([0, 2], [1, 3))
  /// @endcode
  ///
  /// @tparam T Scalar, fixed-size, or dynamic-size Eigen type.
  /// @param[in] min Inclusive lower bound (element-wise).
  /// @param[in] max Upper bound (inclusive for integer types, exclusive for real types).
  template <typename T>
  [[nodiscard]] T uniform(const T& min, const T& max);

  /// Generates a fixed-size vector/matrix from the uniform distribution with a single scalar
  /// range applied to every element.
  ///
  /// Same range semantics as `uniform(const T&, const T&)`.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform<Vector2f>(0, 1); // random Vector2f in ([0, 1), [0, 1))
  /// @endcode
  ///
  /// @tparam FixedSizeT Fixed-size Eigen vector/matrix type.
  /// @param[in] min Inclusive lower bound applied to every element.
  /// @param[in] max Upper bound applied to every element (inclusive for integer scalars,
  /// exclusive for real scalars).
  // TODO: The original example claimed `([0, 1], [0, 1))` which is inconsistent — both elements
  // share the same distribution and same bounds.
  template <typename FixedSizeT>
  [[nodiscard]] FixedSizeT uniform(
      typename FixedSizeT::Scalar min,
      typename FixedSizeT::Scalar max);

  /// Generates a dynamic-size vector from the uniform distribution with a single scalar range
  /// applied to every element.
  ///
  /// Same range semantics as `uniform(const T&, const T&)`.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform<VectorXf>(2, 0, 1); // random VectorXf of dimension 2 where each
  /// element is in [0, 1)
  /// @endcode
  ///
  /// @tparam DynamicVector Dynamic-size Eigen vector type.
  /// @param[in] size Number of elements in the result.
  /// @param[in] min Inclusive lower bound applied to every element.
  /// @param[in] max Upper bound applied to every element (inclusive for integer scalars,
  /// exclusive for real scalars).
  /// @pre `size >= 0`.
  template <typename DynamicVector>
  [[nodiscard]] DynamicVector uniform(
      Eigen::Index size,
      typename DynamicVector::Scalar min,
      typename DynamicVector::Scalar max);

  /// Generates a dynamic-size matrix from the uniform distribution with a single scalar range
  /// applied to every element.
  ///
  /// Same range semantics as `uniform(const T&, const T&)`.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform<MatrixXf>(2, 3, 0, 1); // random MatrixXf of dimension 2x3 where each
  /// element is in [0, 1)
  /// @endcode
  ///
  /// @tparam DynamicMatrix Dynamic-size Eigen matrix type.
  /// @param[in] rows Number of rows in the result.
  /// @param[in] cols Number of columns in the result.
  /// @param[in] min Inclusive lower bound applied to every element.
  /// @param[in] max Upper bound applied to every element (inclusive for integer scalars,
  /// exclusive for real scalars).
  /// @pre `rows >= 0` and `cols >= 0`.
  template <typename DynamicMatrix>
  [[nodiscard]] DynamicMatrix uniform(
      Eigen::Index rows,
      Eigen::Index cols,
      typename DynamicMatrix::Scalar min,
      typename DynamicMatrix::Scalar max);

  /// Generates a random unit quaternion uniformly distributed on SO(3) (Haar measure).
  template <typename T>
  [[nodiscard]] Quaternion<T> uniformQuaternion();

  /// Generates a random rotation matrix uniformly distributed on SO(3) (Haar measure).
  template <typename T>
  [[nodiscard]] Matrix3<T> uniformRotationMatrix();

  /// Generates a random isometry (rigid transformation) on SE(3) with rotation drawn uniformly
  /// from SO(3) and translation drawn uniformly from the box `[min, max]`.
  ///
  /// @tparam T Scalar type of the isometry components, typically `float` or `double`.
  /// @param min Inclusive lower bound on the translation component (per axis).
  /// @param max Inclusive upper bound on the translation component (per axis).
  /// @return A random Isometry3<T>.
  template <typename T>
  [[nodiscard]] Isometry3<T> uniformIsometry3(
      const Vector3<T>& min = Vector3<T>::Zero(),
      const Vector3<T>& max = Vector3<T>::Ones());

  /// Generates a random affine transformation with rotation drawn uniformly from SO(3), uniform
  /// scale drawn from `[scaleMin, scaleMax]`, and translation drawn uniformly from the box
  /// `[min, max]`.
  ///
  /// @tparam T Scalar type of the affine components, typically `float` or `double`.
  /// @param scaleMin Inclusive lower bound on the uniform scale factor.
  /// @param scaleMax Inclusive upper bound on the uniform scale factor.
  /// @param min Inclusive lower bound on the translation component (per axis).
  /// @param max Inclusive upper bound on the translation component (per axis).
  /// @return A random Affine3<T>.
  // TODO: The default `scaleMin = 0.1` / `scaleMax = 10.0` only makes sense when T is a floating
  // type; instantiation with an integer T would silently truncate the defaults.
  template <typename T>
  [[nodiscard]] Affine3<T> uniformAffine3(
      T scaleMin = 0.1,
      T scaleMax = 10.0,
      const Vector3<T>& min = Vector3<T>::Zero(),
      const Vector3<T>& max = Vector3<T>::Ones());

  /// Generates a value from a normal (Gaussian) distribution N(mean, sigma^2). Each element of a
  /// vector/matrix result is drawn independently.
  ///
  /// Backed by `std::normal_distribution`, which requires a real (floating-point) scalar type.
  ///
  /// @pre `sigma >= 0`.
  ///
  /// @tparam T Scalar, fixed-size, or dynamic-size Eigen type with floating-point scalar.
  /// @param[in] mean Distribution mean (mu).
  /// @param[in] sigma Distribution standard deviation (sigma).
  template <typename T>
  [[nodiscard]] T normal(const T& mean, const T& sigma);

  /// Generates a fixed-size vector/matrix with each element drawn independently from
  /// N(mean, sigma^2).
  ///
  /// @pre `sigma >= 0`.
  ///
  /// @tparam FixedSizeT Fixed-size Eigen vector/matrix type with floating-point scalar.
  /// @param[in] mean Distribution mean applied to every element.
  /// @param[in] sigma Distribution standard deviation applied to every element.
  template <typename FixedSizeT>
  [[nodiscard]] FixedSizeT normal(
      typename FixedSizeT::Scalar mean,
      typename FixedSizeT::Scalar sigma);

  /// Generates a dynamic-size vector with each element drawn independently from N(mean, sigma^2).
  ///
  /// @pre `sigma >= 0` and `size >= 0`.
  ///
  /// @tparam DynamicVector Dynamic-size Eigen vector type with floating-point scalar.
  /// @param[in] size Number of elements in the result.
  /// @param[in] mean Distribution mean applied to every element.
  /// @param[in] sigma Distribution standard deviation applied to every element.
  template <typename DynamicVector>
  [[nodiscard]] DynamicVector normal(
      Eigen::Index size,
      typename DynamicVector::Scalar mean,
      typename DynamicVector::Scalar sigma);

  /// Generates a dynamic-size matrix with each element drawn independently from N(mean, sigma^2).
  ///
  /// @pre `sigma >= 0`, `rows >= 0`, and `cols >= 0`.
  ///
  /// @tparam DynamicMatrix Dynamic-size Eigen matrix type with floating-point scalar.
  /// @param[in] rows Number of rows in the result.
  /// @param[in] cols Number of columns in the result.
  /// @param[in] mean Distribution mean applied to every element.
  /// @param[in] sigma Distribution standard deviation applied to every element.
  template <typename DynamicMatrix>
  [[nodiscard]] DynamicMatrix normal(
      Eigen::Index rows,
      Eigen::Index cols,
      typename DynamicMatrix::Scalar mean,
      typename DynamicMatrix::Scalar sigma);

  /// Returns the seed last used to initialize the engine.
  ///
  /// @note This reflects the value passed to the constructor or the most recent `setSeed()` call;
  /// it does not encode the engine's current state, so it cannot be used alone to resume a
  /// sequence after some samples have been drawn.
  [[nodiscard]] uint32_t getSeed() const;

  /// Re-seeds the internal engine, resetting its state to the deterministic sequence implied by
  /// `seed`.
  void setSeed(uint32_t seed);

 private:
  uint32_t seed_;

  Generator generator_;
};

/// Convenience wrappers that forward to `Random<>::GetSingleton()`.
///
/// @warning These share the singleton's mutable engine state and inherit its non-thread-safe
/// behavior; concurrent calls from different threads must be externally synchronized. Range and
/// distribution semantics match the corresponding `Random` member functions.

/// Uniformly distributed value via the global `Random` singleton. See `Random::uniform`.
template <typename T>
[[nodiscard]] T uniform(const T& min, const T& max);

/// Uniformly distributed fixed-size vector/matrix via the global `Random` singleton.
/// See `Random::uniform`.
template <typename FixedSizeT>
[[nodiscard]] FixedSizeT uniform(typename FixedSizeT::Scalar min, typename FixedSizeT::Scalar max);

/// Uniformly distributed dynamic-size vector via the global `Random` singleton.
/// See `Random::uniform`. @pre `size >= 0`.
template <typename DynamicVector>
[[nodiscard]] DynamicVector
uniform(Eigen::Index size, typename DynamicVector::Scalar min, typename DynamicVector::Scalar max);

/// Uniformly distributed dynamic-size matrix via the global `Random` singleton.
/// See `Random::uniform`. @pre `rows >= 0` and `cols >= 0`.
template <typename DynamicMatrix>
[[nodiscard]] DynamicMatrix uniform(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar min,
    typename DynamicMatrix::Scalar max);

/// Uniformly distributed unit quaternion on SO(3) via the global `Random` singleton.
template <typename T>
[[nodiscard]] Quaternion<T> uniformQuaternion();

/// Uniformly distributed rotation matrix on SO(3) via the global `Random` singleton.
template <typename T>
[[nodiscard]] Matrix3<T> uniformRotationMatrix();

/// Random isometry on SE(3) via the global `Random` singleton (uniform rotation, uniform
/// translation in `[min, max]`).
template <typename T>
[[nodiscard]] Isometry3<T> uniformIsometry3(
    const Vector3<T>& min = Vector3<T>::Zero(),
    const Vector3<T>& max = Vector3<T>::Ones());

/// Random affine transformation via the global `Random` singleton (uniform rotation, uniform
/// scale in `[scaleMin, scaleMax]`, uniform translation in `[min, max]`).
template <typename T>
[[nodiscard]] Affine3<T> uniformAffine3(
    T scaleMin = 0.1,
    T scaleMax = 10.0,
    const Vector3<T>& min = Vector3<T>::Zero(),
    const Vector3<T>& max = Vector3<T>::Ones());

/// Normally distributed value via the global `Random` singleton. See `Random::normal`.
/// @pre `sigma >= 0`.
template <typename T>
[[nodiscard]] T normal(const T& mean, const T& sigma);

/// Normally distributed fixed-size vector/matrix via the global `Random` singleton.
/// See `Random::normal`. @pre `sigma >= 0`.
template <typename FixedSizeT>
[[nodiscard]] FixedSizeT normal(
    typename FixedSizeT::Scalar mean,
    typename FixedSizeT::Scalar sigma);

/// Normally distributed dynamic-size vector via the global `Random` singleton.
/// See `Random::normal`. @pre `sigma >= 0` and `size >= 0`.
template <typename DynamicVector>
[[nodiscard]] DynamicVector normal(
    Eigen::Index size,
    typename DynamicVector::Scalar mean,
    typename DynamicVector::Scalar sigma);

/// Normally distributed dynamic-size matrix via the global `Random` singleton.
/// See `Random::normal`. @pre `sigma >= 0`, `rows >= 0`, and `cols >= 0`.
template <typename DynamicMatrix>
[[nodiscard]] DynamicMatrix normal(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar mean,
    typename DynamicMatrix::Scalar sigma);

} // namespace momentum

#include <momentum/math/random-inl.h>
