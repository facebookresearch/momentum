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

/// A consolidated random number generator to provide convenient APIs wrapping around the random
/// library of C++ standard library <random>.
///
/// The default engine used in this class is std::mt19937, but it can be replaced, by specifying the
/// template argument, with any preferable engines (see:
/// https://en.cppreference.com/w/cpp/numeric/random).
template <typename Generator_ = std::mt19937>
class Random final {
 public:
  using Generator = Generator_;

  /// Returns the singleton instance
  [[nodiscard]] static Random& GetSingleton();

  /// Constructor
  explicit Random(uint32_t seed = std::random_device{}());

  /// Generates a random scalar/vector/matrix from the uniform distribution
  ///
  /// The supported types are scalar, vector, or matrix where the elements can be either integer or
  /// real. The supported integer types are whichever supported by std::uniform_int_distribution<T>
  /// (see: https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution), and the
  /// supported real types are whichever std::is_floating_point<T>::value is true (see:
  /// https://en.cppreference.com/w/cpp/types/is_floating_point).
  ///
  /// The generated random number is in the range of [min, max] for integer types or [min, max) for
  /// real types. However, when built with the -ffast-math flag, the range for real types could be
  /// [min, max] instead.
  ///
  /// @warning While the upper bound is typically exclusive for real types according to
  /// std::uniform_real_distribution, there may be instances where the sampled value equals the
  /// upper bound, depending on the platform and compiler specifics.
  ///
  /// For vector/matrix types, the bounds can be specified in either scalar or (the same dimension
  /// of) vector/matrix. Use this function to specify the bounds in vector/matrix to apply different
  /// bounds for each element in the generated vector/matrix. Use other versions of uniform() that
  /// specify the bounds in scalar to apply the same bounds to all the elements of the generated
  /// vector/matrix.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform(0.0, 1.0); // random double in [0, 1)
  /// auto r1 = rand.uniform(0.f, 1.f); // random float in [0, 1)
  /// auto r3 = rand.uniform(0, 10);    // random int in [0, 10]
  /// auto r4 = rand.uniform(Vector2f(0, 1), Vector2f(2, 3)); // random Vector2f in ([0, 2], [1, 3))
  /// @endcode
  ///
  /// @tparam T The random number type to be generated
  /// @param[in] min The lower bound of the random number
  /// @param[in] max The upper bound of the random number
  template <typename T>
  [[nodiscard]] T uniform(const T& min, const T& max);

  /// Generates a random vector/matrix from the uniform distribution, where the size is fixed
  ///
  /// This is a special case of T uniform(const T&, const T&) where the random number type is fixed
  /// size of vector/matrix and the bounds are specified by the scalar type.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform<Vector2f>(0, 1); // random Vector2f in ([0, 1], [0, 1))
  /// @endcode
  ///
  /// @tparam FixedSizeT The random vector/matrix type to be generated
  /// @param[in] min The lower bound in scalar.
  /// @param[in] max The upper bound in scalar.
  template <typename FixedSizeT>
  [[nodiscard]] FixedSizeT uniform(
      typename FixedSizeT::Scalar min,
      typename FixedSizeT::Scalar max);

  /// Generates a random vector from the uniform distribution, where the size is dynamic
  ///
  /// This is a special case of T uniform(const T&, const T&) where the random number type is
  /// dynamic size of vector and the bounds are specified by the scalar type.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform<VectorXf>(2, 0, 1); // random VectorXf of dimension 2 where each
  /// element is in [0, 1)
  /// @endcode
  ///
  /// @tparam DynamicVector The random vector type to be generated
  /// @param[in] size The size of the vector.
  /// @param[in] min The lower bound in scalar.
  /// @param[in] max The upper bound in scalar.
  template <typename DynamicVector>
  [[nodiscard]] DynamicVector uniform(
      Eigen::Index size,
      typename DynamicVector::Scalar min,
      typename DynamicVector::Scalar max);

  /// Generates a random matrix from the uniform distribution, where the size is dynamic
  ///
  /// This is a special case of T uniform(const T&, const T&) where the random number type is
  /// dynamic size of matrix and the bounds are specified by the scalar type.
  ///
  /// @code
  /// Random rand;
  /// auto r0 = rand.uniform<MatrixXf>(2, 3, 0, 1); // random MatrixXf of dimension 2x3 where each
  /// element is in [0, 1)
  /// @endcode
  ///
  /// @tparam DynamicMatrix The random matrix type to be generated
  /// @param[in] rows The row size of the matrix.
  /// @param[in] cols The column size of the matrix.
  /// @param[in] min The lower bound in scalar.
  /// @param[in] max The upper bound in scalar.
  template <typename DynamicMatrix>
  [[nodiscard]] DynamicMatrix uniform(
      Eigen::Index rows,
      Eigen::Index cols,
      typename DynamicMatrix::Scalar min,
      typename DynamicMatrix::Scalar max);

  /// Generates a random quaternion from a uniform distribution on SO(3)
  template <typename T>
  [[nodiscard]] Quaternion<T> uniformQuaternion();

  /// Generates a random rotation matrix from a uniform distribution on SO(3)
  template <typename T>
  [[nodiscard]] Matrix3<T> uniformRotationMatrix();

  /// Generates a random isometry (rigid transformation) from a uniform distribution on SE(3)
  /// based on input minimum and maximum 3D vectors for the translation part.
  ///
  /// @tparam Generator The type of random number generator.
  /// @tparam T The type of isometry component, typically a float or double.
  /// @param min The minimum 3D vector for the translation part.
  /// @param max The maximum 3D vector for the translation part.
  /// @return A random Isometry3 of type T.
  template <typename T>
  [[nodiscard]] Isometry3<T> uniformIsometry3(
      const Vector3<T>& min = Vector3<T>::Zero(),
      const Vector3<T>& max = Vector3<T>::Ones());

  /// Generates a random affine transformation from a uniform distribution on the space of all
  /// affine transformations.
  ///
  /// The transformation is created based on input minimum and maximum 3D vectors for the
  /// translation part, and minimum and maximum scale factors for the linear part.
  ///
  /// @tparam Generator The type of random number generator.
  /// @tparam T The type of affine transformation component, typically a float or double.
  /// @param scaleMin The minimum scale factor for the linear part.
  /// @param scaleMax The maximum scale factor for the linear part.
  /// @param min The minimum 3D vector for the translation part.
  /// @param max The maximum 3D vector for the translation part.
  /// @return A random Affine3 of type T.
  template <typename T>
  [[nodiscard]] Affine3<T> uniformAffine3(
      T scaleMin = 0.1,
      T scaleMax = 10.0,
      const Vector3<T>& min = Vector3<T>::Zero(),
      const Vector3<T>& max = Vector3<T>::Ones());

  /// Generates a random value from the Gaussian distribution
  ///
  /// @tparam T The random number type to be generated
  /// @param[in] mean The mean of the Gaussian distribution
  /// @param[in] sigma The standard deviation of the Gaussian distribution
  template <typename T>
  [[nodiscard]] T normal(const T& mean, const T& sigma);

  /// Generates a random value from the Gaussian distribution
  ///
  /// @tparam FixedSizeT The random vector/matrix type to be generated
  /// @param[in] mean The mean of the Gaussian distribution
  /// @param[in] sigma The standard deviation of the Gaussian distribution
  template <typename FixedSizeT>
  [[nodiscard]] FixedSizeT normal(
      typename FixedSizeT::Scalar mean,
      typename FixedSizeT::Scalar sigma);

  /// Generates a random value from the Gaussian distribution
  ///
  /// @tparam DynamicVector The random vector type to be generated
  /// @param[in] size The size of the vector.
  /// @param[in] mean The mean of the Gaussian distribution
  /// @param[in] sigma The standard deviation of the Gaussian distribution
  template <typename DynamicVector>
  [[nodiscard]] DynamicVector normal(
      Eigen::Index size,
      typename DynamicVector::Scalar mean,
      typename DynamicVector::Scalar sigma);

  /// Generates a random value from the Gaussian distribution
  ///
  /// @tparam DynamicMatrix The random matrix type to be generated
  /// @param[in] rows The row size of the matrix.
  /// @param[in] cols The column size of the matrix.
  /// @param[in] mean The mean of the Gaussian distribution
  /// @param[in] sigma The standard deviation of the Gaussian distribution
  template <typename DynamicMatrix>
  [[nodiscard]] DynamicMatrix normal(
      Eigen::Index rows,
      Eigen::Index cols,
      typename DynamicMatrix::Scalar mean,
      typename DynamicMatrix::Scalar sigma);

  /// Returns the seed.
  [[nodiscard]] uint32_t getSeed() const;

  /// Sets a new seed for the internal random number engine.
  void setSeed(uint32_t seed);

 private:
  /// The seed used for the internal random number engine.
  uint32_t seed_;

  /// The internal random number engine.
  Generator generator_;
};

/// Generates a random type T from the uniform distribution, using the global random number
/// generator Random
template <typename T>
[[nodiscard]] T uniform(const T& min, const T& max);

/// Generates a random fixed size vector/matrix from the uniform distribution, using the global
/// random number generator Random
template <typename FixedSizeT>
[[nodiscard]] FixedSizeT uniform(typename FixedSizeT::Scalar min, typename FixedSizeT::Scalar max);

/// Generates a random dynamic size vector from the uniform distribution, using the global random
/// number generator Random
template <typename DynamicVector>
[[nodiscard]] DynamicVector
uniform(Eigen::Index size, typename DynamicVector::Scalar min, typename DynamicVector::Scalar max);

/// Generates a random type dynamic size matrix from the uniform distribution, using the global
/// random number generator Random
template <typename DynamicMatrix>
[[nodiscard]] DynamicMatrix uniform(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar min,
    typename DynamicMatrix::Scalar max);

/// Generates a random quaternion from a uniform distribution on SO(3)
template <typename T>
[[nodiscard]] Quaternion<T> uniformQuaternion();

/// Generates a random rotation matrix from a uniform distribution on SO(3)
template <typename T>
[[nodiscard]] Matrix3<T> uniformRotationMatrix();

/// Generates a random isometry (rigid transformation) from a uniform distribution on SE(3)
/// based on input minimum and maximum 3D vectors for the translation part.
template <typename T>
[[nodiscard]] Isometry3<T> uniformIsometry3(
    const Vector3<T>& min = Vector3<T>::Zero(),
    const Vector3<T>& max = Vector3<T>::Ones());

/// Generates a random affine transformation from a uniform distribution on the space of all
/// affine transformations.
template <typename T>
[[nodiscard]] Affine3<T> uniformAffine3(
    T scaleMin = 0.1,
    T scaleMax = 10.0,
    const Vector3<T>& min = Vector3<T>::Zero(),
    const Vector3<T>& max = Vector3<T>::Ones());

/// Generates a random type T from the Gaussian distribution, using the global random number
/// generator Random
template <typename T>
[[nodiscard]] T normal(const T& mean, const T& sigma);

/// Generates a random fixed size vector/matrix from the Gaussian distribution, using the global
/// random number generator Random
template <typename FixedSizeT>
[[nodiscard]] FixedSizeT normal(
    typename FixedSizeT::Scalar mean,
    typename FixedSizeT::Scalar sigma);

/// Generates a random dynamic size vector from the Gaussian distribution, using the global random
/// number generator Random
template <typename DynamicVector>
[[nodiscard]] DynamicVector normal(
    Eigen::Index size,
    typename DynamicVector::Scalar mean,
    typename DynamicVector::Scalar sigma);

/// Generates a random dynamic size matrix from the Gaussian distribution, using the global random
/// number generator Random
template <typename DynamicMatrix>
[[nodiscard]] DynamicMatrix normal(
    Eigen::Index rows,
    Eigen::Index cols,
    typename DynamicMatrix::Scalar mean,
    typename DynamicMatrix::Scalar sigma);

} // namespace momentum

#include <momentum/math/random-inl.h>
