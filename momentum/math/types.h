/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <bitset>
#include <cstdint>

//------------------------------------------------------------------------------
// Momentum specific Eigen defines
//------------------------------------------------------------------------------

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4456) // hidden local declaration
#pragma warning(disable : 4702) // unreachable code
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <span>

//------------------------------------------------------------------------------
// Define some additional Eigen types
//------------------------------------------------------------------------------

namespace Eigen {
using Vector3b = Vector3<uint8_t>;
}

namespace momentum {

inline constexpr int Dynamic = Eigen::Dynamic;

template <class T>
using Map = Eigen::Map<T>;

template <class T>
using Ref = Eigen::Ref<T>;

template <typename Scalar, int Dim>
using Line = Eigen::ParametrizedLine<Scalar, Dim>;

template <typename Scalar, int Dim>
using Plane = Eigen::Hyperplane<Scalar, Dim>;

template <typename Scalar, int Dim>
using Box = Eigen::AlignedBox<Scalar, Dim>;

template <int Dim>
using Boxf = Eigen::AlignedBox<float, Dim>;

template <int Dim>
using Boxd = Eigen::AlignedBox<double, Dim>;

template <int Dim>
using Boxi = Eigen::AlignedBox<std::int32_t, Dim>;

template <typename T>
using Box2 = Eigen::AlignedBox<T, 2>;
template <typename T>
using Box3 = Eigen::AlignedBox<T, 3>;

using Box2f = Box2<float>;
using Box3f = Box3<float>;

using Box2d = Box2<double>;
using Box3d = Box3<double>;

using Box2i = Box2<std::int32_t>;
using Box3i = Box3<std::int32_t>;

template <typename T>
using MatrixX = Eigen::Matrix<T, Dynamic, Dynamic>;

using MatrixXf = MatrixX<float>;
using MatrixXd = MatrixX<double>;
using MatrixXb = MatrixX<std::uint8_t>;
using MatrixXs = MatrixX<std::uint16_t>;
using MatrixXu = MatrixX<std::uint32_t>;
using MatrixXi = MatrixX<std::int32_t>;

template <typename T>
using Matrix0 = Eigen::Matrix<T, 0, 0>;
template <typename T>
using Matrix1 = Eigen::Matrix<T, 1, 1>;
template <typename T>
using Matrix2 = Eigen::Matrix<T, 2, 2>;
template <typename T>
using Matrix3 = Eigen::Matrix<T, 3, 3>;
template <typename T>
using Matrix4 = Eigen::Matrix<T, 4, 4>;
template <typename T>
using Matrix5 = Eigen::Matrix<T, 5, 5>;
template <typename T>
using Matrix6 = Eigen::Matrix<T, 6, 6>;

using Matrix0f = Matrix0<float>;
using Matrix1f = Matrix1<float>;
using Matrix2f = Matrix2<float>;
using Matrix3f = Matrix3<float>;
using Matrix4f = Matrix4<float>;
using Matrix5f = Matrix5<float>;
using Matrix6f = Matrix6<float>;

using Matrix0d = Matrix0<double>;
using Matrix1d = Matrix1<double>;
using Matrix2d = Matrix2<double>;
using Matrix3d = Matrix3<double>;
using Matrix4d = Matrix4<double>;
using Matrix5d = Matrix5<double>;
using Matrix6d = Matrix6<double>;

using Matrix0b = Matrix0<std::uint8_t>;
using Matrix1b = Matrix1<std::uint8_t>;
using Matrix2b = Matrix2<std::uint8_t>;
using Matrix3b = Matrix3<std::uint8_t>;
using Matrix4b = Matrix4<std::uint8_t>;
using Matrix5b = Matrix5<std::uint8_t>;
using Matrix6b = Matrix6<std::uint8_t>;

using Matrix0s = Matrix0<std::uint16_t>;
using Matrix1s = Matrix1<std::uint16_t>;
using Matrix2s = Matrix2<std::uint16_t>;
using Matrix3s = Matrix3<std::uint16_t>;
using Matrix4s = Matrix4<std::uint16_t>;
using Matrix5s = Matrix5<std::uint16_t>;
using Matrix6s = Matrix6<std::uint16_t>;

using Matrix0u = Matrix0<std::uint32_t>;
using Matrix1u = Matrix1<std::uint32_t>;
using Matrix2u = Matrix2<std::uint32_t>;
using Matrix3u = Matrix3<std::uint32_t>;
using Matrix4u = Matrix4<std::uint32_t>;
using Matrix5u = Matrix5<std::uint32_t>;
using Matrix6u = Matrix6<std::uint32_t>;

using Matrix0i = Matrix0<std::int32_t>;
using Matrix1i = Matrix1<std::int32_t>;
using Matrix2i = Matrix2<std::int32_t>;
using Matrix3i = Matrix3<std::int32_t>;
using Matrix4i = Matrix4<std::int32_t>;
using Matrix5i = Matrix5<std::int32_t>;
using Matrix6i = Matrix6<std::int32_t>;

template <typename T>
using Matrix3X = Eigen::Matrix<T, 3, Dynamic>;
using Matrix3Xf = Matrix3X<float>;
using Matrix3Xd = Matrix3X<double>;
using Matrix3Xb = Matrix3X<std::uint8_t>;
using Matrix3Xs = Matrix3X<std::uint16_t>;
using Matrix3Xu = Matrix3X<std::uint32_t>;
using Matrix3Xi = Matrix3X<std::int32_t>;

template <typename T>
using MatrixX3 = Eigen::Matrix<T, Dynamic, 3>;
using MatrixX3f = MatrixX3<float>;
using MatrixX3d = MatrixX3<double>;
using MatrixX3b = MatrixX3<std::uint8_t>;
using MatrixX3s = MatrixX3<std::uint16_t>;
using MatrixX3u = MatrixX3<std::uint32_t>;
using MatrixX3i = MatrixX3<std::int32_t>;

template <typename T, int N, int M = N>
using RowMatrix = Eigen::Matrix<T, N, M, Eigen::RowMajor>;
template <typename T>
using RowMatrixX = RowMatrix<T, Dynamic>;

template <typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;
using SparseMatrixf = SparseMatrix<float>;
using SparseMatrixd = SparseMatrix<double>;
using SparseMatrixb = SparseMatrix<std::uint8_t>;
using SparseMatrixs = SparseMatrix<std::uint16_t>;
using SparseMatrixu = SparseMatrix<std::uint32_t>;
using SparseMatrixi = SparseMatrix<std::int32_t>;

template <typename T>
using SparseRowMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;
using SparseRowMatrixf = SparseRowMatrix<float>;
using SparseRowMatrixd = SparseRowMatrix<double>;
using SparseRowMatrixb = SparseRowMatrix<std::uint8_t>;
using SparseRowMatrixs = SparseRowMatrix<std::uint16_t>;
using SparseRowMatrixu = SparseRowMatrix<std::uint32_t>;
using SparseRowMatrixi = SparseRowMatrix<std::int32_t>;

template <typename T, int Dim, int Options = 0>
using Vector = Eigen::Matrix<T, Dim, 1, Options>;

template <typename T>
using VectorX = Vector<T, Dynamic>;

using VectorXf = VectorX<float>;
using VectorXd = VectorX<double>;
using VectorXb = VectorX<std::uint8_t>;
using VectorXs = VectorX<std::uint16_t>;
using VectorXu = VectorX<std::uint32_t>;
using VectorXi = VectorX<std::int32_t>;

template <int Dim>
using Vectorf = Vector<float, Dim>;
template <int Dim>
using Vectord = Vector<double, Dim>;
template <int Dim>
using Vectorb = Vector<std::uint8_t, Dim>;
template <int Dim>
using Vectors = Vector<std::uint16_t, Dim>;
template <int Dim>
using Vectoru = Vector<std::uint32_t, Dim>;
template <int Dim>
using Vectori = Vector<std::int32_t, Dim>;

template <typename T>
using Vector0 = Vector<T, 0>;
template <typename T>
using Vector1 = Vector<T, 1>;
template <typename T>
using Vector2 = Vector<T, 2>;
template <typename T>
using Vector3 = Vector<T, 3>;
template <typename T>
using Vector4 = Vector<T, 4>;
template <typename T>
using Vector5 = Vector<T, 5>;
template <typename T>
using Vector6 = Vector<T, 6>;
template <typename T>
using Vector7 = Vector<T, 7>;
template <typename T>
using Vector8 = Vector<T, 8>;

using Vector0f = Vector0<float>;
using Vector1f = Vector1<float>;
using Vector2f = Vector2<float>;
using Vector3f = Vector3<float>;
using Vector4f = Vector4<float>;
using Vector5f = Vector5<float>;
using Vector6f = Vector6<float>;
using Vector7f = Vector7<float>;
using Vector8f = Vector8<float>;

using Vector0d = Vector0<double>;
using Vector1d = Vector1<double>;
using Vector2d = Vector2<double>;
using Vector3d = Vector3<double>;
using Vector4d = Vector4<double>;
using Vector5d = Vector5<double>;
using Vector6d = Vector6<double>;

using Vector0b = Vector0<std::uint8_t>;
using Vector1b = Vector1<std::uint8_t>;
using Vector2b = Vector2<std::uint8_t>;
using Vector3b = Vector3<std::uint8_t>;
using Vector4b = Vector4<std::uint8_t>;
using Vector5b = Vector5<std::uint8_t>;
using Vector6b = Vector6<std::uint8_t>;

using Vector0s = Vector0<std::uint16_t>;
using Vector1s = Vector1<std::uint16_t>;
using Vector2s = Vector2<std::uint16_t>;
using Vector3s = Vector3<std::uint16_t>;
using Vector4s = Vector4<std::uint16_t>;
using Vector5s = Vector5<std::uint16_t>;
using Vector6s = Vector6<std::uint16_t>;

using Vector0u = Vector0<std::uint32_t>;
using Vector1u = Vector1<std::uint32_t>;
using Vector2u = Vector2<std::uint32_t>;
using Vector3u = Vector3<std::uint32_t>;
using Vector4u = Vector4<std::uint32_t>;
using Vector5u = Vector5<std::uint32_t>;
using Vector6u = Vector6<std::uint32_t>;

using Vector0i = Vector0<std::int32_t>;
using Vector1i = Vector1<std::int32_t>;
using Vector2i = Vector2<std::int32_t>;
using Vector3i = Vector3<std::int32_t>;
using Vector4i = Vector4<std::int32_t>;
using Vector5i = Vector5<std::int32_t>;
using Vector6i = Vector6<std::int32_t>;

template <typename T, int Dim>
using RowVector = Eigen::Matrix<T, 1, Dim>;
template <typename T>
using RowVectorX = RowVector<T, Dynamic>;

using RowVectorXf = RowVectorX<float>;
using RowVectorXd = RowVectorX<double>;
using RowVectorXb = RowVectorX<std::uint8_t>;
using RowVectorXs = RowVectorX<std::uint16_t>;
using RowVectorXu = RowVectorX<std::uint32_t>;
using RowVectorXi = RowVectorX<std::int32_t>;

template <typename T>
using RowVector0 = RowVector<T, 0>;
template <typename T>
using RowVector1 = RowVector<T, 1>;
template <typename T>
using RowVector2 = RowVector<T, 2>;
template <typename T>
using RowVector3 = RowVector<T, 3>;
template <typename T>
using RowVector4 = RowVector<T, 4>;
template <typename T>
using RowVector5 = RowVector<T, 5>;
template <typename T>
using RowVector6 = RowVector<T, 6>;

using RowVector0f = RowVector0<float>;
using RowVector1f = RowVector1<float>;
using RowVector2f = RowVector2<float>;
using RowVector3f = RowVector3<float>;
using RowVector4f = RowVector4<float>;
using RowVector5f = RowVector5<float>;
using RowVector6f = RowVector6<float>;

using RowVector0d = RowVector0<double>;
using RowVector1d = RowVector1<double>;
using RowVector2d = RowVector2<double>;
using RowVector3d = RowVector3<double>;
using RowVector4d = RowVector4<double>;
using RowVector5d = RowVector5<double>;
using RowVector6d = RowVector6<double>;

using RowVector0b = RowVector0<std::uint8_t>;
using RowVector1b = RowVector1<std::uint8_t>;
using RowVector2b = RowVector2<std::uint8_t>;
using RowVector3b = RowVector3<std::uint8_t>;
using RowVector4b = RowVector4<std::uint8_t>;
using RowVector5b = RowVector5<std::uint8_t>;
using RowVector6b = RowVector6<std::uint8_t>;

using RowVector0s = RowVector0<std::uint16_t>;
using RowVector1s = RowVector1<std::uint16_t>;
using RowVector2s = RowVector2<std::uint16_t>;
using RowVector3s = RowVector3<std::uint16_t>;
using RowVector4s = RowVector4<std::uint16_t>;
using RowVector5s = RowVector5<std::uint16_t>;
using RowVector6s = RowVector6<std::uint16_t>;

using RowVector0u = RowVector0<std::uint32_t>;
using RowVector1u = RowVector1<std::uint32_t>;
using RowVector2u = RowVector2<std::uint32_t>;
using RowVector3u = RowVector3<std::uint32_t>;
using RowVector4u = RowVector4<std::uint32_t>;
using RowVector5u = RowVector5<std::uint32_t>;
using RowVector6u = RowVector6<std::uint32_t>;

using RowVector0i = RowVector0<std::int32_t>;
using RowVector1i = RowVector1<std::int32_t>;
using RowVector2i = RowVector2<std::int32_t>;
using RowVector3i = RowVector3<std::int32_t>;
using RowVector4i = RowVector4<std::int32_t>;
using RowVector5i = RowVector5<std::int32_t>;
using RowVector6i = RowVector6<std::int32_t>;

template <typename T>
using AngleAxis = Eigen::AngleAxis<T>;
using AngleAxisf = AngleAxis<float>;
using AngleAxisd = AngleAxis<double>;

template <typename T>
using Quaternion = Eigen::Quaternion<T>;
using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

template <typename T>
using Translation3 = Eigen::Translation<T, 3>;
using Translation3f = Translation3<float>;
using Translation3d = Translation3<double>;

template <typename T>
using Isometry3 = Eigen::Transform<T, 3, Eigen::Isometry>;
using Isometry3f = Isometry3<float>;
using Isometry3d = Isometry3<double>;

template <typename T>
using Affine3 = Eigen::Transform<T, 3, Eigen::Affine>;
using Affine3f = Affine3<float>;
using Affine3d = Affine3<double>;

// Some aligned arrays
using TransformationList = std::vector<Affine3f>;
template <typename T>
using TransformationListT = std::vector<Affine3<T>>;

using VertexArray = Matrix3Xf;
using NormalArray = Matrix3Xf;
using TriangleArray = Matrix3Xi;
using ColorArray = Matrix3Xb;

// define a parameter set
constexpr size_t kMaxModelParams = 2048; // at most 2048 parameters per frame
using ParameterSet = std::bitset<kMaxModelParams>;

/// @brief A utility struct that facilitates the deduction of a `std::span` type from a given type.
///
/// This utility is particularly useful when a function accepts a `std::span<Vector3<T>>` as an
/// argument, where the template argument of `std::span` is also a template. In such cases, direct
/// deduction of the `std::span` type from other container types (e.g., `std::vector<Vector3<T>>`)
/// at the call site may fail.
///
/// By using this utility, the appropriate `std::span` type can be deduced automatically, ensuring
/// smoother function calls and enhancing code readability.
///
/// Example usage:
/// @code
/// void foo(DeduceSpanType<Vector3f>::type points); // Equivalent to std::span<Vector3f>
///
/// std::vector<Vector3f> points;
/// foo(points);                                     // This line will compile without any issues.
/// @endcode
///
/// @note The `std::span` generated by this utility gives a mutable view into the sequence. If an
/// immutable view is needed, consider using `std::span<const T>` instead.
template <typename T>
struct DeduceSpanType {
  using type = std::span<T>;
};

/// Casts the scalar type of objects in a container that support the `cast<>()` method.
///
/// @tparam OtherScalar The target scalar type for the cast.
/// @tparam ContainerType The type of the container.
/// @tparam ObjectType The type of the objects in the original container.
/// @param originalContainer A container of objects.
/// @return A new container with objects of the scalar type casted to `OtherScalar`. If
/// `OtherScalar` matches the original type, the original container is returned.
template <typename OtherScalar, template <typename...> class ContainerType, typename ObjectType>
[[nodiscard]] decltype(auto) cast(const ContainerType<ObjectType>& originalContainer) {
  // Check for Scalar typedef
  using Scalar = typename ObjectType::Scalar;

  if constexpr (std::is_same_v<OtherScalar, Scalar>) {
    return originalContainer;
  } else {
    using CastedType = std::remove_const_t<
        std::remove_reference_t<decltype(std::declval<ObjectType>().template cast<OtherScalar>())>>;

    ContainerType<CastedType> castedContainer;
    castedContainer.reserve(originalContainer.size()); // Optional, for performance

    std::transform(
        originalContainer.begin(),
        originalContainer.end(),
        std::back_inserter(castedContainer),
        [](const ObjectType& item) { return item.template cast<OtherScalar>(); });

    return castedContainer;
  }
}

} // namespace momentum
