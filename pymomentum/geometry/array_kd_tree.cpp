/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_kd_tree.h"

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>

#include <axel/SimdKdTree.h>
#include <axel/TriBvh.h>
#include <dispenso/parallel_for.h>
#include <momentum/common/exception.h>

#include <cfloat>
#include <cstdint>

namespace pymomentum {

namespace {

template <typename T, int Dim>
void findClosestPointsImpl(
    const VectorArrayAccessor<T, Dim>& srcAcc,
    const VectorArrayAccessor<T, Dim>& tgtAcc,
    const std::vector<py::ssize_t>& srcIndices,
    const std::vector<py::ssize_t>& tgtIndices,
    float maxSqrDist,
    typename VectorArrayAccessor<T, Dim>::ElementView resultPoints,
    typename VectorArrayAccessor<int32_t, 1>::ElementView resultIndices) {
  using Vec = typename axel::SimdKdTreef<Dim>::Vec;

  // Get target points as matrix for KD-tree construction
  const auto tgtMatrix = tgtAcc.toMatrix(tgtIndices);
  const Vec* pts_tgt_ptr = reinterpret_cast<const Vec*>(tgtMatrix.data());
  const int64_t nTgtPts = tgtMatrix.rows();

  // Build KD-tree from target points
  const axel::SimdKdTreef<Dim> kdTree_target({pts_tgt_ptr, pts_tgt_ptr + nTgtPts});

  // Get source points view
  auto srcView = srcAcc.view(srcIndices);
  const int64_t nSrcPts = srcView.size();

  // Query each source point
  for (int64_t k = 0; k < nSrcPts; ++k) {
    const Vec p_src = srcView.get(k).template cast<float>();
    const auto [valid, tgt_index, sqrDist] = kdTree_target.closestPoint(p_src, maxSqrDist);

    if (valid) {
      resultIndices.set(k, Eigen::Vector<int32_t, 1>{static_cast<int32_t>(tgt_index)});
      resultPoints.set(k, tgtMatrix.row(tgt_index).template cast<T>());
    } else {
      // No valid point found so return -1
      resultIndices.set(k, Eigen::Vector<int32_t, 1>{-1});
      resultPoints.set(k, Eigen::Vector<T, Dim>::Zero());
    }
  }
}

template <typename T>
void findClosestPointsWithNormalsImpl(
    const VectorArrayAccessor<T, 3>& srcPointsAcc,
    const VectorArrayAccessor<T, 3>& srcNormalsAcc,
    const VectorArrayAccessor<T, 3>& tgtPointsAcc,
    const VectorArrayAccessor<T, 3>& tgtNormalsAcc,
    const std::vector<py::ssize_t>& srcIndices,
    const std::vector<py::ssize_t>& tgtIndices,
    float maxSqrDist,
    float maxNormalDot,
    typename VectorArrayAccessor<T, 3>::ElementView resultPoints,
    typename VectorArrayAccessor<T, 3>::ElementView resultNormals,
    typename VectorArrayAccessor<int32_t, 1>::ElementView resultIndices) {
  // Get target data as matrices for KD-tree construction
  const auto tgtPointsMatrix = tgtPointsAcc.toMatrix(tgtIndices);
  const auto tgtNormalsMatrix = tgtNormalsAcc.toMatrix(tgtIndices);

  const Eigen::Vector3f* pts_tgt_ptr =
      reinterpret_cast<const Eigen::Vector3f*>(tgtPointsMatrix.data());
  const Eigen::Vector3f* normals_tgt_ptr =
      reinterpret_cast<const Eigen::Vector3f*>(tgtNormalsMatrix.data());
  const int64_t nTgtPts = tgtPointsMatrix.rows();

  // Build KD-tree from target points and normals
  const axel::SimdKdTreef<3> kdTree_target(
      {pts_tgt_ptr, pts_tgt_ptr + nTgtPts}, {normals_tgt_ptr, normals_tgt_ptr + nTgtPts});

  // Get source points views
  auto srcPointsView = srcPointsAcc.view(srcIndices);
  auto srcNormalsView = srcNormalsAcc.view(srcIndices);
  const int64_t nSrcPts = srcPointsView.size();

  // Query each source point
  for (int64_t k = 0; k < nSrcPts; ++k) {
    const Eigen::Vector3<T> p_src = srcPointsView.get(k);
    const Eigen::Vector3<T> normal_src = srcNormalsView.get(k);

    const auto [valid, tgt_index, sqrDist] = kdTree_target.closestPoint(
        p_src.template cast<float>(), normal_src.template cast<float>(), maxSqrDist, maxNormalDot);

    if (valid) {
      resultIndices.set(k, Eigen::Vector<int32_t, 1>{static_cast<int32_t>(tgt_index)});
      resultPoints.set(k, tgtPointsMatrix.row(tgt_index).template cast<T>());
      resultNormals.set(k, tgtNormalsMatrix.row(tgt_index).template cast<T>());
    } else {
      // No valid point found so return -1
      resultIndices.set(k, Eigen::Vector<int32_t, 1>{-1});
      resultPoints.set(k, Eigen::Vector3<T>::Zero());
      resultNormals.set(k, Eigen::Vector3<T>::Zero());
    }
  }
}

template <typename T>
void findClosestPointsOnMeshImpl(
    const VectorArrayAccessor<T, 3>& srcPointsAcc,
    const VectorArrayAccessor<T, 3>& tgtVerticesAcc,
    const IntVectorArrayAccessor<3>& tgtFacesAcc,
    const std::vector<py::ssize_t>& srcIndices,
    const std::vector<py::ssize_t>& tgtIndices,
    typename VectorArrayAccessor<T, 3>::ElementView resultPoints,
    typename VectorArrayAccessor<int32_t, 1>::ElementView resultFaceIndices,
    typename VectorArrayAccessor<T, 3>::ElementView resultBarycentric) {
  using TriBvh = typename axel::TriBvh<T, axel::kNativeLaneWidth<T>>;

  // Get target vertices as matrix (returns row-major, but TriBvh constructor handles conversion)
  Eigen::MatrixX3<T> targetVerticesMat = tgtVerticesAcc.toMatrix(tgtIndices);
  const int64_t nTgtVertices = targetVerticesMat.rows();

  MT_THROW_IF(
      targetVerticesMat.cols() != 3,
      "find_closest_points_on_mesh: vertices_target must have 3 columns");

  // Get faces as matrix using bulk conversion (single switch on dtype)
  auto facesView = tgtFacesAcc.view(tgtIndices);
  Eigen::MatrixX3i targetFacesMat = facesView.toMatrix();

  // Check for valid face indices
  const auto [minIdx, maxIdx] = facesView.minmax();
  MT_THROW_IF(
      minIdx < 0, "find_closest_points_on_mesh: faces_target contains negative index {}", minIdx);
  MT_THROW_IF(
      maxIdx >= nTgtVertices,
      "find_closest_points_on_mesh: faces_target contains an index {} >= nTgtVertices {}",
      maxIdx,
      nTgtVertices);

  // Build BVH from mesh
  const TriBvh targetTree(std::move(targetVerticesMat), std::move(targetFacesMat));

  // Get source points view
  auto srcView = srcPointsAcc.view(srcIndices);
  const int64_t nSrcPts = srcView.size();

  // Query each source point
  for (int64_t k = 0; k < nSrcPts; ++k) {
    const Eigen::Vector3<T> p_src = srcView.get(k);
    const auto queryResult = targetTree.closestSurfacePoint(p_src);

    if (queryResult.triangleIdx == axel::kInvalidTriangleIdx) {
      resultFaceIndices.set(k, Eigen::Vector<int32_t, 1>{-1});
      resultPoints.set(k, Eigen::Vector3<T>::Zero());
      resultBarycentric.set(k, Eigen::Vector3<T>::Zero());
    } else {
      resultFaceIndices.set(
          k, Eigen::Vector<int32_t, 1>{static_cast<int32_t>(queryResult.triangleIdx)});
      resultPoints.set(k, queryResult.point);
      if (queryResult.baryCoords) {
        resultBarycentric.set(k, *queryResult.baryCoords);
      } else {
        resultBarycentric.set(k, Eigen::Vector3<T>::Zero());
      }
    }
  }
}

bool isNormalized(const py::array& arr) {
  if (arr.size() == 0) {
    return true;
  }

  auto bufInfo = arr.request();
  if (bufInfo.shape[bufInfo.ndim - 1] != 3) {
    return false;
  }

  // Check first few elements only (like the tensor version)
  const auto totalElements = arr.size() / 3;
  const auto elementsToCheck = std::min<py::ssize_t>(10, totalElements);

  if (bufInfo.format == py::format_descriptor<float>::format()) {
    const float* data = static_cast<const float*>(bufInfo.ptr);
    for (py::ssize_t i = 0; i < elementsToCheck; ++i) {
      float norm = 0.0f;
      for (int d = 0; d < 3; ++d) {
        float val = data[i * 3 + d];
        norm += val * val;
      }
      norm = std::sqrt(norm);
      if (norm < 0.95f || norm > 1.05f) {
        return false;
      }
    }
  } else if (bufInfo.format == py::format_descriptor<double>::format()) {
    const double* data = static_cast<const double*>(bufInfo.ptr);
    for (py::ssize_t i = 0; i < elementsToCheck; ++i) {
      double norm = 0.0;
      for (int d = 0; d < 3; ++d) {
        double val = data[i * 3 + d];
        norm += val * val;
      }
      norm = std::sqrt(norm);
      if (norm < 0.95 || norm > 1.05) {
        return false;
      }
    }
  }

  return true;
}

} // namespace

std::tuple<py::array, py::array, py::array>
findClosestPointsArray(py::buffer points_source, py::buffer points_target, float maxDist) {
  ArrayChecker checker("find_closest_points");

  const float maxSqrDist = (maxDist == FLT_MAX) ? FLT_MAX : maxDist * maxDist;

  // Get buffer info first to determine dimensionality before validation
  auto srcInfo = py::buffer(points_source).request();

  MT_THROW_IF(
      srcInfo.ndim < 2,
      "find_closest_points: points_source must have at least 2 dimensions, got {}",
      srcInfo.ndim);

  const auto dim = srcInfo.shape[srcInfo.ndim - 1];

  MT_THROW_IF(dim != 2 && dim != 3, "find_closest_points: dimension must be 2 or 3, got {}", dim);

  const auto nSrcPts = srcInfo.shape[srcInfo.ndim - 2];

  // Validate points_source: shape [..., nSrcPoints, dim]
  checker.validateBuffer(
      points_source,
      "points_source",
      {-1, static_cast<int>(dim)},
      {"nSrcPoints", "dim"},
      false /* allowEmpty */);

  // Get leading dimensions from validated points_source
  const auto leadingDims = checker.getLeadingDimensions();

  // Validate points_target: shape [..., nTgtPoints, dim]
  checker.validateBuffer(
      points_target,
      "points_target",
      {-2, static_cast<int>(dim)},
      {"nTgtPoints", "dim"},
      false /* allowEmpty */);

  auto tgtInfo = py::buffer(points_target).request();
  const auto nTgtPts = tgtInfo.shape[tgtInfo.ndim - 2];

  // Convert to arrays after validation
  py::array srcArr(points_source);
  py::array tgtArr(points_target);

  const auto nBatch = leadingDims.totalBatchElements();

  // Create output arrays
  py::array resultIndices, resultPoints, resultValid;

  if (checker.isFloat64()) {
    resultIndices = createOutputArray<int32_t>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(1)});
    resultPoints = createOutputArray<double>(leadingDims, {nSrcPts, dim});

    if (dim == 2) {
      VectorArrayAccessor<double, 2> srcAcc(srcArr, leadingDims, nSrcPts);
      VectorArrayAccessor<double, 2> tgtAcc(tgtArr, leadingDims, nTgtPts);
      VectorArrayAccessor<int32_t, 1> indicesAcc(resultIndices, leadingDims, nSrcPts);
      VectorArrayAccessor<double, 2> pointsAcc(resultPoints, leadingDims, nSrcPts);

      BatchIndexer indexer(leadingDims);

      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsImpl<double, 2>(
            srcAcc,
            tgtAcc,
            indices,
            indices,
            maxSqrDist,
            pointsAcc.view(indices),
            indicesAcc.view(indices));
      });
    } else { // dim == 3
      VectorArrayAccessor<double, 3> srcAcc(srcArr, leadingDims, nSrcPts);
      VectorArrayAccessor<double, 3> tgtAcc(tgtArr, leadingDims, nTgtPts);
      VectorArrayAccessor<int32_t, 1> indicesAcc(resultIndices, leadingDims, nSrcPts);
      VectorArrayAccessor<double, 3> pointsAcc(resultPoints, leadingDims, nSrcPts);

      BatchIndexer indexer(leadingDims);

      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsImpl<double, 3>(
            srcAcc,
            tgtAcc,
            indices,
            indices,
            maxSqrDist,
            pointsAcc.view(indices),
            indicesAcc.view(indices));
      });
    }
  } else {
    resultIndices = createOutputArray<int32_t>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(1)});
    resultPoints = createOutputArray<float>(leadingDims, {nSrcPts, dim});

    if (dim == 2) {
      VectorArrayAccessor<float, 2> srcAcc(srcArr, leadingDims, nSrcPts);
      VectorArrayAccessor<float, 2> tgtAcc(tgtArr, leadingDims, nTgtPts);
      VectorArrayAccessor<int32_t, 1> indicesAcc(resultIndices, leadingDims, nSrcPts);
      VectorArrayAccessor<float, 2> pointsAcc(resultPoints, leadingDims, nSrcPts);

      BatchIndexer indexer(leadingDims);

      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsImpl<float, 2>(
            srcAcc,
            tgtAcc,
            indices,
            indices,
            maxSqrDist,
            pointsAcc.view(indices),
            indicesAcc.view(indices));
      });
    } else { // dim == 3
      VectorArrayAccessor<float, 3> srcAcc(srcArr, leadingDims, nSrcPts);
      VectorArrayAccessor<float, 3> tgtAcc(tgtArr, leadingDims, nTgtPts);
      VectorArrayAccessor<int32_t, 1> indicesAcc(resultIndices, leadingDims, nSrcPts);
      VectorArrayAccessor<float, 3> pointsAcc(resultPoints, leadingDims, nSrcPts);

      BatchIndexer indexer(leadingDims);

      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsImpl<float, 3>(
            srcAcc,
            tgtAcc,
            indices,
            indices,
            maxSqrDist,
            pointsAcc.view(indices),
            indicesAcc.view(indices));
      });
    }
  }

  // Create validity mask from indices (valid when >= 0)
  resultValid = createOutputArray<bool>(leadingDims, {nSrcPts});
  auto validInfo = resultValid.request();
  auto indicesInfo = resultIndices.request();
  const int32_t* indicesData = static_cast<const int32_t*>(indicesInfo.ptr);
  bool* validData = static_cast<bool*>(validInfo.ptr);

  for (py::ssize_t i = 0; i < resultIndices.size(); ++i) {
    validData[i] = indicesData[i] >= 0;
  }

  // Squeeze the last dimension from indices to match expected shape [..., nSrcPts]
  resultIndices = resultIndices.attr("squeeze")(-1);

  return {resultPoints, resultIndices, resultValid};
}

std::tuple<py::array, py::array, py::array, py::array> findClosestPointsWithNormalsArray(
    py::buffer points_source,
    py::buffer normals_source,
    py::buffer points_target,
    py::buffer normals_target,
    float maxDist,
    float maxNormalDot) {
  ArrayChecker checker("find_closest_points");

  const float maxSqrDist = (maxDist == FLT_MAX) ? FLT_MAX : maxDist * maxDist;

  // Get buffer info first to determine sizes before validation
  auto srcInfo = py::buffer(points_source).request();
  const auto nSrcPts = srcInfo.shape[srcInfo.ndim - 2];

  auto tgtInfo = py::buffer(points_target).request();
  const auto nTgtPts = tgtInfo.shape[tgtInfo.ndim - 2];

  // Validate points_source: shape [..., nSrcPoints, 3]
  checker.validateBuffer(
      points_source, "points_source", {-1, 3}, {"nSrcPoints", "xyz"}, false /* allowEmpty */);

  const auto leadingDims = checker.getLeadingDimensions();

  // Validate normals_source: shape [..., nSrcPoints, 3]
  checker.validateBuffer(
      normals_source, "normals_source", {-1, 3}, {"nSrcPoints", "xyz"}, false /* allowEmpty */);

  // Validate points_target: shape [..., nTgtPoints, 3]
  checker.validateBuffer(
      points_target, "points_target", {-2, 3}, {"nTgtPoints", "xyz"}, false /* allowEmpty */);

  // Validate normals_target: shape [..., nTgtPoints, 3]
  checker.validateBuffer(
      normals_target, "normals_target", {-2, 3}, {"nTgtPoints", "xyz"}, false /* allowEmpty */);

  // Convert to arrays after validation
  py::array srcPointsArr(points_source);
  py::array srcNormalsArr(normals_source);
  py::array tgtPointsArr(points_target);
  py::array tgtNormalsArr(normals_target);

  // Check normalization
  if (!isNormalized(srcNormalsArr)) {
    py::print(
        "Inside find_closest_points, the tensor of source normals does not appear to be normalized.  This likely indicates a bug.");
  }

  if (!isNormalized(tgtNormalsArr)) {
    py::print(
        "Inside find_closest_points, the tensor of target normals does not appear to be normalized.  This likely indicates a bug.");
  }

  const auto nBatch = leadingDims.totalBatchElements();

  // Create output arrays
  py::array resultIndices, resultPoints, resultNormals, resultValid;

  if (checker.isFloat64()) {
    resultIndices = createOutputArray<int32_t>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(1)});
    resultPoints = createOutputArray<double>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});
    resultNormals = createOutputArray<double>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});

    VectorArrayAccessor<double, 3> srcPointsAcc(srcPointsArr, leadingDims, nSrcPts);
    VectorArrayAccessor<double, 3> srcNormalsAcc(srcNormalsArr, leadingDims, nSrcPts);
    VectorArrayAccessor<double, 3> tgtPointsAcc(tgtPointsArr, leadingDims, nTgtPts);
    VectorArrayAccessor<double, 3> tgtNormalsAcc(tgtNormalsArr, leadingDims, nTgtPts);
    VectorArrayAccessor<int32_t, 1> indicesAcc(resultIndices, leadingDims, nSrcPts);
    VectorArrayAccessor<double, 3> pointsAcc(resultPoints, leadingDims, nSrcPts);
    VectorArrayAccessor<double, 3> normalsAcc(resultNormals, leadingDims, nSrcPts);

    BatchIndexer indexer(leadingDims);

    {
      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsWithNormalsImpl<double>(
            srcPointsAcc,
            srcNormalsAcc,
            tgtPointsAcc,
            tgtNormalsAcc,
            indices,
            indices,
            maxSqrDist,
            maxNormalDot,
            pointsAcc.view(indices),
            normalsAcc.view(indices),
            indicesAcc.view(indices));
      });
    }
  } else {
    resultIndices = createOutputArray<int32_t>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(1)});
    resultPoints = createOutputArray<float>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});
    resultNormals = createOutputArray<float>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});

    VectorArrayAccessor<float, 3> srcPointsAcc(srcPointsArr, leadingDims, nSrcPts);
    VectorArrayAccessor<float, 3> srcNormalsAcc(srcNormalsArr, leadingDims, nSrcPts);
    VectorArrayAccessor<float, 3> tgtPointsAcc(tgtPointsArr, leadingDims, nTgtPts);
    VectorArrayAccessor<float, 3> tgtNormalsAcc(tgtNormalsArr, leadingDims, nTgtPts);
    VectorArrayAccessor<int32_t, 1> indicesAcc(resultIndices, leadingDims, nSrcPts);
    VectorArrayAccessor<float, 3> pointsAcc(resultPoints, leadingDims, nSrcPts);
    VectorArrayAccessor<float, 3> normalsAcc(resultNormals, leadingDims, nSrcPts);

    BatchIndexer indexer(leadingDims);

    {
      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsWithNormalsImpl<float>(
            srcPointsAcc,
            srcNormalsAcc,
            tgtPointsAcc,
            tgtNormalsAcc,
            indices,
            indices,
            maxSqrDist,
            maxNormalDot,
            pointsAcc.view(indices),
            normalsAcc.view(indices),
            indicesAcc.view(indices));
      });
    }
  }

  // Create validity mask from indices (valid when >= 0)
  resultValid = createOutputArray<bool>(leadingDims, {nSrcPts});
  auto validInfo = resultValid.request();
  auto indicesInfo = resultIndices.request();
  const int32_t* indicesData = static_cast<const int32_t*>(indicesInfo.ptr);
  bool* validData = static_cast<bool*>(validInfo.ptr);

  for (py::ssize_t i = 0; i < resultIndices.size(); ++i) {
    validData[i] = indicesData[i] >= 0;
  }

  // Squeeze the last dimension from indices to match expected shape [..., nSrcPts]
  resultIndices = resultIndices.attr("squeeze")(-1);

  return {resultPoints, resultNormals, resultIndices, resultValid};
}

std::tuple<py::array, py::array, py::array, py::array> findClosestPointsOnMeshArray(
    py::buffer points_source,
    py::buffer vertices_target,
    py::buffer faces_target) {
  ArrayChecker checker("find_closest_points_on_mesh");

  // Get buffer info first to determine sizes before validation
  auto srcInfo = py::buffer(points_source).request();
  const auto nSrcPts = srcInfo.shape[srcInfo.ndim - 2];

  auto verticesInfo = py::buffer(vertices_target).request();
  const auto nTgtVertices = verticesInfo.shape[verticesInfo.ndim - 2];

  auto facesInfo = py::buffer(faces_target).request();
  const auto nTgtFaces = facesInfo.shape[facesInfo.ndim - 2];

  // Validate points_source: shape [..., nSrcPoints, 3]
  checker.validateBuffer(
      points_source, "points_source", {-1, 3}, {"nSrcPoints", "xyz"}, false /* allowEmpty */);

  const auto leadingDims = checker.getLeadingDimensions();

  // Validate vertices_target: shape [..., nTgtVertices, 3]
  checker.validateBuffer(
      vertices_target, "vertices_target", {-2, 3}, {"nTgtVertices", "xyz"}, false /* allowEmpty */);

  // Note: faces_target is validated by IntVectorArrayAccessor which accepts int32/int64

  // Convert to arrays after validation
  py::array srcArr(points_source);
  py::array verticesArr(vertices_target);
  py::array facesArr(faces_target);

  // Create IntVectorArrayAccessor for faces - handles int32/int64 conversion automatically
  IntVectorArrayAccessor<3> facesAcc(facesArr, leadingDims, nTgtFaces);

  const auto nBatch = leadingDims.totalBatchElements();

  // Create output arrays
  py::array resultFaceIndices, resultPoints, resultBarycentric, resultValid;

  if (checker.isFloat64()) {
    resultFaceIndices =
        createOutputArray<int32_t>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(1)});
    resultPoints = createOutputArray<double>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});
    resultBarycentric =
        createOutputArray<double>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});

    VectorArrayAccessor<double, 3> srcAcc(srcArr, leadingDims, nSrcPts);
    VectorArrayAccessor<double, 3> verticesAcc(verticesArr, leadingDims, nTgtVertices);
    VectorArrayAccessor<int32_t, 1> faceIndicesAcc(resultFaceIndices, leadingDims, nSrcPts);
    VectorArrayAccessor<double, 3> pointsAcc(resultPoints, leadingDims, nSrcPts);
    VectorArrayAccessor<double, 3> baryAcc(resultBarycentric, leadingDims, nSrcPts);

    BatchIndexer indexer(leadingDims);

    {
      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsOnMeshImpl<double>(
            srcAcc,
            verticesAcc,
            facesAcc,
            indices,
            indices,
            pointsAcc.view(indices),
            faceIndicesAcc.view(indices),
            baryAcc.view(indices));
      });
    }
  } else {
    resultFaceIndices =
        createOutputArray<int32_t>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(1)});
    resultPoints = createOutputArray<float>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});
    resultBarycentric =
        createOutputArray<float>(leadingDims, {nSrcPts, static_cast<py::ssize_t>(3)});

    VectorArrayAccessor<float, 3> srcAcc(srcArr, leadingDims, nSrcPts);
    VectorArrayAccessor<float, 3> verticesAcc(verticesArr, leadingDims, nTgtVertices);
    VectorArrayAccessor<int32_t, 1> faceIndicesAcc(resultFaceIndices, leadingDims, nSrcPts);
    VectorArrayAccessor<float, 3> pointsAcc(resultPoints, leadingDims, nSrcPts);
    VectorArrayAccessor<float, 3> baryAcc(resultBarycentric, leadingDims, nSrcPts);

    BatchIndexer indexer(leadingDims);

    {
      py::gil_scoped_release release;
      dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
        auto indices = indexer.decompose(iBatch);
        findClosestPointsOnMeshImpl<float>(
            srcAcc,
            verticesAcc,
            facesAcc,
            indices,
            indices,
            pointsAcc.view(indices),
            faceIndicesAcc.view(indices),
            baryAcc.view(indices));
      });
    }
  }

  // Create validity mask from face indices (valid when >= 0)
  resultValid = createOutputArray<bool>(leadingDims, {nSrcPts});
  auto validInfo = resultValid.request();
  auto indicesInfo = resultFaceIndices.request();
  const int32_t* indicesData = static_cast<const int32_t*>(indicesInfo.ptr);
  bool* validData = static_cast<bool*>(validInfo.ptr);

  for (py::ssize_t i = 0; i < resultFaceIndices.size(); ++i) {
    validData[i] = indicesData[i] >= 0;
  }

  // Squeeze the last dimension from face indices to match expected shape [..., nSrcPts]
  resultFaceIndices = resultFaceIndices.attr("squeeze")(-1);

  return {resultValid, resultPoints, resultFaceIndices, resultBarycentric};
}

} // namespace pymomentum
