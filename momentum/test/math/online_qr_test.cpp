/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/online_householder_qr.h"

#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/math/fmt_eigen.h"

#include <gtest/gtest.h>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <chrono>
#include <random>

namespace {

template <typename T>
Eigen::MatrixX<T> stack(const std::vector<Eigen::MatrixX<T>>& mats) {
  MT_THROW_IF(mats.empty(), "Empty stack.");

  Eigen::Index nRows = 0;
  for (const auto& m : mats) {
    nRows += m.rows();
    MT_THROW_IF(m.cols() != mats.front().cols(), "Mismatch in col count.");
  }

  Eigen::MatrixX<T> result(nRows, mats.front().cols());
  Eigen::Index offset = 0;
  for (const auto& m : mats) {
    result.block(offset, 0, m.rows(), m.cols()) = m;
    offset += m.rows();
  }
  return result;
}

template <typename T>
Eigen::VectorX<T> stack(const std::vector<Eigen::VectorX<T>>& vecs) {
  return stack<T>(std::vector<Eigen::MatrixX<T>>(vecs.begin(), vecs.end()));
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
makeRandomMatrix(Eigen::Index nRows, Eigen::Index nCols, std::mt19937& rng) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result(nRows, nCols);
  std::normal_distribution<float> norm;

  for (Eigen::Index iRow = 0; iRow < nRows; ++iRow) {
    for (Eigen::Index jCol = 0; jCol < nCols; ++jCol) {
      result(iRow, jCol) = norm(rng);
    }
  }

  return result;
}

} // namespace

using namespace momentum;

// Simple test on a small matrix:
TEST(OnlineQR, Basic) {
  Eigen::Vector3d b(1, 2, 3);

  Eigen::Matrix3d A;
  A << 1, 3, 4, 2, 1, 4, 5, 2, 3;

  Eigen::Vector3d x1 = A.householderQr().solve(b);

  {
    // Try adding the whole matrix at once:
    OnlineHouseholderQR<double> qr(A.cols());
    qr.add(A, b);
    const Eigen::Vector3d x2 = qr.result();
    ASSERT_LT((x1 - x2).squaredNorm(), 1e-10);
  }

  {
    // Try adding the rows in random combinations at once:
    OnlineHouseholderQR<double> qr2(A.cols());
    for (int i = 0; i < A.rows(); ++i)
      qr2.add(A.row(i), b.row(i));
    const Eigen::Vector3d x3 = qr2.result();
    ASSERT_LT((x1 - x3).squaredNorm(), 1e-10);
  }
}

// Generate a random matrix with the desired condition number.
Eigen::MatrixXd randomMatrix(
    Eigen::Index rows,
    Eigen::Index cols,
    std::mt19937& rng,
    const double conditionNumber) {
  // Generate a completely random matrix.
  Eigen::MatrixXd result(rows, cols);
  std::uniform_real_distribution<double> unif(-1, 1);
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index j = 0; j < cols; ++j) {
      result(i, j) = unif(rng);
    }
  }

  // Compute the SVD and rescale the singular values.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(result, Eigen::ComputeThinU | Eigen::ComputeThinV);

  const auto nSVD = std::min(rows, cols);

  std::vector<double> singularValues;
  for (Eigen::Index i = 0; i < nSVD; ++i) {
    singularValues.push_back(exp((double)i / (double)nSVD * log(conditionNumber)));
  }
  std::shuffle(singularValues.begin(), singularValues.end(), rng);

  Eigen::VectorXd sv = svd.singularValues();
  for (Eigen::Index i = 0; i < nSVD; ++i) {
    sv(i) = singularValues[i];
  }

  const Eigen::MatrixXd U = svd.matrixU();
  const Eigen::MatrixXd V = svd.matrixV();
  return U * Eigen::DiagonalMatrix<double, -1>(sv) * V.transpose();
}

// Test the built-in lambda inclusion:
TEST(OnlineQR, WithLambda) {
  std::mt19937 rng;

  const Eigen::Index dim = 3;
  const Eigen::Index nRows = 5;
  const double cond = 10.0;

  const Eigen::MatrixXd A = randomMatrix(nRows, dim, rng, cond);
  const Eigen::VectorXd b = randomMatrix(nRows, 1, rng, cond);

  OnlineHouseholderQR<double> qr(dim);
  for (int i = 0; i < 5; ++i) {
    const double lambda = i / 2.0;
    qr.reset(dim, lambda);

    Eigen::MatrixXd A_augmented(nRows + dim, dim);
    A_augmented.setZero();
    A_augmented.topRows(nRows) = A;
    A_augmented.bottomRows(dim).diagonal().setConstant(lambda);

    Eigen::VectorXd b_augmented(nRows + dim);
    b_augmented.topRows(nRows) = b;
    b_augmented.bottomRows(dim).setZero();

    const Eigen::VectorXd x1 = A_augmented.fullPivHouseholderQr().solve(b_augmented);

    qr.add(A, b);
    const Eigen::VectorXd x2 = qr.result();

    ASSERT_LT((x1 - x2).squaredNorm(), 1e-10);

    const Eigen::VectorXd Atb = qr.At_times_b();
    const Eigen::VectorXd Atb2 = A.transpose() * b;
    ASSERT_LT((Atb - Atb2).squaredNorm(), 1e-10);
  }
}

TEST(OnlineQR, MatrixWithZeros) {
  std::mt19937 rng;

  const Eigen::Index dim = 3;

  std::vector<Eigen::MatrixXf> A_i;
  std::vector<Eigen::MatrixXf> b_i;

  OnlineHouseholderQR<float> qr(dim);
  for (int i = 0; i < 5; ++i) {
    A_i.push_back(makeRandomMatrix<float>(i + 3, dim, rng));
    for (int j = 0; j < i && j < dim; ++j) {
      A_i.back().col(j).setZero();
    }
    b_i.push_back(makeRandomMatrix<float>(i + 3, 1, rng));
    qr.add(A_i.back(), b_i.back());
  }

  const Eigen::MatrixXf A = stack<float>(A_i);
  const Eigen::MatrixXf b = stack<float>(b_i);

  const Eigen::VectorXf x_qr = qr.result();
  const Eigen::VectorXf x_gt = A.householderQr().solve(b);
  ASSERT_LT((x_qr - x_gt).norm(), 1e-3f);
}

// Try solving the same problem but adding the rows in various combinations.
TEST(OnlineQR, AdditionCombinations) {
  std::mt19937 rng;

  for (Eigen::Index iTest = 0; iTest < 10; ++iTest) {
    const Eigen::Index dim = 3;
    const Eigen::Index nRows = (iTest + 2) * 5;
    const double cond = 10.0;

    const Eigen::MatrixXd A = randomMatrix(nRows, dim, rng, cond);
    const Eigen::VectorXd b = randomMatrix(nRows, 1, rng, cond);

    const Eigen::VectorXd x = A.fullPivHouseholderQr().solve(b);

    for (int jCombination = 0; jCombination < 10; ++jCombination) {
      OnlineHouseholderQR<double> qr(dim);
      Eigen::Index curRow = 0;
      while (curRow < nRows) {
        std::uniform_int_distribution<Eigen::Index> unif(1, 10);
        const auto nConsCur = unif(rng);
        const Eigen::Index rowEnd = std::min(curRow + nConsCur, nRows);
        qr.add(A.middleRows(curRow, rowEnd - curRow), b.middleRows(curRow, rowEnd - curRow));

        curRow = rowEnd;
      }

      const Eigen::VectorXd x2 = qr.result();
      ASSERT_LT((x2 - x).squaredNorm(), 1e-10);

      const Eigen::VectorXd Atb = qr.At_times_b();
      const Eigen::VectorXd Atb2 = A.transpose() * b;
      ASSERT_LT((Atb - Atb2).squaredNorm(), 1e-10);
    }
  }
}

// Solve with a known x value for easy validation and to compare against
// other methods (e.g. the normal equations, etc.).
TEST(OnlineQR, LargeMatrix) {
  std::mt19937 rng;

  const Eigen::Index dim = 20;
  const Eigen::Index nRows = 5000;

  const double cond = 1e8;
  Eigen::MatrixXd A_combined = randomMatrix(nRows, dim, rng, cond);
  Eigen::VectorXd x_combined = randomMatrix(dim, 1, rng, 10.0);
  Eigen::VectorXd b_combined = A_combined * x_combined;

  // Split 'em up:
  std::vector<Eigen::MatrixXd> A_mat;
  std::vector<Eigen::VectorXd> b_vec;

  const Eigen::Index minConstraints = 64;
  const Eigen::Index maxConstraints = 256;
  Eigen::Index curRow = 0;
  while (curRow < nRows) {
    std::uniform_int_distribution<Eigen::Index> unif(minConstraints, maxConstraints);
    const auto nConsCur = unif(rng);
    const Eigen::Index rowEnd = std::min(curRow + nConsCur, nRows);
    A_mat.push_back(A_combined.middleRows(curRow, rowEnd - curRow));
    b_vec.push_back(b_combined.middleRows(curRow, rowEnd - curRow));

    curRow = rowEnd;
  }

  std::vector<Eigen::MatrixXf> A_mat_f;
  std::vector<Eigen::VectorXf> b_vec_f;
  for (const auto& a : A_mat) {
    A_mat_f.push_back(a.cast<float>());
  }
  for (const auto& b : b_vec) {
    b_vec_f.push_back(b.cast<float>());
  }

  const auto nMat = A_mat.size();

  const Eigen::VectorXd Atb_gt = A_combined.transpose() * b_combined;

  // Validate the answer:
  {
    auto A_mat_copy = A_mat;
    auto b_vec_copy = b_vec;

    auto timeStart = std::chrono::high_resolution_clock::now();
    OnlineHouseholderQR<double> qr(dim);
    for (size_t i = 0; i < nMat; ++i) {
      qr.addMutating(A_mat_copy[i], b_vec_copy[i]);
    }
    MT_LOGI("-------------");
    MT_LOGI(
        "Elapsed: {}us",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count());

    const Eigen::VectorXd x = qr.result();
    const Eigen::VectorXd err = A_combined * x.cast<double>() - b_combined;
    MT_LOGI("Online QR solver l2 error (double): {:.20}", err.squaredNorm());
    MT_LOGI(
        "Online QR solver x difference (double): {:.20}",
        (x.cast<double>() - x_combined.cast<double>()).norm());

    ASSERT_LE((x - x_combined).squaredNorm(), 1e-10);

    const Eigen::VectorXd Atb = qr.At_times_b();
    ASSERT_LT((Atb - Atb_gt).squaredNorm() / Atb_gt.squaredNorm(), 1e-10);
  }

  {
    auto A_mat_copy = A_mat_f;
    auto b_vec_copy = b_vec_f;

    auto timeStart = std::chrono::high_resolution_clock::now();
    OnlineHouseholderQR<float> qr(dim);
    for (size_t i = 0; i < nMat; ++i) {
      qr.addMutating(A_mat_copy[i], b_vec_copy[i]);
    }
    MT_LOGI("-------------");
    MT_LOGI(
        "Elapsed: {}us",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count());

    const Eigen::VectorXf x = qr.result();
    const Eigen::VectorXd err = A_combined * x.cast<double>() - b_combined;
    MT_LOGI("Online QR solver l2 error (float): {:.20}", err.squaredNorm());
    MT_LOGI(
        "Online QR solver x difference (float): {:.20}",
        (x.cast<double>() - x_combined.cast<double>()).norm());

    // Need to use a bit larger eps here due to single precision.
    ASSERT_LE((x.cast<double>() - x_combined).squaredNorm(), 2.0f);

    const Eigen::VectorXf Atb = qr.At_times_b();
    ASSERT_LT((Atb.cast<double>() - Atb_gt).squaredNorm() / Atb_gt.squaredNorm(), 1e-10);
  }

  // Compare against a few other approaches.
  // Normal equations in double precision:
  {
    const auto timeStart = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd Atb = Eigen::VectorXd::Zero(dim);
    for (size_t i = 0; i < nMat; ++i) {
      const auto& a = A_mat[i];
      const auto& b = b_vec[i];
      AtA += a.transpose() * a;
      Atb += a.transpose() * b;
    }

    Eigen::LDLT<Eigen::MatrixXd> solver(AtA);
    const Eigen::VectorXd x = solver.solve(Atb);
    MT_LOGI("-------------");
    MT_LOGI(
        "Elapsed: {}us",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count());

    const Eigen::VectorXd err = A_combined * x - b_combined;
    MT_LOGI("Normal equations l2 error (double): {:.20}", err.squaredNorm());
    MT_LOGI(
        "Normal equations solver x difference (double): {:.20}",
        (x.cast<double>() - x_combined.cast<double>()).norm());
  }

  // Normal equations in single precision:
  {
    const auto timeStart = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXf AtA = Eigen::MatrixXf::Zero(dim, dim);
    Eigen::VectorXf Atb = Eigen::VectorXf::Zero(dim);
    for (size_t i = 0; i < nMat; ++i) {
      const auto& a = A_mat_f[i];
      const auto& b = b_vec_f[i];
      AtA += a.transpose() * a;
      Atb += a.transpose() * b;
    }

    Eigen::LDLT<Eigen::MatrixXf> solver(AtA);
    const Eigen::VectorXf x = solver.solve(Atb).eval();
    MT_LOGI("-------------");
    MT_LOGI(
        "Elapsed: {}us",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count());

    const Eigen::VectorXd err = A_combined * x.cast<double>() - b_combined;
    MT_LOGI("Normal equations l2 error (float): {:.20}", err.squaredNorm());
    MT_LOGI(
        "Normal equations solver x difference (float): {:.20}",
        (x.cast<double>() - x_combined.cast<double>()).norm());
  }

  // Eigen non-pivoting QR solver in single precision:
  {
    const auto timeStart = std::chrono::high_resolution_clock::now();
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A_combined.cast<float>());
    const Eigen::VectorXf x = qr.solve(b_combined.cast<float>());
    MT_LOGI("-------------");
    MT_LOGI(
        "Elapsed: {}us",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count());

    const Eigen::VectorXd err = A_combined * x.cast<double>() - b_combined;
    MT_LOGI("Eigen basic QR solver l2 error (float): {:.20}", err.squaredNorm());
    MT_LOGI(
        "Eigen basic QR solver x difference (float): {:.20}",
        (x.cast<double>() - x_combined.cast<double>()).norm());
  }

  // Eigen pivoting QR solver in single precision:
  {
    auto timeStart = std::chrono::high_resolution_clock::now();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr(A_combined.cast<float>());
    const Eigen::VectorXf x = qr.solve(b_combined.cast<float>());
    MT_LOGI("-------------");
    MT_LOGI(
        "Elapsed: {}us",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count());

    const Eigen::VectorXd err = A_combined * x.cast<double>() - b_combined;
    MT_LOGI("Eigen pivoting QR solver l2 error (float): {:.20}", err.squaredNorm());
    MT_LOGI(
        "Eigen pivoting QR solver x difference (float): {:.20}",
        (x.cast<double>() - x_combined.cast<double>()).norm());
  }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> assembleBlockMatrix(
    const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& A_diag,
    const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& A_common,
    const int n_common) {
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
  MT_CHECK(A_diag.size() == A_common.size());

  int nRows = 0;
  int nCols_diag = 0;
  for (size_t i = 0; i < A_diag.size(); ++i) {
    MT_CHECK(n_common == A_common[i].cols());
    MT_CHECK(A_common[i].rows() == A_diag[i].rows());
    nRows += A_diag[i].rows();
    nCols_diag += A_diag[i].cols();
  }

  MatrixType result = MatrixType::Zero(nRows, nCols_diag + n_common);
  int curRow = 0;
  int curCol = 0;
  for (size_t i = 0; i < A_diag.size(); ++i) {
    result.block(curRow, curCol, A_diag[i].rows(), A_diag[i].cols()) = A_diag[i];
    result.block(curRow, nCols_diag, A_diag[i].rows(), n_common) = A_common[i];

    curRow += A_diag[i].rows();
    curCol += A_diag[i].cols();
  }

  return result;
}

// Simple test on a small matrix:
TEST(OnlineBlockQR, Basic) {
  Eigen::Vector3d b_diag(1, 2, 3);

  Eigen::Matrix<double, 3, 2> A_diag;
  A_diag << 1, 3, 4, 2, 1, 4;

  Eigen::Matrix<double, 3, 1> A_common;
  A_common << 1, 4, 2;

  const Eigen::MatrixXd A_dense = assembleBlockMatrix<double>({A_diag}, {A_common}, 1);

  Eigen::Vector3d x_groundTruth = A_dense.householderQr().solve(b_diag);

  OnlineHouseholderQR<double> qr_dense(A_dense.cols());
  qr_dense.add(A_dense, b_diag);
  const Eigen::VectorXd x3 = qr_dense.result();

  // Try adding the whole matrix at once:
  OnlineBlockHouseholderQR<double> qr_blockwise(1);
  qr_blockwise.add(0, A_diag, A_common, b_diag);

  const Eigen::MatrixXd R_dense = qr_blockwise.R_dense();
  const Eigen::VectorXd y_dense = qr_blockwise.y_dense();

  ASSERT_LT((R_dense - qr_dense.R()).lpNorm<Eigen::Infinity>(), 1e-4);
  ASSERT_LT((y_dense - qr_dense.y()).lpNorm<Eigen::Infinity>(), 1e-4);

  const Eigen::VectorXd x = qr_blockwise.x_dense();
  ASSERT_LT((x - x_groundTruth).lpNorm<Eigen::Infinity>(), 1e-4);
  ASSERT_LT((x - x3).lpNorm<Eigen::Infinity>(), 1e-4);

  const Eigen::VectorXd Atb_blockwise = qr_blockwise.At_times_b_dense();
  const Eigen::VectorXd Atb_dense = qr_dense.At_times_b();
  const Eigen::VectorXd Atb_gt = A_dense.transpose() * b_diag;
  ASSERT_LT((Atb_blockwise - Atb_gt).squaredNorm(), 1e-10);
  ASSERT_LT((Atb_dense - Atb_gt).squaredNorm(), 1e-10);
}

TEST(OnlineBlockQR, TwoBlocks) {
  Eigen::VectorXd b_diag(6);
  b_diag << 1, 2, 3, 4, 5, 6;

  Eigen::Matrix<double, 3, 2> A1_diag;
  A1_diag << 1, 3, 4, 2, 1, 4;

  Eigen::Matrix<double, 3, 1> A1_common;
  A1_common << 1, 4, 2;

  Eigen::Matrix<double, 3, 2> A2_diag;
  A2_diag << 2, 4, 7, -1, -4, 4;

  Eigen::Matrix<double, 3, 1> A2_common;
  A2_common << -3, 5, -2;

  const Eigen::MatrixXd A_dense =
      assembleBlockMatrix<double>({A1_diag, A2_diag}, {A1_common, A2_common}, 1);

  Eigen::VectorXd x_groundTruth = A_dense.householderQr().solve(b_diag);

  OnlineHouseholderQR<double> qr_dense(A_dense.cols());
  qr_dense.add(A_dense, b_diag);
  const Eigen::VectorXd x3 = qr_dense.result();

  // Try adding the whole matrix at once:
  OnlineBlockHouseholderQR<double> qr_blockwise(1);
  qr_blockwise.add(0, A1_diag, A1_common, b_diag.head<3>());
  qr_blockwise.add(1, A2_diag, A2_common, b_diag.tail<3>());

  const Eigen::MatrixXd R_dense = qr_blockwise.R_dense();
  const Eigen::VectorXd y_dense = qr_blockwise.y_dense();

  ASSERT_LT((R_dense - qr_dense.R()).lpNorm<Eigen::Infinity>(), 1e-4);
  ASSERT_LT((y_dense - qr_dense.y()).lpNorm<Eigen::Infinity>(), 1e-4);

  const Eigen::VectorXd x = qr_blockwise.x_dense();
  ASSERT_LT((x - x_groundTruth).lpNorm<Eigen::Infinity>(), 1e-4);
  ASSERT_LT((x - x3).lpNorm<Eigen::Infinity>(), 1e-4);

  const Eigen::VectorXd Atb_blockwise = qr_blockwise.At_times_b_dense();
  const Eigen::VectorXd Atb_dense = qr_dense.At_times_b();
  const Eigen::VectorXd Atb_gt = A_dense.transpose() * b_diag;
  ASSERT_LT((Atb_blockwise - Atb_gt).squaredNorm(), 1e-10);
  ASSERT_LT((Atb_dense - Atb_gt).squaredNorm(), 1e-10);
}

template <typename T>
void validateBlockwiseSolver(
    const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& A_ii,
    const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& A_in,
    const std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>>& b_i,
    const Eigen::Index nCols_common,
    std::mt19937& rng) {
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorType;

  const MatrixType A_dense = assembleBlockMatrix<T>(A_ii, A_in, nCols_common);
  const VectorType b_dense = stack<T>(b_i);

  OnlineHouseholderQR<T> qr_dense(A_dense.cols());
  qr_dense.add(A_dense, b_dense);
  const VectorType x_denseQR = qr_dense.result();

  OnlineBlockHouseholderQR<T> qr_blockwise(nCols_common);

  // Make sure clearing works:
  for (size_t iIter = 0; iIter < 3; ++iIter) {
    for (size_t jBlock = 0; jBlock < A_ii.size(); ++jBlock) {
      qr_blockwise.add(
          jBlock,
          makeRandomMatrix<T>(A_ii[jBlock].rows(), A_ii[jBlock].cols(), rng),
          makeRandomMatrix<T>(A_in[jBlock].rows(), A_in[jBlock].cols(), rng),
          b_i[jBlock]);
    }
  }
  qr_blockwise.reset();

  for (size_t i = 0; i < A_ii.size(); ++i) {
    qr_blockwise.add(i, A_ii[i], A_in[i], b_i[i]);
  }

  // Make sure the R and y matrices are the same, since they're using the
  // same algorithm:
  const MatrixType R_dense = qr_blockwise.R_dense();
  const VectorType y_dense = qr_blockwise.y_dense();
  ASSERT_LT((R_dense - qr_dense.R()).template lpNorm<Eigen::Infinity>(), 1e-4);
  ASSERT_LT((y_dense - qr_dense.y()).template lpNorm<Eigen::Infinity>(), 1e-4);

  // Now validate the results (x).
  const VectorType x_blockwise = qr_blockwise.x_dense();
  const VectorType x_groundTruth = A_dense.householderQr().solve(b_dense);

  // These results should be identical since they're supposed to be using the same algorithm:
  // (note that this only holds if the matrix is full rank).
  if (A_dense.rows() > A_dense.cols()) {
    ASSERT_LT((x_denseQR - x_blockwise).template lpNorm<Eigen::Infinity>(), 0.1f);
  }

  const VectorType err_blockwise = A_dense * x_blockwise - b_dense;
  const VectorType err_groundTruth = A_dense * x_groundTruth - b_dense;
  // Two solvers can come up with different answers; the key here is that the
  // errors need to be similar:
  const T sqrerr_groundTruth = err_groundTruth.squaredNorm();
  const T sqrerr_blockwise = err_blockwise.squaredNorm();
  ASSERT_LT(sqrerr_blockwise, sqrerr_groundTruth + 1e-4);

  // Make sure per-block result accessor produces the same result:
  size_t offset = 0;
  for (size_t iBlock = 0; iBlock < A_ii.size(); ++iBlock) {
    const VectorType x_i = x_blockwise.segment(offset, A_ii[iBlock].cols());
    const VectorType x_i_2 = qr_blockwise.x_i(iBlock);
    ASSERT_LT((x_i - x_i_2).template lpNorm<Eigen::Infinity>(), 1e-4);
    offset += A_ii[iBlock].cols();
  }

  {
    const VectorType x_i = x_blockwise.segment(offset, nCols_common);
    const VectorType x_i_2 = qr_blockwise.x_n();
    ASSERT_LT((x_i - x_i_2).template lpNorm<Eigen::Infinity>(), 1e-4);
  }

  // Now add the rows in a random order and make sure we get the same result:
  std::vector<std::pair<size_t, Eigen::Index>> rowsToAdd;
  for (size_t iBlock = 0; iBlock < A_ii.size(); ++iBlock) {
    for (Eigen::Index iRow = 0; iRow < A_ii[iBlock].rows(); ++iRow) {
      rowsToAdd.emplace_back(iBlock, iRow);
    }
  }

  std::shuffle(rowsToAdd.begin(), rowsToAdd.end(), rng);
  OnlineBlockHouseholderQR<T> qr_blockwise_random(nCols_common);

  for (const auto& row : rowsToAdd) {
    const auto iBlock = row.first;
    const auto jRow = row.second;
    qr_blockwise_random.add(
        iBlock, A_ii[iBlock].row(jRow), A_in[iBlock].row(jRow), b_i[iBlock].row(jRow));
  }

  const MatrixType R_dense_random = qr_blockwise_random.R_dense();
  const VectorType y_dense_random = qr_blockwise_random.y_dense();
  ASSERT_LT((R_dense_random - qr_dense.R()).template lpNorm<Eigen::Infinity>(), 1e-4);
  ASSERT_LT((y_dense_random - qr_dense.y()).template lpNorm<Eigen::Infinity>(), 1e-4);

  const VectorType x_blockwise_random = qr_blockwise_random.x_dense();
  const VectorType err_blockwise_random = A_dense * x_blockwise_random - b_dense;
  const T sqrerr_blockwise_random = err_blockwise.squaredNorm();
  ASSERT_LT(sqrerr_blockwise_random, sqrerr_groundTruth + 1e-4);

  const VectorType Atb_gt = A_dense.transpose() * b_dense;
  ASSERT_LT((qr_dense.At_times_b() - Atb_gt).squaredNorm(), 1e-10);
  ASSERT_LT((qr_blockwise.At_times_b_dense() - Atb_gt).squaredNorm(), 1e-10);
  ASSERT_LT((qr_blockwise_random.At_times_b_dense() - Atb_gt).squaredNorm(), 2e-10);

  const double Atb_dot_x_gt = Atb_gt.dot(x_denseQR);
  ASSERT_NEAR(qr_blockwise.At_times_b_dot(x_denseQR), Atb_dot_x_gt, 5e-3);
  ASSERT_NEAR(qr_blockwise_random.At_times_b_dot(x_denseQR), Atb_dot_x_gt, 5e-3);
}

TEST(OnlineBlockQR, RandomMatrix) {
  const size_t nTests = 10;
  const size_t nBlocks = 6;

  std::mt19937 rng;
  for (size_t iTest = 0; iTest < nTests; ++iTest) {
    std::vector<Eigen::MatrixXf> A_ii;
    std::vector<Eigen::MatrixXf> A_in;
    std::vector<Eigen::VectorXf> b_i;

    std::uniform_int_distribution<size_t> nColsDist(1, 5);
    std::uniform_int_distribution<size_t> nRowsDist(0, 3);

    const auto nCols_common = nColsDist(rng);
    for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock) {
      const auto nCols = nColsDist(rng);
      const auto nRows = nCols + nRowsDist(rng);

      A_ii.push_back(makeRandomMatrix<float>(nRows, nCols, rng));
      A_in.push_back(makeRandomMatrix<float>(nRows, nCols_common, rng));
      b_i.push_back(makeRandomMatrix<float>(nRows, 1, rng));

      validateBlockwiseSolver<float>(A_ii, A_in, b_i, nCols_common, rng);
    }
  }
}

TEST(OnlineBandedQR, RandomMatrix) {
  std::mt19937 rng;

  for (size_t n_cols_banded = 0; n_cols_banded < 10; ++n_cols_banded) {
    for (size_t n_cols_shared = 0; n_cols_shared < 10; ++n_cols_shared) {
      for (size_t bandwidth = 1; bandwidth < 4; ++bandwidth) {
        const size_t n_cols_total = n_cols_banded + n_cols_shared;

        OnlineHouseholderQR<float> denseSolver(n_cols_banded + n_cols_shared);
        OnlineBandedHouseholderQR<float> bandedSolver(n_cols_banded, n_cols_shared, bandwidth);

        Eigen::VectorXf Atb_dense = Eigen::VectorXf::Zero(n_cols_banded + n_cols_shared);

        std::uniform_int_distribution<size_t> nRowsDist(0, 3);

        for (size_t iBandedCol = 0; iBandedCol < n_cols_banded; ++iBandedCol) {
          const auto nBandedCols = std::min(bandwidth, n_cols_banded - iBandedCol);
          const auto nRows = nBandedCols + nRowsDist(rng);

          Eigen::MatrixXf A_banded = makeRandomMatrix<float>(nRows, nBandedCols, rng);
          Eigen::MatrixXf A_shared = makeRandomMatrix<float>(nRows, n_cols_shared, rng);
          Eigen::VectorXf b = makeRandomMatrix<float>(nRows, 1, rng);

          Eigen::MatrixXf A_dense = Eigen::MatrixXf::Zero(nRows, n_cols_total);
          A_dense.block(0, iBandedCol, nRows, nBandedCols) = A_banded;
          A_dense.block(0, n_cols_banded, nRows, n_cols_shared) = A_shared;

          denseSolver.add(A_dense, b);
          bandedSolver.add(iBandedCol, A_banded, A_shared, b);

          Atb_dense += A_dense.transpose() * b;
        }

        {
          // Add some constraints for the shared column to make sure we're not singular:
          const auto nRows = n_cols_shared + nRowsDist(rng);
          Eigen::MatrixXf A_shared = makeRandomMatrix<float>(nRows, n_cols_shared, rng);
          Eigen::VectorXf b = makeRandomMatrix<float>(nRows, 1, rng);

          Eigen::MatrixXf A_dense = Eigen::MatrixXf::Zero(nRows, n_cols_total);
          A_dense.block(0, n_cols_banded, nRows, n_cols_shared) = A_shared;

          denseSolver.add(A_dense, b);
          bandedSolver.add(A_shared, b);

          Atb_dense += A_dense.transpose() * b;
        }

        MT_LOGD("R (dense solver):\n{}", denseSolver.R());
        MT_LOGD("R (banded solver):\n{}", bandedSolver.R_dense());

        MT_LOGD("y (dense solver):\n{}", denseSolver.y());
        MT_LOGD("y (banded solver):\n{}", bandedSolver.y_dense());

        MT_LOGD("x (dense solver):\n{}", denseSolver.result());
        MT_LOGD("x (banded solver):\n{}", bandedSolver.x_dense());

        ASSERT_LT((denseSolver.R() - bandedSolver.R_dense()).lpNorm<Eigen::Infinity>(), 1e-3f);
        ASSERT_LT((denseSolver.y() - bandedSolver.y_dense()).lpNorm<Eigen::Infinity>(), 1e-3f);
        ASSERT_LT((denseSolver.result() - bandedSolver.x_dense()).lpNorm<Eigen::Infinity>(), 1e-3f);
        ASSERT_LT((Atb_dense - bandedSolver.At_times_b()).lpNorm<Eigen::Infinity>(), 1e-3f);
      }
    }
  }
}

TEST(ResizableMatrix, Basic) {
  std::mt19937 rng;

  ResizeableMatrix<float> m;

  ASSERT_EQ(0, m.rows());
  ASSERT_EQ(0, m.cols());

  {
    const Eigen::MatrixXf test1 = makeRandomMatrix<float>(5, 2, rng);
    m.resizeAndSetZero(test1.rows(), test1.cols());
    ASSERT_TRUE(m.mat().isZero());
    m.mat() = test1;
    ASSERT_EQ(test1, m.mat());
  }

  {
    const Eigen::MatrixXf test2 = makeRandomMatrix<float>(4, 10, rng);
    m.resizeAndSetZero(test2.rows(), test2.cols());
    ASSERT_TRUE(m.mat().isZero());
    m.mat() = test2;
    ASSERT_EQ(test2, m.mat());
  }
}
