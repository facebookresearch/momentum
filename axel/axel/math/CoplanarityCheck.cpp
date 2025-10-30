/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/math/CoplanarityCheck.h"

namespace axel {
int solveP3(std::span<double, 3> x, double a, const double b, const double c) {
  static const double halfSqrt3 = 0.5 * std::sqrt(3.0);
  constexpr double divBy3 = 1.0 / 3.0;
  constexpr double divBy9 = 1.0 / 9.0;
  constexpr double divBy54 = 1.0 / 54.0;
  constexpr double pi = 3.1415926535897932384626433832795028841971693993751058209749445;

  // solve cubic equation x^3 + a*x^2 + b*x + c
  const double a2 = a * a;
  double q = (a2 - 3.0 * b) * divBy9;
  const double r = (a * (2.0 * a2 - 9.0 * b) + 27.0 * c) * divBy54;
  const double r2 = r * r;
  const double q3 = q * q * q;
  if (r2 < q3) {
    double t = r / std::sqrt(q3);
    if (t < -1.0) {
      t = -1.0;
    }
    if (t > 1.0) {
      t = 1.0;
    }
    t = std::acos(t);
    a /= 3.0;
    q = -2.0 * std::sqrt(q);
    x[0] = q * std::cos(t * divBy3) - a;
    x[1] = q * std::cos((t + (2.0 * pi)) * divBy3) - a;
    x[2] = q * std::cos((t - (2.0 * pi)) * divBy3) - a;
    return 3;
  }
  double A = -std::pow(std::fabs(r) + std::sqrt(r2 - q3), divBy3);
  if (r < 0.0) {
    A = -A;
  }
  const double B = A == 0.0 ? 0.0 : q / A;
  a /= 3.0;
  x[0] = (A + B) - a;
  x[1] = -0.5 * (A + B) - a;
  x[2] = halfSqrt3 * (A - B);
  if (std::fabs(x[2]) < 1e-14) {
    x[2] = x[1];
    return 3;
  }
  return 1;
}

int solveP2(std::span<double, 2> x, const double a, const double b, const double c) {
  const double discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    // We're not interested in imaginary roots
    return 0;
  }

  // At this point a is nonzero so we can safely divide by 2.0*a
  if (discriminant < 1e-9) {
    x[0] = (-b) / (2.0 * a);
    return 1;
  }
  const double sqrtDiscriminant = std::sqrt(discriminant);
  x[0] = (-b + sqrtDiscriminant) / (2.0 * a);
  x[1] = (-b - sqrtDiscriminant) / (2.0 * a);
  return 2;
}

// NOTE: Do not remove the manual AVX code unless benchmarked and no performance degradation is
// shown.
int timesCoplanar(
    std::span<double, 3> t,
    const Eigen::Vector3d& x1,
    const Eigen::Vector3d& x2,
    const Eigen::Vector3d& x3,
    const Eigen::Vector3d& x4,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3,
    const Eigen::Vector3d& v4) {
  // Points are coplanar at the roots of the following cubic scalar equation in t
  // (x21 + t * v21)x(x31 + t * v31).(x41 + t * v41) = 0

  const Eigen::Vector3d x21 = x2 - x1;
  const Eigen::Vector3d x31 = x3 - x1;
  const Eigen::Vector3d x41 = x4 - x1;
  const Eigen::Vector3d v21 = v2 - v1;
  const Eigen::Vector3d v31 = v3 - v1;
  const Eigen::Vector3d v41 = v4 - v1;

#ifdef __AVX__
  // vectorized version of the code
  // verified to produce the same results as previous implementation
  const __m256d r1 = _mm256_setr_pd(-v21.z(), v21.y(), v21.z(), -v21.x());
  const __m256d r2 = _mm256_setr_pd(v31.y(), v31.z(), v31.x(), v31.z());
  const __m256d r3 = _mm256_setr_pd(v41.x(), v41.x(), v41.y(), v41.y());
  const __m256d r1b = _mm256_setr_pd(-v21.y(), v21.x(), 0.0, 0.0);
  const __m256d r2b = _mm256_setr_pd(v31.x(), v31.y(), 0.0, 0.0);
  const __m256d r3b = _mm256_setr_pd(v41.z(), v41.z(), 0.0, 0.0);
  const __m256d r1Xr2 = _mm256_mul_pd(r1, r2);
  const __m256d r1Xr2Xr3 = _mm256_mul_pd(r1Xr2, r3);
  const __m256d r1bXr2b = _mm256_mul_pd(r1b, r2b);
  const __m256d r1bXr2bXr3b = _mm256_mul_pd(r1bXr2b, r3b);

  const double* mult1 = (double*)&r1Xr2Xr3; // NOLINT
  const double* mult2 = (double*)&r1bXr2bXr3b; // NOLINT
  const double d1 =
      mult1[0] + mult1[1] + mult1[2] + mult1[3]; // is there intrinsic for horizontal add? NOLINT
  const double d2 = mult2[0] + mult2[1]; // NOLINT
  const double d = d1 + d2;

  // ---------------------------
  // Coefficient t^2
  const __m256d ar1a = _mm256_setr_pd(-v31.z(), v31.y(), v31.z(), -v31.x());
  const __m256d ar2a = _mm256_setr_pd(v41.y(), v41.z(), v41.x(), v41.z());
  const __m256d ar3a = _mm256_setr_pd(x21.x(), x21.x(), x21.y(), x21.y());
  const __m256d ar1Xr2 = _mm256_mul_pd(ar1a, ar2a);
  const __m256d ar1Xr2Xr3 = _mm256_mul_pd(ar1Xr2, ar3a);
  const double* amult1 = (double*)&ar1Xr2Xr3; // NOLINT
  const double a1 = amult1[0] + amult1[1] + amult1[2] + amult1[3]; // NOLINT

  const __m256d ar1b = _mm256_setr_pd(-v31.y(), v31.x(), v21.z(), -v21.y());
  const __m256d ar2b = _mm256_setr_pd(v41.x(), v41.y(), v41.y(), v41.z());
  const __m256d ar3b = _mm256_setr_pd(x21.z(), x21.z(), x31.x(), x31.x());
  const __m256d ar1Xr2b = _mm256_mul_pd(ar1b, ar2b);
  const __m256d ar1Xr2Xr3b = _mm256_mul_pd(ar1Xr2b, ar3b);
  const double* amult2 = (double*)&ar1Xr2Xr3b; // NOLINT
  const double a2 = amult2[0] + amult2[1] + amult2[2] + amult2[3]; // NOLINT

  const __m256d ar1c = _mm256_setr_pd(-v21.z(), v21.x(), v21.y(), -v21.x());
  const __m256d ar2c = _mm256_setr_pd(v41.x(), v41.z(), v41.x(), v41.y());
  const __m256d ar3c = _mm256_setr_pd(x31.y(), x31.y(), x31.z(), x31.z());
  const __m256d ar1Xr2c = _mm256_mul_pd(ar1c, ar2c);
  const __m256d ar1Xr2Xr3c = _mm256_mul_pd(ar1Xr2c, ar3c);
  const double* amult3 = (double*)&ar1Xr2Xr3c; // NOLINT
  const double a3 = amult3[0] + amult3[1] + amult3[2] + amult3[3]; // NOLINT

  const __m256d ar1d = _mm256_setr_pd(-v21.z(), v21.y(), v21.z(), -v21.x());
  const __m256d ar2d = _mm256_setr_pd(v31.y(), v31.z(), v31.x(), v31.z());
  const __m256d ar3d = _mm256_setr_pd(x41.x(), x41.x(), x41.y(), x41.y());
  const __m256d ar1Xr2d = _mm256_mul_pd(ar1d, ar2d);
  const __m256d ar1Xr2Xr3d = _mm256_mul_pd(ar1Xr2d, ar3d);
  const double* amult4 = (double*)&ar1Xr2Xr3d; // NOLINT
  const double a4 = amult4[0] + amult4[1] + amult4[2] + amult4[3]; // NOLINT

  const __m256d ar1e = _mm256_setr_pd(-v21.y(), v21.x(), 0.0, 0.0);
  const __m256d ar2e = _mm256_setr_pd(v31.x(), v31.y(), 0.0, 0.0);
  const __m256d ar3e = _mm256_setr_pd(x41.z(), x41.z(), 0.0, 0.0);
  const __m256d ar1Xr2e = _mm256_mul_pd(ar1e, ar2e);
  const __m256d ar1Xr2Xr3e = _mm256_mul_pd(ar1Xr2e, ar3e);
  const double* amult5 = (double*)&ar1Xr2Xr3e; // NOLINT
  const double a5 = amult5[0] + amult5[1]; // NOLINT

  const double a = a1 + a2 + a3 + a4 + a5;

  // Coefficient t^1
  const __m256d br1a = _mm256_setr_pd(-v41.z(), v41.y(), v41.z(), -v41.x());
  const __m256d br2a = _mm256_setr_pd(x21.y(), x21.z(), x21.x(), x21.z());
  const __m256d br3a = _mm256_setr_pd(x31.x(), x31.x(), x31.y(), x31.y());
  const __m256d br1Xr2a = _mm256_mul_pd(br1a, br2a);
  const __m256d br1Xr2Xr3a = _mm256_mul_pd(br1Xr2a, br3a);
  const double* bmult1 = (double*)&br1Xr2Xr3a; // NOLINT
  const double b1 = bmult1[0] + bmult1[1] + bmult1[2] + bmult1[3]; // NOLINT

  const __m256d br1b = _mm256_setr_pd(-v41.y(), v41.x(), v31.z(), -v31.y());
  const __m256d br2b = _mm256_setr_pd(x21.x(), x21.y(), x21.y(), x21.z());
  const __m256d br3b = _mm256_setr_pd(x31.z(), x31.z(), x41.x(), x41.x());
  const __m256d br1Xr2b = _mm256_mul_pd(br1b, br2b);
  const __m256d br1Xr2Xr3b = _mm256_mul_pd(br1Xr2b, br3b);
  const double* bmult2 = (double*)&br1Xr2Xr3b; // NOLINT
  const double b2 = bmult2[0] + bmult2[1] + bmult2[2] + bmult2[3]; // NOLINT

  const __m256d br1c = _mm256_setr_pd(-v21.z(), v21.y(), -v31.z(), v31.x());
  const __m256d br2c = _mm256_setr_pd(x31.y(), x31.z(), x21.x(), x21.z());
  const __m256d br3c = _mm256_setr_pd(x41.x(), x41.x(), x41.y(), x41.y());
  const __m256d br1Xr2c = _mm256_mul_pd(br1c, br2c);
  const __m256d br1Xr2Xr3c = _mm256_mul_pd(br1Xr2c, br3c);
  const double* bmult3 = (double*)&br1Xr2Xr3c; // NOLINT
  const double b3 = bmult3[0] + bmult3[1] + bmult3[2] + bmult3[3]; // NOLINT

  const __m256d br1d = _mm256_setr_pd(v21.z(), -v21.x(), v31.y(), -v31.x());
  const __m256d br2d = _mm256_setr_pd(x31.x(), x31.z(), x21.x(), x21.y());
  const __m256d br3d = _mm256_setr_pd(x41.y(), x41.y(), x41.z(), x41.z());
  const __m256d br1Xr2d = _mm256_mul_pd(br1d, br2d);
  const __m256d br1Xr2Xr3d = _mm256_mul_pd(br1Xr2d, br3d);
  const double* bmult4 = (double*)&br1Xr2Xr3d; // NOLINT
  const double b4 = bmult4[0] + bmult4[1] + bmult4[2] + bmult4[3]; // NOLINT

  const __m256d br1e = _mm256_setr_pd(-v21.y(), v21.x(), 0.0, 0.0);
  const __m256d br2e = _mm256_setr_pd(x31.x(), x31.y(), 0.0, 0.0);
  const __m256d br3e = _mm256_setr_pd(x41.z(), x41.z(), 0.0, 0.0);
  const __m256d br1Xr2e = _mm256_mul_pd(br1e, br2e);
  const __m256d br1Xr2Xr3e = _mm256_mul_pd(br1Xr2e, br3e);
  const double* bmult5 = (double*)&br1Xr2Xr3e; // NOLINT
  const double b5 = bmult5[0] + bmult5[1]; // NOLINT

  const double b = b1 + b2 + b3 + b4 + b5;

  // Coefficient t^0
  const __m256d cr1a = _mm256_setr_pd(-x21.z(), x21.y(), x21.z(), -x21.x());
  const __m256d cr2a = _mm256_setr_pd(x31.y(), x31.z(), x31.x(), x31.z());
  const __m256d cr3a = _mm256_setr_pd(x41.x(), x41.x(), x41.y(), x41.y());
  const __m256d cr1Xr2a = _mm256_mul_pd(cr1a, cr2a);
  const __m256d cr1Xr2Xr3a = _mm256_mul_pd(cr1Xr2a, cr3a);
  const double* cmult1 = (double*)&cr1Xr2Xr3a; // NOLINT
  const double c1 = cmult1[0] + cmult1[1] + cmult1[2] + cmult1[3]; // NOLINT

  const __m256d cr1b = _mm256_setr_pd(-x21.y(), x21.x(), 0.0, 0.0);
  const __m256d cr2b = _mm256_setr_pd(x31.x(), x31.y(), 0.0, 0.0);
  const __m256d cr3b = _mm256_setr_pd(x41.z(), x41.z(), 0.0, 0.0);
  const __m256d cr1Xr2b = _mm256_mul_pd(cr1b, cr2b);
  const __m256d cr1Xr2Xr3b = _mm256_mul_pd(cr1Xr2b, cr3b);
  const double* cmult2 = (double*)&cr1Xr2Xr3b; // NOLINT
  const double c2 = cmult2[0] + cmult2[1]; // NOLINT
  const double c = c1 + c2;
#else
  // Coefficient t^3
  const float d =
      (-v21.z() * v31.y() * v41.x() + v21.y() * v31.z() * v41.x() + v21.z() * v31.x() * v41.y() -
       v21.x() * v31.z() * v41.y() - v21.y() * v31.x() * v41.z() + v21.x() * v31.y() * v41.z());
  // Coefficient t^2
  const float a =
      (-v31.z() * v41.y() * x21.x() + v31.y() * v41.z() * x21.x() + v31.z() * v41.x() * x21.y() -
       v31.x() * v41.z() * x21.y() - v31.y() * v41.x() * x21.z() + v31.x() * v41.y() * x21.z() +
       v21.z() * v41.y() * x31.x() - v21.y() * v41.z() * x31.x() - v21.z() * v41.x() * x31.y() +
       v21.x() * v41.z() * x31.y() + v21.y() * v41.x() * x31.z() - v21.x() * v41.y() * x31.z() -
       v21.z() * v31.y() * x41.x() + v21.y() * v31.z() * x41.x() + v21.z() * v31.x() * x41.y() -
       v21.x() * v31.z() * x41.y() - v21.y() * v31.x() * x41.z() + v21.x() * v31.y() * x41.z());
  // Coefficient t^1
  const float b =
      (-v41.z() * x21.y() * x31.x() + v41.y() * x21.z() * x31.x() + v41.z() * x21.x() * x31.y() -
       v41.x() * x21.z() * x31.y() - v41.y() * x21.x() * x31.z() + v41.x() * x21.y() * x31.z() +
       v31.z() * x21.y() * x41.x() - v31.y() * x21.z() * x41.x() - v21.z() * x31.y() * x41.x() +
       v21.y() * x31.z() * x41.x() - v31.z() * x21.x() * x41.y() + v31.x() * x21.z() * x41.y() +
       v21.z() * x31.x() * x41.y() - v21.x() * x31.z() * x41.y() + v31.y() * x21.x() * x41.z() -
       v31.x() * x21.y() * x41.z() - v21.y() * x31.x() * x41.z() + v21.x() * x31.y() * x41.z());
  // Coefficient t^0
  const float c = -x21.z() * x31.y() * x41.x() + x21.y() * x31.z() * x41.x() +
      x21.z() * x31.x() * x41.y() - x21.x() * x31.z() * x41.y() - x21.y() * x31.x() * x41.z() +
      x21.x() * x31.y() * x41.z();
#endif

  int solutions = 0;

  if (std::abs(d) < 1e-9) {
    // solution is not cubic equation
    if (std::abs(a) < 1e-9) {
      // solution is not quadratic either
      if (std::abs(b) > 1e-9) {
        // Solve linear equation for t
        solutions = 1;
        t[0] = -c / b;
      } else {
        // Equation is indepenend of t so probably coplanar for the entire
        // duration of the time step.
        // (coplanar and not moving relative to each other)
        // Let's just say there's a potential collision at time 0
        // We'll check more closely in the narrow phase later.
        solutions = 1;
        t[0] = 0.0;
      }
    } else {
      // Solve Quadratic Equation for t
      solutions = solveP2(t.first<2>(), a, b, c); // solve cubic equation + a*x^2 + b*x + c = 0
    }
  } else {
    // Solve cubic Equation for t
    solutions =
        solveP3(t, a / d, b / d, c / d); // solve cubic equation d/dx^3 + a/d*x^2 + b/d*x + c/d = 0
  }

  return solutions;
}
} // namespace axel
