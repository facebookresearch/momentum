/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/solver/momentum_ik.h"
#include "pymomentum/tensor_ik/solver_options.h"
#include "pymomentum/tensor_ik/tensor_ik.h"

#include <momentum/math/mesh.h>

#include <dispenso/parallel_for.h> // @manual
#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace py = pybind11;
namespace mm = momentum;

using namespace pymomentum;

PYBIND11_MODULE(solver, m) {
  // TODO more explanation
  m.attr("__name__") = "pymomentum.solver";
  m.doc() = "Inverse kinematics and other optimizations for momentum models.";

#ifdef PYMOMENTUM_LIMITED_TORCH_API
  m.attr("AUTOGRAD_ENABLED") = false;
#else
  m.attr("AUTOGRAD_ENABLED") = true;
#endif

  pybind11::module_::import("torch"); // @dep=//caffe2:torch
  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbsource//arvr/libraries/pymomentum:geometry
  auto solver2 = pybind11::module_::import(
      "pymomentum.solver2"); // @dep=fbsource//arvr/libraries/pymomentum:solver2

  m.attr("VertexConstraintType") = solver2.attr("VertexConstraintType").attr("Position");

  py::enum_<ErrorFunctionType>(
      m, "ErrorFunctionType", R"(Types of error functions passed to solveBodyIKProblem().)")
      .value("Position", ErrorFunctionType::Position)
      .value("Orientation", ErrorFunctionType::Orientation)
      .value("Limit", ErrorFunctionType::Limit)
      .value("Collision", ErrorFunctionType::Collision)
      .value("PosePrior", ErrorFunctionType::PosePrior)
      .value("Motion", ErrorFunctionType::Motion)
      .value("Projection", ErrorFunctionType::Projection)
      .value("Distance", ErrorFunctionType::Distance)
      .value("Vertex", ErrorFunctionType::Vertex)
      .value("VertexProjection", ErrorFunctionType::VertexProjection)
      .export_values();

  py::enum_<LinearSolverType>(
      m, "LinearSolverType", R"(Types of linear solver supported by solveBodyIKProblem().)")
      .value("Cholesky", LinearSolverType::Cholesky)
      .value("QR", LinearSolverType::QR)
      .value("TrustRegionQR", LinearSolverType::TrustRegionQR);

  py::class_<SolverOptions>(
      m,
      "SolverOptions",
      R"(Options to control the behavior of the Levenberg-Marquardt IK solver.  For advanced use cases only; most users should stick with the defaults.)")
      .def(
          py::init<>([](LinearSolverType linearSolverType,
                        float levmar_lambda,
                        size_t minIter,
                        size_t maxIter,
                        float threshold,
                        bool lineSearch,
                        float sequenceSmoothingWeight,
                        bool verbose) {
            return SolverOptions{
                linearSolverType,
                levmar_lambda,
                minIter,
                maxIter,
                threshold,
                lineSearch,
                sequenceSmoothingWeight,
                verbose};
          }),
          py::arg("linear_solver") = LinearSolverType::QR,
          py::arg("levmar_lambda") = 0.01f,
          py::arg("min_iter") = 4,
          py::arg("max_iter") = 50,
          py::arg("threshold") = 10.0f,
          py::arg("line_search") = true,
          py::arg("sequence_smoothing_weight") = 0.0f,
          py::arg("verbose") = false)
      .def_readwrite(
          "linear_solver",
          &SolverOptions::linearSolverType,
          "Specify the type of linear solver to use (default: QR)")
      .def_readwrite(
          "levmar_lambda",
          &SolverOptions::levmar_lambda,
          "Lambda value used to regularize the Levenberg-Marquardt solver (default: 0.01f).")
      .def_readwrite(
          "min_iter",
          &SolverOptions::minIter,
          "Minimum number of nonlinear iterations (default 4).")
      .def_readwrite(
          "max_iter",
          &SolverOptions::maxIter,
          "Maximum number of nonlinear iterations (default: 50).")
      .def_readwrite(
          "threshold",
          &SolverOptions::threshold,
          "The solver stops taking steps when the difference between iterations is less than threshold*FLT_EPS (default: 10.0f).")
      .def_readwrite(
          "line_search",
          &SolverOptions::lineSearch,
          "Whether or not to use a line search; note that only the QR solver actually uses this (default: true).")
      .def_readwrite(
          "verbose", &SolverOptions::verbose, "Flag to print out solver progress (default: false).")
      .def("__repr__", [](const SolverOptions& options) {
        return fmt::format(
            "SolverOptions(linear_solver=LinearSolverType.{}, levmar_lambda={}, min_iter={}, max_iter={}, threshold={}, line_search={}, verbose={})",
            toString(options.linearSolverType),
            options.levmar_lambda,
            options.minIter,
            options.maxIter,
            options.threshold,
            (options.lineSearch ? "True" : "False"),
            (options.verbose ? "True" : "False"));
      });

  m.def(
      "transform_pose",
      &transformPose,
      R"(
Computes a new set of model parameters such that the character pose is a rigidly transformed version of the
original pose.  While there is technically a closed form solution for any given skeleton, this is complicated
in momentum because different characters attach the rigid parameters to different joints, so a fully general
solution uses IK.  This function optimizes the IK to run on a minimal set of joints and be as fast as possible,
while dealing with local minima problems due to Euler angles.

:return: A new set of model parameters such that the skeleton's root has been transformed.
:param character: Character to use in the solve.
:param model_parameters: Model parameters for the posed character in its initial space.
:param transform: The rigid transform to apply to the character, specified as a skeleton state(tx, ty, tz, rx, ry, rz, rw, s).  Note that the scale is ignored.
:param ensure_continuous_output: If true, the solver will try to ensure that the output is continuous in time.  This helps to remove Euler flips in continuous data.
  )",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("transform"),
      py::arg("ensure_continuous_output") = true);

  // -------------------------------------------------------------
  //                    solveBodyIKProblem
  // -------------------------------------------------------------
  m.def(
      "solve_ik",
      &solveBodyIKProblem,
      R"(
Batch solve a body IK problem.  The IK problem is posed as a least squares problem and minimized with respect to the model parameters :math:`\theta` using Levenberg-Marquardt.  The result is differentiable with respect to most inputs.

The IK energy function that is minimized is a sum of squared residual terms, :math:`E(\theta) = \sum_i || r_i(\theta) ||_2^2`.

We strongly recommend you use named parameters when calling this function as the parameter list is likely to grow over time.

Most tensor arguments can be specified either batched (e.g. [nBatch x nConstraints]) or unbatched (e.g. [nConstraints]).  "Unbatched" tensors will be
automatically expanded to match the batch dimension.  So, for example, you could pass a tensor of size [nConstraints] for the positionConstraints_parents parameter
and a tensor of dimension [nBatch x nConstraints x 3] for the position_constraints_targets tensor, and the solver will understand that you mean to use
the same parents across the batch.

All quaternion parameters use the quaternion order [x, y, z, w], where :math:`q = w + xi + yj + zk`.

:return: The model parameters that (locally) minimize the error function, as a (nBatch x nModelParams)-dimension torch.Tensor.
:param character: Character or list of nBatch Characters to use in the solve.
:param active_parameters: boolean-valued torch.Tensor with dimension (k), which selects the parameters to be active during the solve.  For example, you might choose to exclude scaling parameters with 'activeParameters=~character.getScalingParameters()'.  Limiting the parameters can also be helpful when dealing with underdetermined problems.
:param model_parameters_init: float-valued torch.Tensor which contain the (nBatch x nModelParameters)-dimensional model parameters used to initialize the solve.  This could be important as the solver always converges to a local minimum.
:param active_error_functions: list of pymomentum.ErrorFunctionType which gives the order of input error function types.
:param error_function_weights: float-valued torch.Tensor with dimension (nBatch x len(activeErrorFunctions)) which contains a global weight for each active error function. The order of the weights is specified by activeErrorFunctions.
:param options.linearSolverType: Linear solver used in the Levenberg-Marquardt solve (default: QR).
:param options.levmar_lambda: Lambda value used to regularize the Levenberg-Marquardt solver (default: 0.01).
:param options.min_iter: Minimum number of iterations used in the nonlinear Levenberg-Marquardt solve (default: 4).
:param options.max_iter: Maximum number of iterations used in the nonlinear Levenberg-Marquardt solve (default: 50).
:param options.threshold: The solver stops taking steps when the difference between iterations is less than threshold*FLT_EPS.
:param options.line_search: whether to use line search for the QR solver (default: true).
:param position_cons_parents: integer-valued torch.Tensor of dimension (nBatch x nConstraints); contains a single parent for each constraint.
:param position_cons_offsets: float-valued torch.Tensor of dimension (nBatch x nConstraints x 3); contains the local offset for each constraint in its parent frame.
:param position_cons_weights: float-valued torch.Tensor of dimension (nBatch x nConstraints); contains a per-constraint weight.
:param position_cons_targets: float-valued torch.Tensor of dimension (nBatch x nConstraints x 3); contains the world-space target for each position constraint.
:param orientation_cons_parents: integer-valued tensor of dimension (nBatch x nConstraints); contains a single parents for each of the orientation constraints.
:param orientation_cons_offsets: Optional float-valued tensor of dimension (nBatch x nConstraints x 4); contains the quaternion orientation offset for each orientation constraint in its parent frame.
:param orientation_cons_weights: float-valued torch.Tensor of dimension (nBatch x nConstraints) which contains a per-constraint weight.
:param orientation_cons_targets: float-valued torch.Tensor of dimension (nBatch x nConstraints x 4) (m x n x 4): Per-constraint world-space orientation targets as quaternions.
:param pose_prior_model: Mixture-PCA model used by the pose prior.
    It can either be a :py:class:`pymomentum.geometry.Mpcca`, or it can be a tuple of tensors [pi, mu, W, sigma, param_indices] as described in
    :math:`pymomentum.geometry.Mppca.to_tensors`.  The former is a bit more efficient during the solve (because there is no need to re-calculate the inverse covariance matrix) but
    the latter case allows for differentiability.
:param motion_targets: float-valued torch.Tensor of dimension (nBatch x nModelParams) which contains the model parameter targets for the motionError.
:param motion_weights: float-valued torch.Tensor of dimension (nBatch x nModelParams) which contain a per-model-parameter weight for the motionError.
:param projection_cons_projections: float-valued torch.Tensor of dimension (nBatch x nConstraints x 4 x 3) containing a 4x3 projection matrix for each constraint.  Note
    that while you can use a standard pinhole model matrix as the projection matrix, we actually recommend constructing a separate local projection matrix
    for each constraint centered around the camera ray, which is more robust for e.g. fisheye cameras.
:param projection_cons_parents: integer-valued torch.Tensor of dimension (nBatch x nConstraints); contains a single parent for each projection constraint.
:param projection_cons_offsets: float-valued torch.Tensor of dimension (nBatch x nConstraints x 3); contains the local offset for each projection constraint in its parent frame.
:param projection_cons_weights: float-valued torch.Tensor of dimension (nBatch x nConstraints); contains a per-projection constraint weight.
:param projection_cons_targets: float-valued torch.Tensor of dimension (nBatch x nConstraints x 2); contains the world-space target for each projection constraint.
:param distance_cons_origins: float-valued torch.Tensor of dimension (nBatch x nConstraints x 3); contains the world-space origins to measure distance from.
:param distance_cons_parents: integer-valued torch.Tensor of dimension (nBatch x nConstraints); contains a single parent for each distance constraint.
:param distance_cons_offsets: float-valued torch.Tensor of dimension (nBatch x nConstraints x 3); contains the local offset for each distance constraint in its parent frame.
:param distance_cons_weights: float-valued torch.Tensor of dimension (nBatch x nConstraints); contains a per-projection constraint weight.
:param distance_cons_targets: float-valued torch.Tensor of dimension (nBatch x nConstraints); contains the distance target for each constraint.
:param vertex_cons_vertices: int-valued torch.Tensor of dimension (nBatch x nConstraints); contains the vertex index for each constraint.
:param vertex_cons_weights: float-valued torch.Tensor of dimension (nBatch x nConstraints); contains the per-constraint weight for each constraint.
:param vertex_cons_target_positions: float-valued torch.Tensor of dimension (nBatch x nConstraints x 3); contains the target position.
:param vertex_cons_target_normals: float-valued torch.Tensor of dimension (nBatch x nConstraints x 3); contains the target normal.  Not used if the vertex constraint type is POSITION.
:param vertex_cons_type: Type of vertex constraint.  POSITION is a position constraint, while PLANE, NORMAL, and SYMMETRIC_NORMAL are variants on point-to-plane (PLANE uses the target normal, NORMAL uses the source normal, and SYMMETRIC_NORMAL uses a blend of the two normals).
:param vertex_proj_cons_vertices: int-valued torch.Tensor of dimension (nBatch x nConstraints); contains the vertex index for each constraint.
:param vertex_proj_cons_weights: float-valued torch.Tensor of dimension (nBatch x nConstraints); contains the per-constraint weight for each constraint.
:param vertex_proj_cons_target_positions: float-valued torch.Tensor of dimension (nBatch x nConstraints x 2); contains the target position.
:param vertex_proj_cons_target_projections: float-valued torch.Tensor of dimension (nBatch x nConstraints x 4 x 3); containing a 4x3 projection matrix for each constraint.  Note
    that while you can use a standard pinhole model matrix as the projection matrix, we actually recommend constructing a separate local projection matrix
    for each constraint centered around the camera ray, which is more robust for e.g. fisheye cameras.
)",
      py::arg("character"),
      py::arg("active_parameters"),
      py::arg("model_parameters_init"),
      py::arg("active_error_functions"),
      py::arg("error_function_weights"),
      py::arg("options") = SolverOptions(),
      py::arg("position_cons_parents") = std::optional<at::Tensor>{},
      py::arg("position_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("position_cons_weights") = std::optional<at::Tensor>{},
      py::arg("position_cons_targets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_parents") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_weights") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_targets") = std::optional<at::Tensor>{},
      py::arg("pose_prior_model") = std::optional<at::Tensor>{},
      py::arg("motion_targets") = std::optional<at::Tensor>{},
      py::arg("motion_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_projections") = std::optional<at::Tensor>{},
      py::arg("projection_cons_parents") = std::optional<at::Tensor>{},
      py::arg("projection_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("projection_cons_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_targets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_origins") = std::optional<at::Tensor>{},
      py::arg("distance_cons_parents") = std::optional<at::Tensor>{},
      py::arg("distance_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_weights") = std::optional<at::Tensor>{},
      py::arg("distance_cons_targets") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_normals") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_type") = momentum::VertexConstraintType::Position,
      py::arg("vertex_proj_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_projections") = std::optional<at::Tensor>{});

  m.def(
      "gradient",
      &computeGradient,
      R"(
Compute the gradient of the IK energy.
.. math::

     \frac{dE(\theta)}{d\theta} = 2 \sum_i \frac{dr_i(\theta)}{d\theta} r_i(\theta) = \mathbf{J}^T \mathbf{r}

We can compute gradients of the :py:func:`gradient` function with respect to all of the inputs.

For details on the arguments, see :py:func:`solve_ik`.

:return: A (nBatch x m) torch.Tensor containing the residual.
)",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("active_error_functions"),
      py::arg("error_function_weights"),
      py::arg("position_cons_parents") = std::optional<at::Tensor>{},
      py::arg("position_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("position_cons_weights") = std::optional<at::Tensor>{},
      py::arg("position_cons_targets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_parents") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_weights") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_targets") = std::optional<at::Tensor>{},
      py::arg("pose_prior_model") = std::optional<at::Tensor>{},
      py::arg("motion_targets") = std::optional<at::Tensor>{},
      py::arg("motion_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_projections") = std::optional<at::Tensor>{},
      py::arg("projection_cons_parents") = std::optional<at::Tensor>{},
      py::arg("projection_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("projection_cons_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_targets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_origins") = std::optional<at::Tensor>{},
      py::arg("distance_cons_parents") = std::optional<at::Tensor>{},
      py::arg("distance_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_weights") = std::optional<at::Tensor>{},
      py::arg("distance_cons_targets") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_normals") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_type") = momentum::VertexConstraintType::Position,
      py::arg("vertex_proj_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_projections") = std::optional<at::Tensor>{});

  m.def(
      "residual",
      &computeResidual,
      R"(
Compute the residual of an IK problem.  The IK energy minimized by :py:func:`solve_ik` is the squared norm of the residual.

Note that we can only compute gradients of the residual with respect to the input model parameters.

For details on the arguments, see :py:func:`solve_ik`.

:return: A (nBatch x m) torch.Tensor containing the residual.
)",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("active_error_functions"),
      py::arg("error_function_weights"),
      py::arg("position_cons_parents") = std::optional<at::Tensor>{},
      py::arg("position_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("position_cons_weights") = std::optional<at::Tensor>{},
      py::arg("position_cons_targets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_parents") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_weights") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_targets") = std::optional<at::Tensor>{},
      py::arg("pose_prior_model") = std::optional<at::Tensor>{},
      py::arg("motion_targets") = std::optional<at::Tensor>{},
      py::arg("motion_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_projections") = std::optional<at::Tensor>{},
      py::arg("projection_cons_parents") = std::optional<at::Tensor>{},
      py::arg("projection_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("projection_cons_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_targets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_origins") = std::optional<at::Tensor>{},
      py::arg("distance_cons_parents") = std::optional<at::Tensor>{},
      py::arg("distance_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_weights") = std::optional<at::Tensor>{},
      py::arg("distance_cons_targets") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_normals") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_type") = momentum::VertexConstraintType::Position,
      py::arg("vertex_proj_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_projections") = std::optional<at::Tensor>{});

  m.def(
      "jacobian",
      &computeJacobian,
      R"(
Compute the residual and Jacobian of a body IK problem.  For details on the constraint arguments, see :py:func:`solve_ik`.

Note that some momentum error functions may reorder the constraints in a nondeterministic manner
so repeated calls to this function could in principle return different results.  This will not
affect the gradient J^T * r, but you should avoid evaluating the residual and Jacobian in separate
function calls (which is why both are returned together).

For details on the arguments, see :py:func:`solve_ik`.

:return: A pair [r, J] where r is the (nBatch x m) residual and J is the (nBatch x m x nParams) Jacobian.
)",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("active_error_functions"),
      py::arg("error_function_weights"),
      /*
      py::kw_only(), // We reserve the right to reorder any of the following
                     // arguments; this allows us to (1) add additional
                     // constraints in the middle of the list; (2) add
                     // additional arguments to existing constraints.
      Therefore
                     // we force all of the following to be keyword-only
      using
                     // this "special" argument.
                     */
      py::arg("position_cons_parents") = std::optional<at::Tensor>{},
      py::arg("position_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("position_cons_weights") = std::optional<at::Tensor>{},
      py::arg("position_cons_targets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_parents") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_weights") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_targets") = std::optional<at::Tensor>{},
      py::arg("pose_prior_model") = std::optional<at::Tensor>{},
      py::arg("motion_targets") = std::optional<at::Tensor>{},
      py::arg("motion_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_projections") = std::optional<at::Tensor>{},
      py::arg("projection_cons_parents") = std::optional<at::Tensor>{},
      py::arg("projection_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("projection_cons_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_targets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_origins") = std::optional<at::Tensor>{},
      py::arg("distance_cons_parents") = std::optional<at::Tensor>{},
      py::arg("distance_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_weights") = std::optional<at::Tensor>{},
      py::arg("distance_cons_targets") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_normals") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_type") = momentum::VertexConstraintType::Position,
      py::arg("vertex_proj_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_projections") = std::optional<at::Tensor>{});

  // -------------------------------------------------------------
  //                    solveBodyIKProblem
  // -------------------------------------------------------------
  m.def(
      "solve_sequence_ik",
      &solveBodySequenceIKProblem,
      R"(
Batch solve a body multi-frame, or sequence, IK problem, with some parameters unique to each frame and some parameters shared between frames.
A typical use case is to solve for a single body scale plus a per-frame pose given per-frame constraints.

The constraints are the same ones available in :meth:`solve_ik`, except that instead of having tensor dimensions [nBatch x ...] they use tensor dimension
[nBatch x nFrames x ...].  For example: posConstraint_parents would be [nBatch x nFrames x nConstraints].  Note that you can choose to elide both the
per-batch and per-frame dimension (if you want to share parents across all frames and the entire batch) but you cannot skip one or the other unless nBatch is 1.
This is to prevent confusing ambiguities in whether you meant sharing across the batch dimension or the frame dimension.

:param character: Character or list of nBatch Characters to use in the solve.
:param activeParameters: boolean-valued torch.Tensor with dimension (k), which selects all the parameters (both shared and per-frame) to be active during the solve.
:param sharedParameters: boolean-valued torch.Tensor with dimension (k), which selects all the parameters to be shared across the frames (typically this would be scale or shape parameters).
:param modelParameters_init: float-valued torch.Tensor which contain the ([nBatch] x nFrames x nModelParameters)-dimensional model parameters used to initialize the solve.  This could be important as the solver always converges to a local minimum.
:param activeErrorFunctions: list of pymomentum.ErrorFunctionType which gives the order of input error function types.
:param errorFunctionWeights: float-valued torch.Tensor with dimension (nBatch x nFrames x len(activeErrorFunctions)) which contains a per-frame global weight for each active error function. The order of the weights is specified by activeErrorFunctions.
)",
      py::arg("character"),
      py::arg("active_parameters"),
      py::arg("shared_parameters"),
      py::arg("model_parameters_init"),
      py::arg("active_error_functions"),
      py::arg("error_function_weights"),
      py::arg("options") = SolverOptions(),
      py::arg("position_cons_parents") = std::optional<at::Tensor>{},
      py::arg("position_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("position_cons_weights") = std::optional<at::Tensor>{},
      py::arg("position_cons_targets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_parents") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_weights") = std::optional<at::Tensor>{},
      py::arg("orientation_cons_targets") = std::optional<at::Tensor>{},
      py::arg("pose_prior_model") = std::optional<at::Tensor>{},
      py::arg("motion_targets") = std::optional<at::Tensor>{},
      py::arg("motion_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_projections") = std::optional<at::Tensor>{},
      py::arg("projection_cons_parents") = std::optional<at::Tensor>{},
      py::arg("projection_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("projection_cons_weights") = std::optional<at::Tensor>{},
      py::arg("projection_cons_targets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_origins") = std::optional<at::Tensor>{},
      py::arg("distance_cons_parents") = std::optional<at::Tensor>{},
      py::arg("distance_cons_offsets") = std::optional<at::Tensor>{},
      py::arg("distance_cons_weights") = std::optional<at::Tensor>{},
      py::arg("distance_cons_targets") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_target_normals") = std::optional<at::Tensor>{},
      py::arg("vertex_cons_type") = momentum::VertexConstraintType::Position,
      py::arg("vertex_proj_cons_vertices") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_weights") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_target_positions") = std::optional<at::Tensor>{},
      py::arg("vertex_proj_cons_projections") = std::optional<at::Tensor>{});

  m.def(
      "get_solve_ik_statistics",
      &getSolveIKStatistics,
      R"(Return some basic statistics about ik solver in the forward pass of :meth:`solve_ik`.

This can be useful for detecting whether the input is appropriate; if the solve takes too many iterations,
then the input is likely not in a good shape.

:return: a pair [nTotalSolveIK, nTotalSolveIKIterations].
        )");

  m.def(
      "reset_solve_ik_statistics",
      &resetSolveIKStatistics,
      R"(Reset the counters used by :meth:`getSolveIKStatistics`.)");

  m.def(
      "get_gradient_statistics",
      &getGradientStatistics,
      R"(Return some basic statistics about how many IK gradients were nonzero in the backward pass of :meth:`solve_ik`.

This can be useful for verifying that the IK solver is converging properly; if the gradient is nonzero at the solution,
the derivatives will not be computed.

:return: a pair [nNonZeroGradients, nTotalGradients].
        )");

  m.def(
      "reset_gradient_statistics",
      &resetGradientStatistics,
      R"(Reset the counters used by :meth:`get_gradient_statistics`.)");

  m.def(
      "set_num_threads",
      [](int nThreads) {
        MT_THROW_IF(
            nThreads == 0, "Expected nThreads >= 1; use -1 to specify the number of processors.");

        if (nThreads < 0) {
          dispenso::globalThreadPool().resize(std::thread::hardware_concurrency());
          return;
        }

        MT_THROW_IF(
            nThreads > 2 * std::thread::hardware_concurrency(),
            "num_threads is too high; expected a value between 1 and 2x the number of processors (={}).",
            std::thread::hardware_concurrency());

        dispenso::globalThreadPool().resize(nThreads);
      },
      R"(Set the maximum number of threads used to solve IK problems.   This is particularly useful in limiting parallelism when you have multiple pytorch processes running on a single machine and want to limit how many threads each uses to avoid CPU contention.

:param num_threads: Number of threads to use.  Pass -1 to mean "number of processors".)",
      py::arg("num_threads"));
}
