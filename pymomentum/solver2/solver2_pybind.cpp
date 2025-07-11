/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_sequence_solver/sequence_error_function.h>
#include <momentum/character_sequence_solver/sequence_solver.h>
#include <momentum/character_sequence_solver/sequence_solver_function.h>
#include <momentum/character_solver/gauss_newton_solver_qr.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/solver/gauss_newton_solver.h>
#include <momentum/solver/subset_gauss_newton_solver.h>

#include <dispenso/parallel_for.h> // @manual
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <pymomentum/solver2/solver2_error_functions.h>
#include <pymomentum/solver2/solver2_sequence_error_functions.h>
#include <pymomentum/solver2/solver2_utility.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

PYBIND11_MODULE(solver2, m) {
  // TODO more explanation
  m.attr("__name__") = "pymomentum.solver2";
  m.doc() = "Inverse kinematics and other optimizations for momentum models.";

  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbcode//pymomentum:geometry

  // Error functions:
  py::class_<
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::SkeletonErrorFunction>>(m, "SkeletonErrorFunction")
      .def_property(
          "weight",
          &mm::SkeletonErrorFunction::setWeight,
          &mm::SkeletonErrorFunction::getWeight)
      .def(
          "get_error",
          [](mm::SkeletonErrorFunction& self,
             const py::array_t<float>& modelParameters) -> double {
            const auto& pt = self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(
                pt.apply(params), self.getSkeleton());
            return self.getError(params, state);
          },
          R"(Compute the error for a given set of model parameters.

If you want to compute the error of multiple error functions, consider wrapping them in a :class:`SkeletonSolverFunction` and using :func:`SkeletonSolverFunction.get_error`.

:param model_parameters: The model parameters to evaluate.
:return: The error value.)",
          py::arg("model_parameters"))
      .def(
          "get_gradient",
          [](mm::SkeletonErrorFunction& self,
             const py::array_t<float>& modelParameters) -> Eigen::VectorXf {
            const auto& pt = self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(
                pt.apply(params), self.getSkeleton());
            Eigen::VectorXf gradient =
                Eigen::VectorXf::Zero(pt.numAllModelParameters());
            self.getGradient(params, state, gradient);
            return gradient;
          },
          R"(Compute the gradient for a given set of model parameters.

If you want to compute the gradient of multiple error functions, consider wrapping them in a :class:`SkeletonSolverFunction` and using :func:`SkeletonSolverFunction.get_gradient`.

:param model_parameters: The model parameters to evaluate.
:return: The gradient of the error function with respect to the model parameters.)",
          py::arg("model_parameters"))
      .def(
          "get_jacobian",
          [](mm::SkeletonErrorFunction& self,
             const py::array_t<float>& modelParameters)
              -> std::tuple<Eigen::VectorXf, Eigen::MatrixXf> {
            const auto& pt = self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(
                pt.apply(params), self.getSkeleton());
            const auto jacSize = self.getJacobianSize();
            Eigen::MatrixXf jacobian =
                Eigen::MatrixXf::Zero(jacSize, pt.numAllModelParameters());
            Eigen::VectorXf residual = Eigen::VectorXf::Zero(jacSize);
            int usedRows = 0;
            self.getJacobian(params, state, jacobian, residual, usedRows);
            return {residual.head(usedRows), jacobian.topRows(usedRows)};
          },
          R"(Compute the jacobian and residual for a given set of model parameters.

If you want to compute the jacobian of multiple error functions, consider wrapping them in a :class:`SkeletonSolverFunction` and using :func:`SkeletonSolverFunction.get_jacobian`.

:param model_parameters: The model parameters to evaluate.
:return: A tuple containing the residual and the jacobian of the error function with respect to the model parameters.)",
          py::arg("model_parameters"));

  addErrorFunctions(m);

  py::class_<
      mm::SequenceErrorFunction,
      std::shared_ptr<mm::SequenceErrorFunction>>(m, "SequenceErrorFunction");

  addSequenceErrorFunctions(m);

  // Solver functions:
  py::class_<mm::SolverFunction, std::shared_ptr<mm::SolverFunction>>(
      m, "SolverFunction");

  py::class_<
      mm::SkeletonSolverFunction,
      mm::SolverFunction,
      std::shared_ptr<mm::SkeletonSolverFunction>>(m, "SkeletonSolverFunction")
      .def(
          py::init<>(
              [](const mm::Character& character,
                 const std::vector<std::shared_ptr<mm::SkeletonErrorFunction>>&
                     errorFunctions) {
                auto result = std::make_shared<mm::SkeletonSolverFunction>(
                    &character.skeleton, &character.parameterTransform);
                for (const auto& errorFunction : errorFunctions) {
                  result->addErrorFunction(errorFunction);
                }
                return result;
              }),
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("error_functions") =
              std::vector<std::shared_ptr<mm::SkeletonErrorFunction>>())
      .def(
          "add_error_function",
          [](mm::SkeletonSolverFunction& self,
             std::shared_ptr<mm::SkeletonErrorFunction> errorFunction) {
            self.addErrorFunction(errorFunction);
          })
      .def(
          "get_error",
          [](mm::SkeletonSolverFunction& self,
             const py::array_t<float>& modelParameters) -> double {
            const auto& pt = *self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(
                pt.apply(params), *self.getSkeleton());
            return self.getError(params.v);
          },
          R"(Compute the error for a given set of model parameters.

  :param model_parameters: The model parameters to evaluate.
  :return: The error value.)",
          py::arg("model_parameters"))
      .def(
          "get_gradient",
          [](mm::SkeletonSolverFunction& self,
             const py::array_t<float>& modelParameters) -> Eigen::VectorXf {
            const auto& pt = *self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(
                pt.apply(params), *self.getSkeleton());
            Eigen::VectorXf gradient =
                Eigen::VectorXf::Zero(pt.numAllModelParameters());
            self.getGradient(params.v, gradient);
            return gradient;
          },
          R"(Compute the gradient for a given set of model parameters.

Note that if you're trying to actually solve a problem using SGD, you should consider using one of the :class:`Solver` class instead, which are more efficient since they don't require passing anything through the Python binding layers.

  :param model_parameters: The model parameters to evaluate.
  :return: The gradient of the error function with respect to the model parameters.)",
          py::arg("model_parameters"))
      .def(
          "get_jacobian",
          [](mm::SkeletonSolverFunction& self,
             const py::array_t<float>& modelParameters)
              -> std::tuple<Eigen::VectorXf, Eigen::MatrixXf> {
            const auto& pt = *self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(
                pt.apply(params), *self.getSkeleton());
            Eigen::MatrixXf jacobian;
            Eigen::VectorXf residual;
            size_t actualRows = 0;
            self.getJacobian(params.v, jacobian, residual, actualRows);
            return {residual.head(actualRows), jacobian.topRows(actualRows)};
          },
          R"(Compute the jacobian and residual for a given set of model parameters.

  :param model_parameters: The model parameters to evaluate.
  :return: A tuple containing the residual and the jacobian of the error function with respect to the model parameters.)",
          py::arg("model_parameters"));

  py::class_<
      mm::SequenceSolverFunction,
      mm::SolverFunction,
      std::shared_ptr<mm::SequenceSolverFunction>>(m, "SequenceSolverFunction")
      .def(
          py::init<>(
              [](const mm::Character& character,
                 size_t nFrames,
                 const std::optional<py::array_t<bool>>& shared_parameters) {
                auto result = std::make_shared<mm::SequenceSolverFunction>(
                    &character.skeleton,
                    &character.parameterTransform,
                    arrayToParameterSet(
                        shared_parameters, character.parameterTransform, false),
                    nFrames);
                return result;
              }),
          R"(Creates a new solver function for a sequence of poses.  This allows you to optimize a sequence of poses, where the shared parameters are the same for each frame.

:param character: The character to use.
:param n_frames: The number of frames in the sequence.
:param shared_parameters: The parameters that are shared between all frames.  This is a boolean array of size `character.parameter_transform.size()`. )",

          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("n_frames"),
          py::arg("shared_parameters") = std::optional<py::array_t<bool>>{})
      .def(
          "add_error_function",
          [](mm::SequenceSolverFunction& self,
             int iFrame,
             std::shared_ptr<mm::SkeletonErrorFunction> errorFunction) {
            self.addErrorFunction(iFrame, errorFunction);
          },
          R"(Adds an error function that applies to a single frame to the solver.

:param frame_idx: The frame index to add the error function to.
:param error_function: The error function to add.
        )",
          py::arg("frame_idx"),
          py::arg("error_function"))
      .def(
          "add_error_function",
          [](mm::SequenceSolverFunction& self,
             std::shared_ptr<mm::SkeletonErrorFunction> errorFunction) {
            self.addErrorFunction(mm::kAllFrames, errorFunction);
          },
          R"(Adds an error function that applies to all frames to the solver.

:param error_function: The error function to add.
        )",
          py::arg("error_function"))
      .def(
          "add_sequence_error_function",
          [](mm::SequenceSolverFunction& self,
             int iFrame,
             std::shared_ptr<mm::SequenceErrorFunction> errorFunction) {
            self.addSequenceErrorFunction(iFrame, errorFunction);
          },
          R"(Adds an error function that spans multiple frames in the solver.

:param frame_idx: The frame index to add the error function to.
:param error_function: The error function to add.)",
          py::arg("frame_idx"),
          py::arg("error_function"))
      .def(
          "set_enabled_parameters",
          [](mm::SequenceSolverFunction& self,
             const py::array_t<bool>& activeParams) {
            self.setEnabledParameters(arrayToParameterSet(
                activeParams, *self.getParameterTransform(), true));
          },
          R"(Sets the active model parameters for the solver.  The array should be of size `character.parameter_transform.size()` and applies to every frame of the solve.

:param active_parameters: Numpy array of length num of model params.)",
          py::arg("active_parameters"));

  // Solver options:
  py::class_<mm::SolverOptions, std::shared_ptr<mm::SolverOptions>>(
      m, "SolverOptions")
      .def(py::init<>())
      .def_readwrite(
          "min_iterations",
          &mm::SolverOptions::minIterations,
          "Minimum number of iterations before checking convergence criteria")
      .def_readwrite(
          "max_iterations",
          &mm::SolverOptions::maxIterations,
          "Maximum number of iterations before terminating")
      .def_readwrite(
          "threshold",
          &mm::SolverOptions::threshold,
          "Convergence threshold for relative error change")
      .def_readwrite(
          "verbose",
          &mm::SolverOptions::verbose,
          "Enable detailed logging during optimization");

  py::class_<
      mm::GaussNewtonSolverOptions,
      mm::SolverOptions,
      std::shared_ptr<mm::GaussNewtonSolverOptions>>(
      m, "GaussNewtonSolverOptions")
      .def(py::init<>())
      .def_readwrite(
          "regularization",
          &mm::GaussNewtonSolverOptions::regularization,
          "Damping parameter added to Hessian diagonal for numerical stability")
      .def_readwrite(
          "do_line_search",
          &mm::GaussNewtonSolverOptions::doLineSearch,
          "Enables backtracking line search to ensure error reduction at each step")
      .def_readwrite(
          "use_block_jtj",
          &mm::GaussNewtonSolverOptions::useBlockJtJ,
          "Uses pre-computed JᵀJ and JᵀR from the solver function")
      .def_readwrite(
          "direct_sparse_jtj",
          &mm::GaussNewtonSolverOptions::directSparseJtJ,
          "Directly computes sparse JᵀJ without dense intermediate representation")
      .def_readwrite(
          "sparse_matrix_threshold",
          &mm::GaussNewtonSolverOptions::sparseMatrixThreshold,
          "Parameter count threshold for switching to sparse matrix operations");

  py::class_<
      mm::GaussNewtonSolverQROptions,
      mm::SolverOptions,
      std::shared_ptr<mm::GaussNewtonSolverQROptions>>(
      m,
      "GaussNewtonSolverQROptions",
      "Options specific to the Gauss-Newton QR solver")
      .def(py::init<>())
      .def_readwrite(
          "regularization",
          &mm::GaussNewtonSolverQROptions::regularization,
          "Regularization parameter for QR decomposition.")
      .def_readwrite(
          "do_line_search",
          &mm::GaussNewtonSolverQROptions::doLineSearch,
          "Flag to enable line search during optimization.");

  py::class_<
      mm::SubsetGaussNewtonSolverOptions,
      mm::SolverOptions,
      std::shared_ptr<mm::SubsetGaussNewtonSolverOptions>>(
      m,
      "SubsetGaussNewtonSolverOptions",
      "Options specific to the Subset Gauss-Newton solver")
      .def(py::init<>())
      .def_readwrite(
          "regularization",
          &mm::SubsetGaussNewtonSolverOptions::regularization,
          "Damping parameter added to Hessian diagonal for numerical stability")
      .def_readwrite(
          "do_line_search",
          &mm::SubsetGaussNewtonSolverOptions::doLineSearch,
          "Enables backtracking line search to ensure error reduction at each step");

  py::class_<
      mm::SequenceSolverOptions,
      mm::SolverOptions,
      std::shared_ptr<mm::SequenceSolverOptions>>(
      m, "SequenceSolverOptions", "Options specific to the sequence solver")
      .def(py::init<>())
      .def_readwrite(
          "regularization",
          &mm::SequenceSolverOptions::regularization,
          "Regularization parameter for QR decomposition.")
      .def_readwrite(
          "do_line_search",
          &mm::SequenceSolverOptions::doLineSearch,
          "Flag to enable line search during optimization.")
      .def_readwrite(
          "multithreaded",
          &mm::SequenceSolverOptions::multithreaded,
          "Flag to enable multithreading for the Sequence solver.")
      .def_readwrite(
          "progress_bar",
          &mm::SequenceSolverOptions::progressBar,
          "Flag to enable a progress bar during optimization.");

  py::class_<mm::Solver, std::shared_ptr<mm::Solver>>(m, "Solver")
      .def(
          "set_enabled_parameters",
          [](mm::Solver& self, const py::array_t<bool>& activeParams) {
            self.setEnabledParameters(arrayToParameterSet(
                activeParams, self.getNumParameters(), true));
          },
          R"(Sets the active model parameters for the solver.  The array should be of size `character.parameter_transform.size()` and applies to every frame of the solve.

:param active_parameters: Numpy array of length num of model params.)",
          py::arg("active_parameters"))
      .def(
          "solve",
          [](mm::Solver& self, Eigen::VectorXf parameters) -> Eigen::VectorXf {
            self.solve(parameters);
            return parameters;
          },
          py::arg("model_parameters"))
      .def_property_readonly(
          "per_iteration_errors",
          [](mm::Solver& self) { return self.getErrorHistory(); },
          "The error after each iteration of the solver.");

  py::class_<
      mm::GaussNewtonSolver,
      mm::Solver,
      std::shared_ptr<mm::GaussNewtonSolver>>(
      m, "GaussNewtonSolver", "Basic 2nd-order Gauss-Newton solver")
      .def(
          py::init<>(
              [](mm::SolverFunction* solverFunction,
                 const std::optional<mm::GaussNewtonSolverOptions>& options) {
                return std::make_shared<mm::GaussNewtonSolver>(
                    options.value_or(mm::GaussNewtonSolverOptions{}),
                    solverFunction);
              }),
          py::keep_alive<1, 2>(),
          py::arg("solver_function"),
          py::arg("options") = std::optional<mm::GaussNewtonSolverOptions>{});

  py::class_<
      mm::GaussNewtonSolverQR,
      mm::Solver,
      std::shared_ptr<mm::GaussNewtonSolverQR>>(
      m, "GaussNewtonSolverQR", "Gauss-Newton solver with QR decomposition")
      .def(
          py::init<>(
              [](mm::SkeletonSolverFunction* solverFunction,
                 const std::optional<mm::GaussNewtonSolverQROptions>& options) {
                return std::make_shared<mm::GaussNewtonSolverQRT<float>>(
                    options.value_or(mm::GaussNewtonSolverQROptions{}),
                    solverFunction);
              }),
          py::keep_alive<1, 2>(),
          py::arg("solver_function"),
          py::arg("options") = std::optional<mm::GaussNewtonSolverQROptions>{});

  py::class_<
      mm::SubsetGaussNewtonSolver,
      mm::Solver,
      std::shared_ptr<mm::SubsetGaussNewtonSolver>>(
      m,
      "SubsetGaussNewtonSolver",
      "Gauss-Newton solver that optimizes only a selected subset of parameters")
      .def(
          py::init<>([](mm::SolverFunction* solverFunction,
                        const std::optional<mm::SubsetGaussNewtonSolverOptions>&
                            options) {
            return std::make_shared<mm::SubsetGaussNewtonSolver>(
                options.value_or(mm::SubsetGaussNewtonSolverOptions{}),
                solverFunction);
          }),
          py::keep_alive<1, 2>(),
          py::arg("solver_function"),
          py::arg("options") =
              std::optional<mm::SubsetGaussNewtonSolverOptions>{});

  m.def(
      "solve_sequence",
      [](mm::SequenceSolverFunction& solverFunction,
         const py::array_t<float>& modelParams,
         const std::optional<mm::SequenceSolverOptions>& options)
          -> py::array_t<float> {
        mm::SequenceSolver solver(
            options.value_or(mm::SequenceSolverOptions{}), &solverFunction);

        if (modelParams.ndim() != 2) {
          throw std::runtime_error(
              "Expected model parameters to be a 2D array of shape (n_frames, n_params)");
        }

        if (modelParams.shape(1) !=
            solverFunction.getParameterTransform()->numAllModelParameters()) {
          throw std::runtime_error(
              "Expected model parameters to have n_params == " +
              std::to_string(solverFunction.getNumParameters()));
        }

        const auto nFrames = modelParams.shape(0);
        const auto nParams = modelParams.shape(1);

        {
          auto modelParams_acc = modelParams.unchecked<2>();
          py::gil_scoped_release release;
          for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
            Eigen::VectorXf modelParams_frame = Eigen::VectorXf::Zero(nParams);
            for (int k = 0; k < nParams; ++k) {
              modelParams_frame(k) = modelParams_acc(iFrame, k);
            }
            solverFunction.setFrameParameters(iFrame, modelParams_frame);
          }
        }

        {
          py::gil_scoped_release release;
          Eigen::VectorXf parameters =
              solverFunction.getJoinedParameterVector();
          solver.solve(parameters);
          solverFunction.setJoinedParameterVector(parameters);
        }

        py::array_t<float> result = py::array_t<float>({nFrames, nParams});
        {
          auto result_acc = result.mutable_unchecked<2>();
          py::gil_scoped_release release;
          for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
            const auto& modelParams_frame =
                solverFunction.getFrameParameters(iFrame);
            for (int k = 0; k < nParams; ++k) {
              result_acc(iFrame, k) = modelParams_frame(k);
            }
          }
        }

        return result;
      },
      R"(Solves a sequence of poses using the sequence solver.

:param solver_function: The solver function to use.
:param model_params: The initial model parameters.  This is an n_frames x n_params array.
:param options: The solver options to use.
:returns: The optimized model parameters.)",
      py::arg("solver_function"),
      py::arg("model_params"),
      py::arg("options") = std::optional<mm::SequenceSolverOptions>{});
}

} // namespace pymomentum
