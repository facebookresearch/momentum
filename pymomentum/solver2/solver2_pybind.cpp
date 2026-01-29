/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/mesh_state.h>
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
#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <pymomentum/solver2/solver2_error_functions.h>
#include <pymomentum/solver2/solver2_sequence_error_functions.h>
#include <pymomentum/solver2/solver2_utility.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

// Utility function to convert bool to Python-style string representation
std::string boolToString(bool value) {
  return value ? "True" : "False";
}

// Helper function to copy frame parameters from input array to solver function
void copyInputFrameParameters(
    const py::detail::unchecked_reference<float, 2>& modelParams_acc,
    mm::SequenceSolverFunction& solverFunction,
    py::ssize_t nFrames,
    py::ssize_t nParams) {
  for (py::ssize_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    Eigen::VectorXf modelParams_frame = Eigen::VectorXf::Zero(nParams);
    for (py::ssize_t k = 0; k < nParams; ++k) {
      modelParams_frame(k) = modelParams_acc(iFrame, k);
    }
    solverFunction.setFrameParameters(iFrame, modelParams_frame);
  }
}

// Helper function to copy frame parameters from solver function to output array
void copyOutputFrameParameters(
    py::detail::unchecked_mutable_reference<float, 2>& result_acc,
    const mm::SequenceSolverFunction& solverFunction,
    py::ssize_t nFrames,
    py::ssize_t nParams) {
  for (py::ssize_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    const auto& modelParams_frame = solverFunction.getFrameParameters(iFrame);
    for (py::ssize_t k = 0; k < nParams; ++k) {
      result_acc(iFrame, k) = modelParams_frame(k);
    }
  }
}

// Helper function to format error functions list for __repr__
std::string formatErrorFunctionsList(
    const std::vector<std::shared_ptr<mm::SkeletonErrorFunction>>& errorFunctions) {
  std::string result;
  for (size_t i = 0; i < errorFunctions.size(); ++i) {
    // Call the __repr__ method of each error function
    py::object pyErrorFunction = py::cast(errorFunctions[i]);
    result += py::str(pyErrorFunction);
    if (i + 1 < errorFunctions.size()) {
      result += ", ";
    }
  }
  return result;
}

PYBIND11_MODULE(solver2, m) {
  // TODO more explanation
  m.attr("__name__") = "pymomentum.solver2";
  m.doc() = "Inverse kinematics and other optimizations for momentum models.";

#ifdef PYMOMENTUM_LIMITED_TORCH_API
  m.attr("AUTOGRAD_ENABLED") = false;
#else
  m.attr("AUTOGRAD_ENABLED") = true;
#endif

  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbsource//arvr/libraries/pymomentum:geometry

  // Error functions:
  py::class_<mm::SkeletonErrorFunction, std::shared_ptr<mm::SkeletonErrorFunction>>(
      m, "SkeletonErrorFunction")
      .def(
          "__repr__",
          [](const mm::SkeletonErrorFunction& self) {
            return fmt::format("SkeletonErrorFunction(weight={})", self.getWeight());
          })
      .def_property(
          "weight",
          &mm::SkeletonErrorFunction::getWeight,
          [](mm::SkeletonErrorFunction& self, float weight) {
            validateWeight(weight, "weight");
            self.setWeight(weight);
          })
      .def(
          "get_error",
          [](mm::SkeletonErrorFunction& self, const py::array_t<float>& modelParameters) -> double {
            const auto& pt = self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(pt.apply(params), self.getSkeleton());
            momentum::MeshState meshState;
            if (self.needsMesh()) {
              const auto* character = self.getCharacter();
              MT_THROW_IF(character == nullptr, "Character is null but mesh is required.");
              meshState.update(params, state, *character);
            }
            return self.getError(params, state, meshState);
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
            const momentum::SkeletonState state(pt.apply(params), self.getSkeleton());
            momentum::MeshState meshState;
            if (self.needsMesh()) {
              const auto* character = self.getCharacter();
              MT_THROW_IF(character == nullptr, "Character is null but mesh is required.");
              meshState.update(params, state, *character);
            }
            Eigen::VectorXf gradient = Eigen::VectorXf::Zero(pt.numAllModelParameters());
            self.getGradient(params, state, meshState, gradient);
            return gradient;
          },
          R"(Compute the gradient for a given set of model parameters.

If you want to compute the gradient of multiple error functions, consider wrapping them in a :class:`SkeletonSolverFunction` and using :func:`SkeletonSolverFunction.get_gradient`.

:param model_parameters: The model parameters to evaluate.
:return: The gradient of the error function with respect to the model parameters.)",
          py::arg("model_parameters"))
      .def(
          "get_jacobian",
          [](mm::SkeletonErrorFunction& self, const py::array_t<float>& modelParameters)
              -> std::tuple<Eigen::VectorXf, Eigen::MatrixXf> {
            const auto& pt = self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(pt.apply(params), self.getSkeleton());

            momentum::MeshState meshState;
            if (self.needsMesh()) {
              const auto* character = self.getCharacter();
              MT_THROW_IF(character == nullptr, "Character is null but mesh is required.");
              meshState.update(params, state, *character);
            }

            const auto jacSize = self.getJacobianSize();
            Eigen::MatrixXf jacobian = Eigen::MatrixXf::Zero(jacSize, pt.numAllModelParameters());
            Eigen::VectorXf residual = Eigen::VectorXf::Zero(jacSize);
            int usedRows = 0;
            self.getJacobian(params, state, meshState, jacobian, residual, usedRows);
            return {residual.head(usedRows), jacobian.topRows(usedRows)};
          },
          R"(Compute the jacobian and residual for a given set of model parameters.

If you want to compute the jacobian of multiple error functions, consider wrapping them in a :class:`SkeletonSolverFunction` and using :func:`SkeletonSolverFunction.get_jacobian`.

:param model_parameters: The model parameters to evaluate.
:return: A tuple containing the residual and the jacobian of the error function with respect to the model parameters.)",
          py::arg("model_parameters"));

  addErrorFunctions(m);

  py::class_<mm::SequenceErrorFunction, std::shared_ptr<mm::SequenceErrorFunction>>(
      m, "SequenceErrorFunction");

  addSequenceErrorFunctions(m);

  // Solver functions:
  py::class_<mm::SolverFunction, std::shared_ptr<mm::SolverFunction>>(m, "SolverFunction");

  py::class_<
      mm::SkeletonSolverFunction,
      mm::SolverFunction,
      std::shared_ptr<mm::SkeletonSolverFunction>>(m, "SkeletonSolverFunction")
      .def(
          "__repr__",
          [](const mm::SkeletonSolverFunction& self) {
            std::string result = "SkeletonSolverFunction(error_functions=[";
            result += formatErrorFunctionsList(self.getErrorFunctions());
            result += "])";
            return result;
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 const std::vector<std::shared_ptr<mm::SkeletonErrorFunction>>& errorFunctions) {
                auto result = std::make_shared<mm::SkeletonSolverFunction>(
                    character, character.parameterTransform);
                for (const auto& errorFunction : errorFunctions) {
                  validateErrorFunctionMatchesCharacter(*result, *errorFunction);
                  result->addErrorFunction(errorFunction);
                }
                return result;
              }),
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("error_functions") = std::vector<std::shared_ptr<mm::SkeletonErrorFunction>>())
      .def(
          "add_error_function",
          [](mm::SkeletonSolverFunction& self,
             std::shared_ptr<mm::SkeletonErrorFunction> errorFunction) {
            validateErrorFunctionMatchesCharacter(self, *errorFunction);
            self.addErrorFunction(errorFunction);
          })
      .def(
          "clear_error_functions",
          &mm::SkeletonSolverFunction::clearErrorFunctions,
          "Clears all error functions from the solver.")
      .def_property(
          "error_functions",
          &mm::SkeletonSolverFunction::getErrorFunctions,
          [](mm::SkeletonSolverFunction& self,
             const std::vector<std::shared_ptr<mm::SkeletonErrorFunction>>& errorFunctions) {
            for (const auto& e : errorFunctions) {
              validateErrorFunctionMatchesCharacter(self, *e);
            }

            self.clearErrorFunctions();
            for (const auto& errorFunction : errorFunctions) {
              self.addErrorFunction(errorFunction);
            }
          })

      .def(
          "get_error",
          [](mm::SkeletonSolverFunction& self,
             const py::array_t<float>& modelParameters) -> double {
            const auto& pt = *self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(pt.apply(params), *self.getSkeleton());
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
            const momentum::SkeletonState state(pt.apply(params), *self.getSkeleton());
            Eigen::VectorXf gradient = Eigen::VectorXf::Zero(pt.numAllModelParameters());
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
          [](mm::SkeletonSolverFunction& self, const py::array_t<float>& modelParameters)
              -> std::tuple<Eigen::VectorXf, Eigen::MatrixXf> {
            const auto& pt = *self.getParameterTransform();
            auto params = toModelParameters(modelParameters, pt);
            const momentum::SkeletonState state(pt.apply(params), *self.getSkeleton());
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
          py::init<>([](const mm::Character& character,
                        size_t nFrames,
                        const std::optional<py::array_t<bool>>& shared_parameters) {
            auto result = std::make_shared<mm::SequenceSolverFunction>(
                character,
                character.parameterTransform,
                arrayToParameterSet(shared_parameters, character.parameterTransform, false),
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
            if (iFrame < 0 || iFrame >= self.getNumFrames()) {
              throw py::index_error(fmt::format("Invalid frame index {}", iFrame));
            }
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
            if (iFrame < 0 || iFrame >= self.getNumFrames()) {
              throw py::index_error(fmt::format("Invalid frame index {}", iFrame));
            }
            self.addSequenceErrorFunction(iFrame, errorFunction);
          },
          R"(Adds an error function that spans multiple frames in the solver.

:param frame_idx: The frame index to add the error function to.
:param error_function: The error function to add.)",
          py::arg("frame_idx"),
          py::arg("error_function"))
      .def(
          "add_sequence_error_function_all_frames",
          [](mm::SequenceSolverFunction& self,
             std::shared_ptr<mm::SequenceErrorFunction> errorFunction) {
            self.addSequenceErrorFunction(mm::kAllFrames, errorFunction);
          },
          R"(Adds an error function that spans all frames in the solver.

:param error_function: The error function to add.)",
          py::arg("error_function"))
      .def(
          "set_enabled_parameters",
          [](mm::SequenceSolverFunction& self, const py::array_t<bool>& activeParams) {
            self.setEnabledParameters(
                arrayToParameterSet(activeParams, *self.getParameterTransform(), true));
          },
          R"(Sets the active model parameters for the solver.  The array should be of size `character.parameter_transform.size()` and applies to every frame of the solve.

:param active_parameters: Numpy array of length num of model params.)",
          py::arg("active_parameters"))
      .def(
          "get_error_functions",
          [](const mm::SequenceSolverFunction& self, int frameIdx) {
            if (frameIdx < 0 || frameIdx >= self.getNumFrames()) {
              throw py::index_error(fmt::format("Invalid frame index {}", frameIdx));
            }
            return self.getErrorFunctions(frameIdx);
          },
          R"(Returns the per-frame error functions for a given frame.

          Note that this is read-only; appending to this list will not affect the SequenceErrorFunction,
          use :func:`add_error_function` instead.)",
          py::arg("frame_idx"))
      .def(
          "get_sequence_error_functions",
          [](const mm::SequenceSolverFunction& self, int frameIdx) {
            if (frameIdx < 0 || frameIdx >= self.getNumFrames()) {
              throw py::index_error(fmt::format("Invalid frame index {}", frameIdx));
            }
            return self.getSequenceErrorFunctions(frameIdx);
          },
          R"(Returns the sequence error functions.

          Note that this is read-only; appending to this list will not affect the SequenceErrorFunction,
          use :func:`add_sequence_error_function` instead.)",
          py::arg("frame_idx"));

  // Solver options:
  py::class_<mm::SolverOptions, std::shared_ptr<mm::SolverOptions>>(m, "SolverOptions")
      .def(py::init<>())
      .def(
          "__repr__",
          [](const mm::SolverOptions& self) {
            return fmt::format(
                "SolverOptions(min_iterations={}, max_iterations={}, threshold={}, verbose={})",
                self.minIterations,
                self.maxIterations,
                self.threshold,
                boolToString(self.verbose));
          })
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
          "verbose", &mm::SolverOptions::verbose, "Enable detailed logging during optimization");

  py::class_<
      mm::GaussNewtonSolverBaseOptions,
      mm::SolverOptions,
      std::shared_ptr<mm::GaussNewtonSolverBaseOptions>>(
      m, "GaussNewtonSolverBaseOptions", "Base options shared by all Gauss-Newton solver variants")
      .def(py::init<>())
      .def(
          "__repr__",
          [](const mm::GaussNewtonSolverBaseOptions& self) {
            return fmt::format(
                "GaussNewtonSolverBaseOptions(min_iterations={}, max_iterations={}, threshold={}, verbose={}, regularization={}, do_line_search={})",
                self.minIterations,
                self.maxIterations,
                self.threshold,
                boolToString(self.verbose),
                self.regularization,
                boolToString(self.doLineSearch));
          })
      .def_readwrite(
          "regularization",
          &mm::GaussNewtonSolverBaseOptions::regularization,
          "Damping parameter added to Hessian diagonal for numerical stability")
      .def_readwrite(
          "do_line_search",
          &mm::GaussNewtonSolverBaseOptions::doLineSearch,
          "Enables backtracking line search to ensure error reduction at each step");

  py::class_<
      mm::GaussNewtonSolverOptions,
      mm::GaussNewtonSolverBaseOptions,
      std::shared_ptr<mm::GaussNewtonSolverOptions>>(m, "GaussNewtonSolverOptions")
      .def(py::init<>())
      .def(
          "__repr__",
          [](const mm::GaussNewtonSolverOptions& self) {
            return fmt::format(
                "GaussNewtonSolverOptions(min_iterations={}, max_iterations={}, threshold={}, verbose={}, regularization={}, do_line_search={}, use_block_jtj={})",
                self.minIterations,
                self.maxIterations,
                self.threshold,
                boolToString(self.verbose),
                self.regularization,
                boolToString(self.doLineSearch),
                boolToString(self.useBlockJtJ));
          })
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
          "Uses pre-computed JᵀJ and JᵀR from the solver function");

  py::class_<
      mm::GaussNewtonSolverQROptions,
      mm::GaussNewtonSolverBaseOptions,
      std::shared_ptr<mm::GaussNewtonSolverQROptions>>(
      m, "GaussNewtonSolverQROptions", "Options specific to the Gauss-Newton QR solver")
      .def(py::init<>())
      .def("__repr__", [](const mm::GaussNewtonSolverQROptions& self) {
        return fmt::format(
            "GaussNewtonSolverQROptions(min_iterations={}, max_iterations={}, threshold={}, verbose={}, regularization={}, do_line_search={})",
            self.minIterations,
            self.maxIterations,
            self.threshold,
            boolToString(self.verbose),
            self.regularization,
            boolToString(self.doLineSearch));
      });

  py::class_<
      mm::SubsetGaussNewtonSolverOptions,
      mm::GaussNewtonSolverBaseOptions,
      std::shared_ptr<mm::SubsetGaussNewtonSolverOptions>>(
      m, "SubsetGaussNewtonSolverOptions", "Options specific to the Subset Gauss-Newton solver")
      .def(py::init<>())
      .def("__repr__", [](const mm::SubsetGaussNewtonSolverOptions& self) {
        return fmt::format(
            "SubsetGaussNewtonSolverOptions(min_iterations={}, max_iterations={}, threshold={}, verbose={}, regularization={}, do_line_search={})",
            self.minIterations,
            self.maxIterations,
            self.threshold,
            boolToString(self.verbose),
            self.regularization,
            boolToString(self.doLineSearch));
      });

  py::class_<
      mm::SequenceSolverOptions,
      mm::SolverOptions,
      std::shared_ptr<mm::SequenceSolverOptions>>(
      m, "SequenceSolverOptions", "Options specific to the sequence solver")
      .def(py::init<>())
      .def(
          "__repr__",
          [](const mm::SequenceSolverOptions& self) {
            return fmt::format(
                "SequenceSolverOptions(min_iterations={}, max_iterations={}, threshold={}, verbose={}, regularization={}, do_line_search={}, multithreaded={}, progress_bar={})",
                self.minIterations,
                self.maxIterations,
                self.threshold,
                boolToString(self.verbose),
                self.regularization,
                boolToString(self.doLineSearch),
                boolToString(self.multithreaded),
                boolToString(self.progressBar));
          })
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
            self.setEnabledParameters(
                arrayToParameterSet(activeParams, self.getNumParameters(), true));
          },
          R"(Sets the active model parameters for the solver.  The array should be of size `character.parameter_transform.size()` and applies to every frame of the solve.

:param active_parameters: Numpy array of length num of model params.)",
          py::arg("active_parameters"))
      .def(
          "solve",
          [](mm::Solver& self, Eigen::VectorXf parameters) -> Eigen::VectorXf {
            if (parameters.size() != self.getNumParameters()) {
              throw std::runtime_error(
                  "Expected parameters to be of size " + std::to_string(self.getNumParameters()));
            }
            self.solve(parameters);
            return parameters;
          },
          py::arg("model_parameters"))
      .def_property_readonly(
          "per_iteration_errors",
          [](mm::Solver& self) { return self.getErrorHistory(); },
          "The error after each iteration of the solver.");

  py::class_<mm::GaussNewtonSolver, mm::Solver, std::shared_ptr<mm::GaussNewtonSolver>>(
      m, "GaussNewtonSolver", "Basic 2nd-order Gauss-Newton solver")
      .def(
          py::init<>([](mm::SolverFunction* solverFunction,
                        const std::optional<mm::GaussNewtonSolverOptions>& options) {
            return std::make_shared<mm::GaussNewtonSolver>(
                options.value_or(mm::GaussNewtonSolverOptions{}), solverFunction);
          }),
          py::keep_alive<1, 2>(),
          py::arg("solver_function"),
          py::arg("options") = std::optional<mm::GaussNewtonSolverOptions>{});

  py::class_<mm::GaussNewtonSolverQR, mm::Solver, std::shared_ptr<mm::GaussNewtonSolverQR>>(
      m, "GaussNewtonSolverQR", "Gauss-Newton solver with QR decomposition")
      .def(
          py::init<>([](mm::SkeletonSolverFunction* solverFunction,
                        const std::optional<mm::GaussNewtonSolverQROptions>& options) {
            return std::make_shared<mm::GaussNewtonSolverQRT<float>>(
                options.value_or(mm::GaussNewtonSolverQROptions{}), solverFunction);
          }),
          py::keep_alive<1, 2>(),
          py::arg("solver_function"),
          py::arg("options") = std::optional<mm::GaussNewtonSolverQROptions>{});

  py::class_<mm::SubsetGaussNewtonSolver, mm::Solver, std::shared_ptr<mm::SubsetGaussNewtonSolver>>(
      m,
      "SubsetGaussNewtonSolver",
      "Gauss-Newton solver that optimizes only a selected subset of parameters")
      .def(
          py::init<>([](mm::SolverFunction* solverFunction,
                        const std::optional<mm::SubsetGaussNewtonSolverOptions>& options) {
            return std::make_shared<mm::SubsetGaussNewtonSolver>(
                options.value_or(mm::SubsetGaussNewtonSolverOptions{}), solverFunction);
          }),
          py::keep_alive<1, 2>(),
          py::arg("solver_function"),
          py::arg("options") = std::optional<mm::SubsetGaussNewtonSolverOptions>{});

  m.def(
      "solve_sequence",
      [](mm::SequenceSolverFunction& solverFunction,
         const py::array_t<float>& modelParams,
         const std::optional<mm::SequenceSolverOptions>& options) -> py::array_t<float> {
        mm::SequenceSolver solver(options.value_or(mm::SequenceSolverOptions{}), &solverFunction);

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
          copyInputFrameParameters(modelParams_acc, solverFunction, nFrames, nParams);
        }

        {
          py::gil_scoped_release release;
          Eigen::VectorXf parameters = solverFunction.getJoinedParameterVector();
          solver.solve(parameters);
          solverFunction.setJoinedParameterVector(parameters);
        }

        py::array_t<float> result = py::array_t<float>({nFrames, nParams});
        {
          auto result_acc = result.mutable_unchecked<2>();
          py::gil_scoped_release release;
          copyOutputFrameParameters(result_acc, solverFunction, nFrames, nParams);
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
