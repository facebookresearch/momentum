/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

 #include <momentum/character/character.h>
 #include <momentum/character/character_state.h>
 #include <momentum/common/filesystem.h>
 #include <momentum/common/log.h>
 #include <momentum/gui/rerun/logger.h>
 #include <momentum/gui/rerun/logging_redirect.h>
 #include <momentum/io/urdf/urdf_io.h>
 #include "momentum/character/character.h"
 #include "momentum/character/parameter_transform.h"
 #include "momentum/character/skeleton.h"
 #include "momentum/character/skeleton_state.h"
 #include "momentum/character_solver/position_error_function.h"
 #include "momentum/diff_ik/fully_differentiable_position_error_function.h"
 #include "momentum/character_solver/skeleton_solver_function.h"
 #include "momentum/character_solver/plane_error_function.h"
 #include "momentum/solver/fwd.h"
 #include "momentum/solver/gauss_newton_solver.h"
 #include "momentum/character_solver/gauss_newton_solver_qr.h"
 #include "momentum/solver/gradient_descent_solver.h"
 
 #include <string>
 #include <iostream>
 
 
 using namespace momentum;
 
 template <typename T>
 void solve_ik()
 {
   const Character character = loadUrdfCharacter("urdf/fr3_no_gripper.urdf");
   std::cout << "Character loaded successfully!" << std::endl;
 
   const Skeleton& skeleton = character.skeleton;
   const ParameterTransform& transform = character.parameterTransform;
   
   VectorX<T> parameters = VectorX<T>::Zero(transform.numAllModelParameters());
   VectorX<T> optimizedParameters = parameters;
   
   // // create skeleton solvable
   SkeletonSolverFunctionT<T> solverFunction(&skeleton, &transform);
 
    auto errorFunction = std::make_shared<PositionErrorFunction>(skeleton, transform);
 
    //add to solvable

    GaussNewtonSolverOptions options;
     options.maxIterations = 6;
     options.minIterations = 6;
     options.threshold = 1.0;
     options.regularization = 1e-7;
     options.useBlockJtJ = true;
     
   
    // generate rest state and constraints
   std::vector<PositionDataT<T>> constraints;
   constraints.push_back(
    PositionDataT<T>(Vector3<T>::Zero(), Vector3<T>(30.0, 0.0, 40.0), 6, 1.0));
 
   errorFunction->addConstraints(constraints);
   
   solverFunction.addErrorFunction(errorFunction);
     
   GaussNewtonSolver solver(options, &solverFunction);
 
 
   // Line creates segmentation fault
   solver.solve(optimizedParameters);
 
 
 
 
   
 
   return;
 }
 
 int main(){
 
 
   
 
     solve_ik<float>();
     std::cout << "IK solved successfully!" << std::endl;
 
 
 
   return 0;
 }