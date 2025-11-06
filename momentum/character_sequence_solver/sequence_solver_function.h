/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/solver/solver_function.h>

#include <atomic>

namespace momentum {

template <typename T>
struct PerFrameStateT {
  SkeletonStateT<T> skeletonState;
  MeshStateT<T> meshState;
};

inline constexpr size_t kAllFrames = SIZE_MAX;

template <typename T>
class SequenceSolverFunctionT : public SolverFunctionT<T> {
 public:
  SequenceSolverFunctionT(
      const Character& character,
      const ParameterTransformT<T>& parameterTransform,
      const ParameterSet& universal,
      size_t nFrames);
  ~SequenceSolverFunctionT() override;

  double getError(const Eigen::VectorX<T>& parameters) final;

  double getGradient(const Eigen::VectorX<T>& parameters, Eigen::VectorX<T>& gradient) final;

  double getJacobian(
      const Eigen::VectorX<T>& parameters,
      Eigen::MatrixX<T>& jacobian,
      Eigen::VectorX<T>& residual,
      size_t& actualRows) final;

  void updateParameters(Eigen::VectorX<T>& parameters, const Eigen::VectorX<T>& gradient) final;
  void setEnabledParameters(const ParameterSet& parameterSet) final;

  [[nodiscard]] const ParameterSet& getUniversalParameterSet() const {
    return universalParameters_;
  }

  /// Returns whether the per-frame error functions need the mesh.
  [[nodiscard]] bool needsMeshPerFrame() const {
    return needsMeshPerFrame_;
  }

  /// Returns whether the sequence error functions need the mesh.
  [[nodiscard]] bool needsMeshSequence() const {
    return needsMeshSequence_;
  }

  // Passing in the special frame index kAllFrames will add the error function to every frame; this
  // is convenient for e.g. limit errors but requires that the error function be stateless. Note:
  // you are allowed to call this in a multithreaded context but you must ensure the frame indices
  // are different between the different threads.
  void addErrorFunction(size_t frame, std::shared_ptr<SkeletonErrorFunctionT<T>> errorFunction);
  void addSequenceErrorFunction(
      size_t startFrame,
      std::shared_ptr<SequenceErrorFunctionT<T>> errorFunction);

  [[nodiscard]] size_t getNumFrames() const {
    return frameParameters_.size();
  }

  [[nodiscard]] const ModelParametersT<T>& getFrameParameters(size_t frame) const {
    return frameParameters_[frame];
  }

  void setFrameParameters(size_t frame, const ModelParametersT<T>& parameters);
  [[nodiscard]] ModelParametersT<T> getUniversalParameters() const;
  [[nodiscard]] Eigen::VectorX<T> getJoinedParameterVector() const;

  /// Returns a joined parameter vector from a span of frame parameters.
  ///
  /// @param frameParameters A span of frame parameters, where each element is a ModelParametersT<T>
  /// object representing the parameters for a single frame.
  ///
  /// @return An Eigen::VectorX<T> object containing the joined parameter vector of all frames.
  [[nodiscard]] Eigen::VectorX<T> getJoinedParameterVectorFromFrameParameters(
      std::span<const ModelParametersT<T>> frameParameters) const;

  void setJoinedParameterVector(const Eigen::VectorX<T>& joinedParameters);

  [[nodiscard]] const Skeleton* getSkeleton() const;
  [[nodiscard]] const Character& getCharacter() const;

  [[nodiscard]] const ParameterTransformT<T>* getParameterTransform() const {
    return &parameterTransform_;
  }

  [[nodiscard]] const auto& getErrorFunctions(size_t iFrame) const {
    return perFrameErrorFunctions_.at(iFrame);
  }

  [[nodiscard]] const auto& getSequenceErrorFunctions(size_t iFrame) const {
    return sequenceErrorFunctions_.at(iFrame);
  }

 private:
  void setFrameParametersFromJoinedParameterVector(const Eigen::VectorX<T>& parameters);

  /// Helper method to compute per-frame jacobians
  double
  getPerFrameJacobian(Eigen::MatrixX<T>& jacobian, Eigen::VectorX<T>& residual, size_t& position);

  /// Helper method to compute sequence error functions jacobians
  double getSequenceErrorFunctionsJacobian(
      Eigen::MatrixX<T>& jacobian,
      Eigen::VectorX<T>& residual,
      size_t& position);

 private:
  const Character& character_;
  const ParameterTransformT<T>& parameterTransform_;
  VectorX<bool> activeJointParams_;

  std::vector<ModelParametersT<T>> frameParameters_;
  size_t nFrames_;

  void updateParameterSets(const ParameterSet& activeParams);

  // Indices of parameters that are active and solved per-frame:
  std::vector<Eigen::Index> perFrameParameterIndices_;

  // Indices of parameters that are active and solved universally:
  std::vector<Eigen::Index> universalParameterIndices_;

  const ParameterSet universalParameters_;

  std::vector<std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>>> perFrameErrorFunctions_;
  std::vector<std::vector<std::shared_ptr<SequenceErrorFunctionT<T>>>> sequenceErrorFunctions_;

  std::atomic<size_t> numTotalPerFrameErrorFunctions_ = 0;
  std::atomic<size_t> numTotalSequenceErrorFunctions_ = 0;

  // Whether per-frame error functions need the mesh
  std::atomic<bool> needsMeshPerFrame_{false};
  // Whether sequence error functions need the mesh
  std::atomic<bool> needsMeshSequence_{false};

  friend class SequenceSolverT<T>;
};

} // namespace momentum
