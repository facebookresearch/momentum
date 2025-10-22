/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/diff_ik/union_error_function.h>

#include "momentum/common/exception.h"

namespace momentum {

template <typename T>
UnionErrorFunctionT<T>::UnionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>>& errorFunctions)
    : SkeletonErrorFunctionT<T>(skel, pt), errorFunctions_(errorFunctions) {
  // Should this be allowed?
  MT_THROW_IF(errorFunctions_.empty(), "No error functions in union?");

  for (size_t i = 0; i < errorFunctions_.size(); ++i) {
    auto diff_e =
        std::dynamic_pointer_cast<FullyDifferentiableSkeletonErrorFunctionT<T>>(errorFunctions_[i]);
    diffErrorFunctions_.push_back(diff_e);

    if (diff_e) {
      for (const auto& input : diff_e->inputs()) {
        inputs_[input].push_back(i);
      }
    }
  }

  {
    std::vector<std::string> names;
    for (const auto& e : diffErrorFunctions_) {
      if (e) {
        names.push_back(e->name());
      } else {
        names.push_back("Unknown");
      }
    }

    std::ostringstream name;
    name << "Union[";
    if (std::adjacent_find(std::begin(names), std::end(names), std::not_equal_to<>()) ==
        std::end(names)) {
      // All the same:
      name << names.front() << " x " << names.size();
    } else {
      bool first = true;
      for (const auto& n : names) {
        if (!first) {
          name << ", ";
        }
        name << n;
        first = false;
      }
    }
    name << "]";
    name_ = name.str();
  }
}

template <typename T>
double UnionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  double error = 0;
  for (auto& fun : errorFunctions_) {
    error += fun->getError(params, state, meshState);
  }
  return this->weight_ * error;
}

template <typename T>
double UnionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  double error = 0;
  for (auto& fun : errorFunctions_) {
    error += fun->getGradient(params, state, meshState, gradient);
  }
  gradient *= this->weight_;
  return this->weight_ * error;
}

template <typename T>
double UnionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  double error = 0;
  size_t jacobianStart = 0;
  for (auto& fun : errorFunctions_) {
    const size_t memberJacobianSize = fun->getJacobianSize();

    error += fun->getJacobian(
        params,
        state,
        meshState,
        jacobian.block(jacobianStart, 0, memberJacobianSize, jacobian.cols()),
        residual.segment(jacobianStart, memberJacobianSize),
        usedRows);

    jacobianStart += memberJacobianSize;
  }

  jacobian *= std::sqrt(this->weight_);
  residual *= std::sqrt(this->weight_);

  usedRows = static_cast<int>(jacobian.rows());

  return this->weight_ * error;
}

template <typename T>
size_t UnionErrorFunctionT<T>::getJacobianSize() const {
  size_t result = 0;
  for (const auto& e : errorFunctions_) {
    result += e->getJacobianSize();
  }
  return result;
}

template <typename T>
std::vector<std::string> UnionErrorFunctionT<T>::inputs() const {
  std::vector<std::string> result;
  for (const auto& i : inputs_) {
    result.push_back(i.first);
  }
  return result;
}

template <typename T>
Eigen::Index UnionErrorFunctionT<T>::getInputSize(const std::string& name) const {
  auto itr = inputs_.find(name);
  MT_THROW_IF(
      itr == inputs_.end(), "Invalid input '{}' for UnionErrorFunction::getInputSize", name);

  Eigen::Index result = 0;
  for (const auto& iErrFun : itr->second) {
    const auto errorFun = diffErrorFunctions_[iErrFun];
    result += errorFun->getInputSize(name);
  }
  return result;
}

template <typename T>
void UnionErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> result) const {
  auto itr = inputs_.find(name);
  MT_THROW_IF(itr == inputs_.end(), "Invalid input '{}' for UnionErrorFunction::getInput", name);

  Eigen::Index currentOffset = 0;
  for (const auto& iErrFun : itr->second) {
    const auto errorFun = diffErrorFunctions_[iErrFun];
    const Eigen::Index sz = errorFun->getInputSize(name);
    errorFun->getInput(name, result.segment(currentOffset, sz));
    currentOffset += sz;
  }
}

template <typename T>
void UnionErrorFunctionT<T>::setInputImp(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  auto itr = inputs_.find(name);
  MT_THROW_IF(itr == inputs_.end(), "Invalid input '{}' for UnionErrorFunction::setInput", name);

  Eigen::Index currentOffset = 0;
  for (const auto& iErrFun : itr->second) {
    const auto errorFun = diffErrorFunctions_[iErrFun];
    const Eigen::Index sz = errorFun->getInputSize(name);
    errorFun->setInput(name, value.segment(currentOffset, sz));
    currentOffset += sz;
  }
}

template <typename T>
Eigen::VectorX<T> UnionErrorFunctionT<T>::d_gradient_d_input_dot(
    const std::string& name,
    const ModelParametersT<T>& modelParams,
    const SkeletonStateT<T>& state,
    Eigen::Ref<const Eigen::VectorX<T>> inputVec) {
  auto itr = inputs_.find(name);
  MT_THROW_IF(
      itr == inputs_.end(),
      "Invalid input '{}' for UnionErrorFunction::d_gradient_d_input_dot",
      name);

  Eigen::Index currentOffset = 0;
  Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(getInputSize(name));

  for (const auto& iErrFun : itr->second) {
    const auto errFun = diffErrorFunctions_[iErrFun];
    const Eigen::Index sz = errFun->getInputSize(name);
    result.segment(currentOffset, sz) =
        errFun->d_gradient_d_input_dot(name, modelParams, state, inputVec);
    currentOffset += sz;
  }

  result *= this->weight_;

  return result;
}

template class UnionErrorFunctionT<float>;
template class UnionErrorFunctionT<double>;

} // namespace momentum
