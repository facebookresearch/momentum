/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/multi_source_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/math/constants.h"

#include <dispenso/parallel_for.h>

#include <array>
#include <cstdint>
#include <numeric>

namespace momentum {

// SourceT methods
template <typename T>
size_t SourceT<T>::getJointIndex() const {
  return std::visit(
      [](const auto& v) -> size_t {
        using Type = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<Type, JointPoint> || std::is_same_v<Type, JointDirection>) {
          return v.jointIndex;
        } else {
          return kInvalidIndex;
        }
      },
      data);
}

template <typename T>
bool SourceT<T>::isPoint() const {
  return std::holds_alternative<JointPoint>(data);
}

template <typename T>
bool SourceT<T>::isVertex() const {
  return std::holds_alternative<Vertex>(data);
}

template <typename T>
bool SourceT<T>::isVertexNormal() const {
  return std::holds_alternative<VertexNormal>(data);
}

template <typename T>
SourceT<T> SourceT<T>::jointPoint(size_t jointIndex, const Eigen::Vector3<T>& offset) {
  SourceT<T> result;
  result.data = JointPoint{jointIndex, offset};
  return result;
}

template <typename T>
SourceT<T> SourceT<T>::jointDirection(size_t jointIndex, const Eigen::Vector3<T>& direction) {
  SourceT<T> result;
  result.data = JointDirection{jointIndex, direction};
  return result;
}

template <typename T>
SourceT<T> SourceT<T>::vertex(size_t vertexIndex) {
  SourceT<T> result;
  result.data = Vertex{vertexIndex};
  return result;
}

template <typename T>
SourceT<T> SourceT<T>::vertexNormal(
    size_t vertexIndex,
    const Eigen::Vector3<T>& targetPosition,
    T sourceNormalWeight) {
  SourceT<T> result;
  result.data = VertexNormal{vertexIndex, targetPosition, sourceNormalWeight};
  return result;
}

// Explicit instantiations for SourceT
template struct SourceT<float>;
template struct SourceT<double>;

// MultiSourceErrorFunctionT methods
template <typename T, size_t FuncDim>
MultiSourceErrorFunctionT<T, FuncDim>::MultiSourceErrorFunctionT(
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    const T& lossAlpha,
    const T& lossC)
    : SkeletonErrorFunctionT<T>(skeleton, parameterTransform),
      skeletonDerivative_(
          skeleton,
          parameterTransform,
          this->activeJointParams_,
          this->enabledParameters_),
      loss_(lossAlpha, lossC),
      jointGrad_(parameterTransform.numAllModelParameters()) {}

template <typename T, size_t FuncDim>
bool MultiSourceErrorFunctionT<T, FuncDim>::needsMesh() const {
  if (!needsMeshCacheValid_) {
    needsMeshCached_ = false;
    const size_t numConstraints = getNumConstraints();
    for (size_t i = 0; i < numConstraints && !needsMeshCached_; ++i) {
      auto contributions = getContributions(i);
      for (const auto& contrib : contributions) {
        if (contrib.isVertex() || contrib.isVertexNormal()) {
          needsMeshCached_ = true;
          break;
        }
      }
    }
    needsMeshCacheValid_ = true;
  }
  return needsMeshCached_;
}

template <typename T, size_t FuncDim>
size_t MultiSourceErrorFunctionT<T, FuncDim>::getJacobianSize() const {
  return getNumConstraints() * FuncDim;
}

template <typename T, size_t FuncDim>
void MultiSourceErrorFunctionT<T, FuncDim>::computeWorldVectors(
    size_t constrIndex,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    std::vector<Eigen::Vector3<T>>& worldVecs) const {
  auto contributions = getContributions(constrIndex);
  worldVecs.resize(contributions.size());

  for (size_t i = 0; i < contributions.size(); ++i) {
    const auto& contrib = contributions[i];
    std::visit(
        [&](const auto& v) {
          using Type = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<Type, typename SourceT<T>::JointPoint>) {
            const auto& jointState = state.jointState[v.jointIndex];
            worldVecs[i] = jointState.transform * v.localOffset;
          } else if constexpr (std::is_same_v<Type, typename SourceT<T>::JointDirection>) {
            const auto& jointState = state.jointState[v.jointIndex];
            worldVecs[i] = jointState.rotation() * v.localDirection;
          } else if constexpr (std::is_same_v<Type, typename SourceT<T>::Vertex>) {
            MT_CHECK(
                meshState.posedMesh_ != nullptr, "Mesh state is required for vertex contributions");
            MT_CHECK(
                v.vertexIndex < meshState.posedMesh_->vertices.size(),
                "Vertex index {} out of range (mesh has {} vertices)",
                v.vertexIndex,
                meshState.posedMesh_->vertices.size());
            worldVecs[i] = meshState.posedMesh_->vertices[v.vertexIndex].template cast<T>();
          } else if constexpr (std::is_same_v<Type, typename SourceT<T>::VertexNormal>) {
            MT_CHECK(
                meshState.posedMesh_ != nullptr,
                "Mesh state is required for vertex normal contributions");
            MT_CHECK(
                v.vertexIndex < meshState.posedMesh_->vertices.size(),
                "Vertex index {} out of range (mesh has {} vertices)",
                v.vertexIndex,
                meshState.posedMesh_->vertices.size());
            worldVecs[i] = meshState.posedMesh_->vertices[v.vertexIndex].template cast<T>();
          }
        },
        contrib.data);
  }
}

template <typename T, size_t FuncDim>
void MultiSourceErrorFunctionT<T, FuncDim>::accumulateGradient(
    size_t constrIndex,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    std::span<const Eigen::Vector3<T>> worldVecs,
    std::span<const DfdvType> dfdvs,
    const FuncType& weightedResidual,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  auto contributions = getContributions(constrIndex);
  MT_CHECK(contributions.size() == worldVecs.size());
  MT_CHECK(contributions.size() == dfdvs.size());

  const size_t numContribs = contributions.size();

  // Fast path for single-contribution constraints (most common: vertex constraints)
  if (numContribs == 1) {
    const auto& contrib = contributions[0];
    const size_t jointIdx = contrib.getJointIndex();
    if (jointIdx != kInvalidIndex) {
      const uint8_t isPoint = contrib.isPoint() ? 1 : 0;
      skeletonDerivative_.template accumulateJointGradient<FuncDim>(
          jointIdx,
          std::span<const Eigen::Vector3<T>>(&worldVecs[0], 1),
          std::span<const DfdvType>(&dfdvs[0], 1),
          std::span<const uint8_t>(&isPoint, 1),
          weightedResidual,
          state.jointState,
          gradient);
    } else if (contrib.isVertex()) {
      const auto& vertexData = std::get<typename SourceT<T>::Vertex>(contrib.data);
      const Character* character = getCharacter();
      MT_CHECK(character != nullptr, "Character required for vertex contributions");
      skeletonDerivative_.template accumulateVertexGradient<FuncDim>(
          vertexData.vertexIndex,
          worldVecs[0],
          dfdvs[0],
          weightedResidual,
          state,
          meshState,
          *character,
          gradient);
    } else if (contrib.isVertexNormal()) {
      const auto& vnData = std::get<typename SourceT<T>::VertexNormal>(contrib.data);
      const Character* character = getCharacter();
      MT_CHECK(character != nullptr, "Character required for vertex normal contributions");
      skeletonDerivative_.template accumulateVertexGradient<FuncDim>(
          vnData.vertexIndex,
          worldVecs[0],
          dfdvs[0],
          weightedResidual,
          state,
          meshState,
          *character,
          gradient);
      if (vnData.sourceNormalWeight != T(0)) {
        const Eigen::Vector3<T> sourceNormal =
            meshState.posedMesh_->normals[vnData.vertexIndex].template cast<T>();
        const T normalScale = weightedResidual[0] * vnData.sourceNormalWeight;
        skeletonDerivative_.accumulateVertexNormalCorrectionGradient(
            vnData.vertexIndex,
            sourceNormal,
            vnData.targetPosition,
            normalScale,
            state,
            meshState,
            *character,
            gradient);
        const Eigen::Vector3<T> leverArm = worldVecs[0] - vnData.targetPosition;
        skeletonDerivative_.accumulateVertexNormalBlendShapeGradient(
            vnData.vertexIndex, leverArm, normalScale, state, meshState, *character, gradient);
      }
    }
    return;
  }

  // General path for multi-contribution constraints
  // Group contributions by joint index and process each group in a single walk.
  // Use flat stack-allocated arrays to avoid heap allocation (max ~16 contributions).
  constexpr size_t kMaxContribs = 16;
  MT_CHECK(
      numContribs <= kMaxContribs, "Too many contributions: {} > {}", numContribs, kMaxContribs);

  // First pass: collect unique joint indices and map contributions
  std::array<size_t, kMaxContribs> jointIds{};
  std::array<size_t, kMaxContribs> contribJointSlot{}; // which group each contrib belongs to
  size_t numUniqueJoints = 0;

  for (size_t i = 0; i < numContribs; ++i) {
    const size_t jointIdx = contributions[i].getJointIndex();
    if (jointIdx == kInvalidIndex) {
      contribJointSlot[i] = kInvalidIndex;
      continue;
    }
    // Find or add this joint
    size_t slot = kInvalidIndex;
    for (size_t s = 0; s < numUniqueJoints; ++s) {
      if (jointIds[s] == jointIdx) {
        slot = s;
        break;
      }
    }
    if (slot == kInvalidIndex) {
      slot = numUniqueJoints++;
      jointIds[slot] = jointIdx;
    }
    contribJointSlot[i] = slot;
  }

  // Second pass: for each unique joint, collect and batch its contributions
  std::array<Eigen::Vector3<T>, kMaxContribs> batchedWorldVecs;
  std::array<DfdvType, kMaxContribs> batchedDfdvs;
  std::array<uint8_t, kMaxContribs> batchedIsPoints{};

  for (size_t s = 0; s < numUniqueJoints; ++s) {
    size_t count = 0;
    for (size_t i = 0; i < numContribs; ++i) {
      if (contribJointSlot[i] == s) {
        batchedWorldVecs[count] = worldVecs[i];
        batchedDfdvs[count] = dfdvs[i];
        batchedIsPoints[count] = contributions[i].isPoint() ? 1 : 0;
        ++count;
      }
    }
    skeletonDerivative_.template accumulateJointGradient<FuncDim>(
        jointIds[s],
        std::span<const Eigen::Vector3<T>>(batchedWorldVecs.data(), count),
        std::span<const DfdvType>(batchedDfdvs.data(), count),
        std::span<const uint8_t>(batchedIsPoints.data(), count),
        weightedResidual,
        state.jointState,
        gradient);
  }

  // Handle vertex contributions via SkeletonDerivativeT
  const Character* character = getCharacter();
  for (size_t i = 0; i < numContribs; ++i) {
    if (contributions[i].isVertex()) {
      const auto& vertexData = std::get<typename SourceT<T>::Vertex>(contributions[i].data);
      MT_CHECK(character != nullptr, "Character required for vertex contributions");
      skeletonDerivative_.template accumulateVertexGradient<FuncDim>(
          vertexData.vertexIndex,
          worldVecs[i],
          dfdvs[i],
          weightedResidual,
          state,
          meshState,
          *character,
          gradient);
    } else if (contributions[i].isVertexNormal()) {
      const auto& vnData = std::get<typename SourceT<T>::VertexNormal>(contributions[i].data);
      MT_CHECK(character != nullptr, "Character required for vertex normal contributions");
      skeletonDerivative_.template accumulateVertexGradient<FuncDim>(
          vnData.vertexIndex,
          worldVecs[i],
          dfdvs[i],
          weightedResidual,
          state,
          meshState,
          *character,
          gradient);
      if (vnData.sourceNormalWeight != T(0)) {
        const Eigen::Vector3<T> sourceNormal =
            meshState.posedMesh_->normals[vnData.vertexIndex].template cast<T>();
        const T normalScale = weightedResidual[0] * vnData.sourceNormalWeight;
        skeletonDerivative_.accumulateVertexNormalCorrectionGradient(
            vnData.vertexIndex,
            sourceNormal,
            vnData.targetPosition,
            normalScale,
            state,
            meshState,
            *character,
            gradient);
        const Eigen::Vector3<T> leverArm = worldVecs[i] - vnData.targetPosition;
        skeletonDerivative_.accumulateVertexNormalBlendShapeGradient(
            vnData.vertexIndex, leverArm, normalScale, state, meshState, *character, gradient);
      }
    }
  }
}

template <typename T, size_t FuncDim>
void MultiSourceErrorFunctionT<T, FuncDim>::accumulateJacobian(
    size_t constrIndex,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    std::span<const Eigen::Vector3<T>> worldVecs,
    std::span<const DfdvType> dfdvs,
    T scale,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  auto contributions = getContributions(constrIndex);
  MT_CHECK(contributions.size() == worldVecs.size());
  MT_CHECK(contributions.size() == dfdvs.size());

  const size_t numContribs = contributions.size();

  // Fast path for single-contribution constraints (most common: vertex constraints)
  if (numContribs == 1) {
    const auto& contrib = contributions[0];
    const size_t jointIdx = contrib.getJointIndex();
    if (jointIdx != kInvalidIndex) {
      const uint8_t isPoint = contrib.isPoint() ? 1 : 0;
      skeletonDerivative_.template accumulateJointJacobian<FuncDim>(
          jointIdx,
          std::span<const Eigen::Vector3<T>>(&worldVecs[0], 1),
          std::span<const DfdvType>(&dfdvs[0], 1),
          std::span<const uint8_t>(&isPoint, 1),
          scale,
          state.jointState,
          jacobian,
          rowIndex);
    } else if (contrib.isVertex()) {
      const auto& vertexData = std::get<typename SourceT<T>::Vertex>(contrib.data);
      const Character* character = getCharacter();
      MT_CHECK(character != nullptr, "Character required for vertex contributions");
      skeletonDerivative_.template accumulateVertexJacobian<FuncDim>(
          vertexData.vertexIndex,
          worldVecs[0],
          dfdvs[0],
          scale,
          state,
          meshState,
          *character,
          jacobian,
          rowIndex);
    } else if (contrib.isVertexNormal()) {
      const auto& vnData = std::get<typename SourceT<T>::VertexNormal>(contrib.data);
      const Character* character = getCharacter();
      MT_CHECK(character != nullptr, "Character required for vertex normal contributions");
      skeletonDerivative_.template accumulateVertexJacobian<FuncDim>(
          vnData.vertexIndex,
          worldVecs[0],
          dfdvs[0],
          scale,
          state,
          meshState,
          *character,
          jacobian,
          rowIndex);
      if (vnData.sourceNormalWeight != T(0)) {
        const Eigen::Vector3<T> sourceNormal =
            meshState.posedMesh_->normals[vnData.vertexIndex].template cast<T>();
        const T normalScale = scale * vnData.sourceNormalWeight;
        skeletonDerivative_.accumulateVertexNormalCorrectionJacobian(
            vnData.vertexIndex,
            sourceNormal,
            vnData.targetPosition,
            normalScale,
            state,
            meshState,
            *character,
            jacobian,
            rowIndex);
        const Eigen::Vector3<T> leverArm = worldVecs[0] - vnData.targetPosition;
        skeletonDerivative_.accumulateVertexNormalBlendShapeJacobian(
            vnData.vertexIndex,
            leverArm,
            normalScale,
            state,
            meshState,
            *character,
            jacobian,
            rowIndex);
      }
    }
    return;
  }

  // General path for multi-contribution constraints
  // Group contributions by joint index and process each group in a single walk.
  // Use flat stack-allocated arrays to avoid heap allocation (max ~16 contributions).
  constexpr size_t kMaxContribs = 16;
  MT_CHECK(
      numContribs <= kMaxContribs, "Too many contributions: {} > {}", numContribs, kMaxContribs);

  // First pass: collect unique joint indices and map contributions
  std::array<size_t, kMaxContribs> jointIds{};
  std::array<size_t, kMaxContribs> contribJointSlot{};
  size_t numUniqueJoints = 0;

  for (size_t i = 0; i < numContribs; ++i) {
    const size_t jointIdx = contributions[i].getJointIndex();
    if (jointIdx == kInvalidIndex) {
      contribJointSlot[i] = kInvalidIndex;
      continue;
    }
    size_t slot = kInvalidIndex;
    for (size_t s = 0; s < numUniqueJoints; ++s) {
      if (jointIds[s] == jointIdx) {
        slot = s;
        break;
      }
    }
    if (slot == kInvalidIndex) {
      slot = numUniqueJoints++;
      jointIds[slot] = jointIdx;
    }
    contribJointSlot[i] = slot;
  }

  // Second pass: for each unique joint, collect and batch its contributions
  std::array<Eigen::Vector3<T>, kMaxContribs> batchedWorldVecs;
  std::array<DfdvType, kMaxContribs> batchedDfdvs;
  std::array<uint8_t, kMaxContribs> batchedIsPoints{};

  for (size_t s = 0; s < numUniqueJoints; ++s) {
    size_t count = 0;
    for (size_t i = 0; i < numContribs; ++i) {
      if (contribJointSlot[i] == s) {
        batchedWorldVecs[count] = worldVecs[i];
        batchedDfdvs[count] = dfdvs[i];
        batchedIsPoints[count] = contributions[i].isPoint() ? 1 : 0;
        ++count;
      }
    }
    skeletonDerivative_.template accumulateJointJacobian<FuncDim>(
        jointIds[s],
        std::span<const Eigen::Vector3<T>>(batchedWorldVecs.data(), count),
        std::span<const DfdvType>(batchedDfdvs.data(), count),
        std::span<const uint8_t>(batchedIsPoints.data(), count),
        scale,
        state.jointState,
        jacobian,
        rowIndex);
  }

  // Handle vertex contributions via SkeletonDerivativeT
  const Character* character = getCharacter();
  for (size_t i = 0; i < numContribs; ++i) {
    if (contributions[i].isVertex()) {
      const auto& vertexData = std::get<typename SourceT<T>::Vertex>(contributions[i].data);
      MT_CHECK(character != nullptr, "Character required for vertex contributions");
      skeletonDerivative_.template accumulateVertexJacobian<FuncDim>(
          vertexData.vertexIndex,
          worldVecs[i],
          dfdvs[i],
          scale,
          state,
          meshState,
          *character,
          jacobian,
          rowIndex);
    } else if (contributions[i].isVertexNormal()) {
      const auto& vnData = std::get<typename SourceT<T>::VertexNormal>(contributions[i].data);
      MT_CHECK(character != nullptr, "Character required for vertex normal contributions");
      skeletonDerivative_.template accumulateVertexJacobian<FuncDim>(
          vnData.vertexIndex,
          worldVecs[i],
          dfdvs[i],
          scale,
          state,
          meshState,
          *character,
          jacobian,
          rowIndex);
      if (vnData.sourceNormalWeight != T(0)) {
        const Eigen::Vector3<T> sourceNormal =
            meshState.posedMesh_->normals[vnData.vertexIndex].template cast<T>();
        const T normalScale = scale * vnData.sourceNormalWeight;
        skeletonDerivative_.accumulateVertexNormalCorrectionJacobian(
            vnData.vertexIndex,
            sourceNormal,
            vnData.targetPosition,
            normalScale,
            state,
            meshState,
            *character,
            jacobian,
            rowIndex);
        const Eigen::Vector3<T> leverArm = worldVecs[i] - vnData.targetPosition;
        skeletonDerivative_.accumulateVertexNormalBlendShapeJacobian(
            vnData.vertexIndex,
            leverArm,
            normalScale,
            state,
            meshState,
            *character,
            jacobian,
            rowIndex);
      }
    }
  }
}

template <typename T, size_t FuncDim>
double MultiSourceErrorFunctionT<T, FuncDim>::getError(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  const size_t numConstraints = getNumConstraints();
  if (numConstraints == 0) {
    return 0.0;
  }

  double totalError = 0.0;
  std::vector<Eigen::Vector3<T>> worldVecs;
  FuncType f;

  for (size_t i = 0; i < numConstraints; ++i) {
    const T constrWeight = getConstraintWeight(i);
    if (constrWeight == T(0)) {
      continue;
    }

    computeWorldVectors(i, state, meshState, worldVecs);

    evalFunction(i, state, meshState, std::span<const Eigen::Vector3<T>>(worldVecs), f, {});

    const T w = constrWeight * this->weight_;
    totalError += w * loss_.value(f.squaredNorm());
  }

  return totalError;
}

// Enum to tag the contribution type determined during std::visit, avoiding re-dispatch later.
enum class ContribKind : uint8_t { JointPoint, JointDirection, Vertex, VertexNormal };

template <typename T, size_t FuncDim>
double MultiSourceErrorFunctionT<T, FuncDim>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  const size_t numConstraints = getNumConstraints();
  if (numConstraints == 0) {
    return 0.0;
  }

  // Cache getCharacter() outside the hot loop — one virtual call total.
  const Character* character = getCharacter();

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;

  std::vector<std::tuple<double, VectorX<T>>> errorGradThread;
  dispenso::parallel_for(
      errorGradThread,
      [&]() -> std::tuple<double, VectorX<T>> {
        return {0.0, VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters())};
      },
      size_t(0),
      numConstraints,
      [&](std::tuple<double, VectorX<T>>& errorGradLocal, const size_t i) {
        double& errorLocal = std::get<0>(errorGradLocal);
        auto& gradLocal = std::get<1>(errorGradLocal);

        const T constrWeight = getConstraintWeight(i);
        if (constrWeight == T(0)) {
          return;
        }

        auto contributions = getContributions(i);
        const size_t numContribs = contributions.size();

        FuncType f;

        if (numContribs == 1) {
          // Fast path: stack-allocate for single-contribution constraints.
          // We compute the world vector AND remember the contribution kind
          // in a single std::visit, then use the kind directly for accumulation,
          // eliminating the second getContributions() call and type re-checks.
          Eigen::Vector3<T> worldVec;
          DfdvType dfdv;
          dfdv.setZero();

          const auto& contrib = contributions[0];
          ContribKind kind{};
          size_t vertexIndex = 0;

          std::visit(
              [&](const auto& v) {
                using Type = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<Type, typename SourceT<T>::JointPoint>) {
                  worldVec = state.jointState[v.jointIndex].transform * v.localOffset;
                  kind = ContribKind::JointPoint;
                } else if constexpr (std::is_same_v<Type, typename SourceT<T>::JointDirection>) {
                  worldVec = state.jointState[v.jointIndex].rotation() * v.localDirection;
                  kind = ContribKind::JointDirection;
                } else if constexpr (std::is_same_v<Type, typename SourceT<T>::Vertex>) {
                  worldVec = meshState.posedMesh_->vertices[v.vertexIndex].template cast<T>();
                  kind = ContribKind::Vertex;
                  vertexIndex = v.vertexIndex;
                } else if constexpr (std::is_same_v<Type, typename SourceT<T>::VertexNormal>) {
                  worldVec = meshState.posedMesh_->vertices[v.vertexIndex].template cast<T>();
                  kind = ContribKind::VertexNormal;
                  vertexIndex = v.vertexIndex;
                }
              },
              contrib.data);

          evalFunction(
              i,
              state,
              meshState,
              std::span<const Eigen::Vector3<T>>(&worldVec, 1),
              f,
              std::span<DfdvType>(&dfdv, 1));

          if (f.isZero()) {
            return;
          }

          const T sqrError = f.squaredNorm();
          const T w = constrWeight * this->weight_;
          errorLocal += w * loss_.value(sqrError);

          const FuncType weightedResidual = T(2) * w * loss_.deriv(sqrError) * f;

          // Inline accumulation using the remembered kind — no second
          // getContributions() call, no variant re-dispatch, no type re-checks.
          switch (kind) {
            case ContribKind::JointPoint: {
              const uint8_t isPoint = 1;
              skeletonDerivative_.template accumulateJointGradient<FuncDim>(
                  std::get<typename SourceT<T>::JointPoint>(contrib.data).jointIndex,
                  std::span<const Eigen::Vector3<T>>(&worldVec, 1),
                  std::span<const DfdvType>(&dfdv, 1),
                  std::span<const uint8_t>(&isPoint, 1),
                  weightedResidual,
                  state.jointState,
                  gradLocal);
              break;
            }
            case ContribKind::JointDirection: {
              const uint8_t isPoint = 0;
              skeletonDerivative_.template accumulateJointGradient<FuncDim>(
                  std::get<typename SourceT<T>::JointDirection>(contrib.data).jointIndex,
                  std::span<const Eigen::Vector3<T>>(&worldVec, 1),
                  std::span<const DfdvType>(&dfdv, 1),
                  std::span<const uint8_t>(&isPoint, 1),
                  weightedResidual,
                  state.jointState,
                  gradLocal);
              break;
            }
            case ContribKind::Vertex: {
              MT_CHECK(character != nullptr, "Character required for vertex contributions");
              skeletonDerivative_.template accumulateVertexGradient<FuncDim>(
                  vertexIndex,
                  worldVec,
                  dfdv,
                  weightedResidual,
                  state,
                  meshState,
                  *character,
                  gradLocal);
              break;
            }
            case ContribKind::VertexNormal: {
              MT_CHECK(character != nullptr, "Character required for vertex normal contributions");
              skeletonDerivative_.template accumulateVertexGradient<FuncDim>(
                  vertexIndex,
                  worldVec,
                  dfdv,
                  weightedResidual,
                  state,
                  meshState,
                  *character,
                  gradLocal);
              const auto& vnData = std::get<typename SourceT<T>::VertexNormal>(contrib.data);
              if (vnData.sourceNormalWeight != T(0)) {
                const Eigen::Vector3<T> sourceNormal =
                    meshState.posedMesh_->normals[vnData.vertexIndex].template cast<T>();
                const T normalScale = weightedResidual[0] * vnData.sourceNormalWeight;
                skeletonDerivative_.accumulateVertexNormalCorrectionGradient(
                    vnData.vertexIndex,
                    sourceNormal,
                    vnData.targetPosition,
                    normalScale,
                    state,
                    meshState,
                    *character,
                    gradLocal);
                const Eigen::Vector3<T> leverArm = worldVec - vnData.targetPosition;
                skeletonDerivative_.accumulateVertexNormalBlendShapeGradient(
                    vnData.vertexIndex,
                    leverArm,
                    normalScale,
                    state,
                    meshState,
                    *character,
                    gradLocal);
              }
              break;
            }
          }
        } else {
          // General path: heap-allocate for multi-contribution constraints
          std::vector<Eigen::Vector3<T>> worldVecs;
          computeWorldVectors(i, state, meshState, worldVecs);

          std::vector<DfdvType> dfdvs(numContribs);
          for (auto& dfdv : dfdvs) {
            dfdv.setZero();
          }

          evalFunction(
              i, state, meshState, std::span<const Eigen::Vector3<T>>(worldVecs), f, dfdvs);

          if (f.isZero()) {
            return;
          }

          const T sqrError = f.squaredNorm();
          const T w = constrWeight * this->weight_;
          errorLocal += w * loss_.value(sqrError);

          const FuncType weightedResidual = T(2) * w * loss_.deriv(sqrError) * f;

          accumulateGradient(
              i,
              state,
              meshState,
              std::span<const Eigen::Vector3<T>>(worldVecs),
              std::span<const DfdvType>(dfdvs),
              weightedResidual,
              gradLocal);
        }
      },
      dispensoOptions);

  if (!errorGradThread.empty()) {
    errorGradThread[0] = std::accumulate(
        errorGradThread.begin() + 1,
        errorGradThread.end(),
        errorGradThread[0],
        [](const auto& a, const auto& b) -> std::tuple<double, VectorX<T>> {
          return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};
        });

    gradient += std::get<1>(errorGradThread[0]);
    return std::get<0>(errorGradThread[0]);
  }

  return 0.0;
}

template <typename T, size_t FuncDim>
double MultiSourceErrorFunctionT<T, FuncDim>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  const size_t numConstraints = getNumConstraints();
  usedRows = 0;
  if (numConstraints == 0) {
    return 0.0;
  }

  // Cache getCharacter() outside the hot loop — one virtual call total.
  const Character* character = getCharacter();

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;

  std::vector<double> errorThread;
  dispenso::parallel_for(
      errorThread,
      [&]() -> double { return 0.0; },
      size_t(0),
      numConstraints,
      [&](double& errorLocal, const size_t i) {
        const T constrWeight = getConstraintWeight(i);
        if (constrWeight == T(0)) {
          return;
        }

        auto contributions = getContributions(i);
        const size_t numContribs = contributions.size();
        const size_t rowIndex = i * FuncDim;

        FuncType f;

        if (numContribs == 1) {
          // Fast path: stack-allocate for single-contribution constraints.
          // Same optimization as getGradient: single std::visit determines kind,
          // then we accumulate directly without calling accumulateJacobian().
          Eigen::Vector3<T> worldVec;
          DfdvType dfdv;
          dfdv.setZero();

          const auto& contrib = contributions[0];
          ContribKind kind{};
          size_t vertexIndex = 0;

          std::visit(
              [&](const auto& v) {
                using Type = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<Type, typename SourceT<T>::JointPoint>) {
                  worldVec = state.jointState[v.jointIndex].transform * v.localOffset;
                  kind = ContribKind::JointPoint;
                } else if constexpr (std::is_same_v<Type, typename SourceT<T>::JointDirection>) {
                  worldVec = state.jointState[v.jointIndex].rotation() * v.localDirection;
                  kind = ContribKind::JointDirection;
                } else if constexpr (std::is_same_v<Type, typename SourceT<T>::Vertex>) {
                  worldVec = meshState.posedMesh_->vertices[v.vertexIndex].template cast<T>();
                  kind = ContribKind::Vertex;
                  vertexIndex = v.vertexIndex;
                } else if constexpr (std::is_same_v<Type, typename SourceT<T>::VertexNormal>) {
                  worldVec = meshState.posedMesh_->vertices[v.vertexIndex].template cast<T>();
                  kind = ContribKind::VertexNormal;
                  vertexIndex = v.vertexIndex;
                }
              },
              contrib.data);

          evalFunction(
              i,
              state,
              meshState,
              std::span<const Eigen::Vector3<T>>(&worldVec, 1),
              f,
              std::span<DfdvType>(&dfdv, 1));

          const T sqrError = f.squaredNorm();
          const T w = constrWeight * this->weight_;
          errorLocal += w * loss_.value(sqrError);

          const T deriv = std::sqrt(w * loss_.deriv(sqrError));

          residual.template segment<FuncDim>(rowIndex).noalias() = deriv * f;

          if (deriv == T(0)) {
            return;
          }

          // Inline accumulation using the remembered kind.
          switch (kind) {
            case ContribKind::JointPoint: {
              const uint8_t isPoint = 1;
              skeletonDerivative_.template accumulateJointJacobian<FuncDim>(
                  std::get<typename SourceT<T>::JointPoint>(contrib.data).jointIndex,
                  std::span<const Eigen::Vector3<T>>(&worldVec, 1),
                  std::span<const DfdvType>(&dfdv, 1),
                  std::span<const uint8_t>(&isPoint, 1),
                  deriv,
                  state.jointState,
                  jacobian,
                  rowIndex);
              break;
            }
            case ContribKind::JointDirection: {
              const uint8_t isPoint = 0;
              skeletonDerivative_.template accumulateJointJacobian<FuncDim>(
                  std::get<typename SourceT<T>::JointDirection>(contrib.data).jointIndex,
                  std::span<const Eigen::Vector3<T>>(&worldVec, 1),
                  std::span<const DfdvType>(&dfdv, 1),
                  std::span<const uint8_t>(&isPoint, 1),
                  deriv,
                  state.jointState,
                  jacobian,
                  rowIndex);
              break;
            }
            case ContribKind::Vertex: {
              MT_CHECK(character != nullptr, "Character required for vertex contributions");
              skeletonDerivative_.template accumulateVertexJacobian<FuncDim>(
                  vertexIndex,
                  worldVec,
                  dfdv,
                  deriv,
                  state,
                  meshState,
                  *character,
                  jacobian,
                  rowIndex);
              break;
            }
            case ContribKind::VertexNormal: {
              MT_CHECK(character != nullptr, "Character required for vertex normal contributions");
              skeletonDerivative_.template accumulateVertexJacobian<FuncDim>(
                  vertexIndex,
                  worldVec,
                  dfdv,
                  deriv,
                  state,
                  meshState,
                  *character,
                  jacobian,
                  rowIndex);
              const auto& vnData = std::get<typename SourceT<T>::VertexNormal>(contrib.data);
              if (vnData.sourceNormalWeight != T(0)) {
                const Eigen::Vector3<T> sourceNormal =
                    meshState.posedMesh_->normals[vnData.vertexIndex].template cast<T>();
                const T normalScale = deriv * vnData.sourceNormalWeight;
                skeletonDerivative_.accumulateVertexNormalCorrectionJacobian(
                    vnData.vertexIndex,
                    sourceNormal,
                    vnData.targetPosition,
                    normalScale,
                    state,
                    meshState,
                    *character,
                    jacobian,
                    rowIndex);
                const Eigen::Vector3<T> leverArm = worldVec - vnData.targetPosition;
                skeletonDerivative_.accumulateVertexNormalBlendShapeJacobian(
                    vnData.vertexIndex,
                    leverArm,
                    normalScale,
                    state,
                    meshState,
                    *character,
                    jacobian,
                    rowIndex);
              }
              break;
            }
          }
        } else {
          // General path: heap-allocate for multi-contribution constraints
          std::vector<Eigen::Vector3<T>> worldVecs;
          computeWorldVectors(i, state, meshState, worldVecs);

          std::vector<DfdvType> dfdvs(numContribs);
          for (auto& dfdv : dfdvs) {
            dfdv.setZero();
          }

          evalFunction(
              i, state, meshState, std::span<const Eigen::Vector3<T>>(worldVecs), f, dfdvs);

          const T sqrError = f.squaredNorm();
          const T w = constrWeight * this->weight_;
          errorLocal += w * loss_.value(sqrError);

          const T deriv = std::sqrt(w * loss_.deriv(sqrError));

          residual.template segment<FuncDim>(rowIndex).noalias() = deriv * f;

          if (deriv == T(0)) {
            return;
          }

          accumulateJacobian(
              i,
              state,
              meshState,
              std::span<const Eigen::Vector3<T>>(worldVecs),
              std::span<const DfdvType>(dfdvs),
              deriv,
              jacobian,
              rowIndex);
        }
      },
      dispensoOptions);

  usedRows = static_cast<int>(numConstraints * FuncDim);

  if (!errorThread.empty()) {
    return std::accumulate(errorThread.begin() + 1, errorThread.end(), errorThread[0]);
  }

  return 0.0;
}

// Explicit template instantiations
template class MultiSourceErrorFunctionT<float, 1>;
template class MultiSourceErrorFunctionT<float, 2>;
template class MultiSourceErrorFunctionT<float, 3>;
template class MultiSourceErrorFunctionT<float, 9>;

template class MultiSourceErrorFunctionT<double, 1>;
template class MultiSourceErrorFunctionT<double, 2>;
template class MultiSourceErrorFunctionT<double, 3>;
template class MultiSourceErrorFunctionT<double, 9>;

} // namespace momentum
