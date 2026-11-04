/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/exception.h>
#include <momentum/common/log.h>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

namespace momentum {

/// Return @p requestedName if it is not in @p existingNames, otherwise the first of
/// "requestedName_1", "requestedName_2", ... that is not.
///
/// @param requestedName The desired name.
/// @param existingNames Any container keyed by name with `find()` and `end()`.
/// @return A name not present in @p existingNames.
template <typename T>
std::string makeUniqueName(const std::string& requestedName, const T& existingNames) {
  if (existingNames.find(requestedName) == existingNames.end()) {
    return requestedName;
  }

  std::string candidate = requestedName + '_';
  const std::size_t baseLength = candidate.size();
  for (std::size_t suffix = 1;; ++suffix) {
    candidate.resize(baseLength);
    candidate += std::to_string(suffix);
    if (existingNames.find(candidate) == existingNames.end()) {
      return candidate;
    }
  }
}

/// Entries of a scene builder keyed by their exported (unique) name.
///
/// FbxBuilder and GltfBuilder export each character under a unique name and return it, so callers
/// can target one exported character even when several share a source name (`character.name`). An
/// empty requested name resolves to the entry added from the source name that kept that name, or
/// else to the first entry added from the source name, which keeps calls that do not pass the
/// returned name working. A character exported under the empty name is thus reachable with that
/// name.
///
/// @tparam T Per-entry payload.
template <typename T>
class ExportedNameRegistry {
 public:
  /// The name @p requestedName is exported under: itself if unused, otherwise with a numeric suffix
  /// (see makeUniqueName). Does not change the registry, so a caller can finish building an entry
  /// before it calls insert().
  [[nodiscard]] std::string makeExportedName(const std::string& requestedName) const {
    std::string exportedName = makeUniqueName(requestedName, entries_);
    if (exportedName != requestedName) {
      MT_LOGW(
          "Exported name '{}' is already in use; exporting as '{}'.", requestedName, exportedName);
    }
    return exportedName;
  }

  /// Store @p entry under @p exportedName, as returned by makeExportedName(), for a character whose
  /// source name is @p sourceName. An empty name resolves to the entry added from @p sourceName
  /// that kept that name, else to the first entry added from @p sourceName.
  T& insert(const std::string& exportedName, const std::string& sourceName, T entry) {
    const auto [it, inserted] = defaultExportedNameBySourceName_.emplace(sourceName, exportedName);
    if (!inserted && exportedName == sourceName) {
      it->second = exportedName;
    }
    return entries_.insert_or_assign(exportedName, std::move(entry)).first->second;
  }

  /// The entry a request selects: the entry exported as @p exportedName when it is not empty
  /// (throws if there is none), otherwise the default entry for @p sourceName (see insert();
  /// nullptr if there is none).
  [[nodiscard]] T* resolve(const std::string& exportedName, const std::string& sourceName) {
    if (!exportedName.empty()) {
      T* entry = find(exportedName);
      MT_THROW_IF(
          entry == nullptr, "Character '{}' has not been added to the builder", exportedName);
      return entry;
    }
    const auto it = defaultExportedNameBySourceName_.find(sourceName);
    return it == defaultExportedNameBySourceName_.end() ? nullptr : find(it->second);
  }

  /// The entry exported as @p exportedName, or nullptr.
  [[nodiscard]] T* find(const std::string& exportedName) {
    const auto it = entries_.find(exportedName);
    return it == entries_.end() ? nullptr : &it->second;
  }

  /// The number of entries.
  [[nodiscard]] std::size_t size() const {
    return entries_.size();
  }

 private:
  std::unordered_map<std::string, T> entries_;
  std::unordered_map<std::string, std::string> defaultExportedNameBySourceName_;
};

/// Throw unless motion for @p characterName, a character with @p motionJointCount joints, fits the
/// exported character it targets, which has @p targetJointCount joints.
inline void checkMotionTargetJointCount(
    const std::string& characterName,
    std::size_t motionJointCount,
    std::size_t targetJointCount) {
  MT_THROW_IF(
      motionJointCount != targetJointCount,
      "Character '{}' has {} joints but the exported character it targets has {}. Pass the name "
      "returned when the character was added to target a specific character.",
      characterName,
      motionJointCount,
      targetJointCount);
}

} // namespace momentum
