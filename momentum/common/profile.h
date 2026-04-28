/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/// @file
/// Profiling macros that dispatch to one of three backends, selected at compile
/// time:
///   - XR Profiler when `MOMENTUM_WITH_XR_PROFILER` is defined (forwards to
///     `XR_PROFILE_*` from `arvr/libraries/profile_redirect/annotate.hpp`).
///   - Tracy when `MOMENTUM_WITH_TRACY_PROFILER` is defined (forwards to
///     Tracy's `ZoneScoped` / `ZoneNamedN` / `FrameMark`, etc.).
///   - No-op otherwise: every `MT_PROFILE_*` macro expands to nothing, so
///     instrumentation can be left in source with zero runtime cost.
///
/// All macros are safe to use unconditionally in library code; the active
/// backend is determined entirely by the build configuration.

#if defined(MOMENTUM_WITH_XR_PROFILER)

#include <arvr/libraries/profile_redirect/annotate.hpp>

/// Scoped zone covering the enclosing function. Begins at the macro site and
/// ends when the enclosing scope exits.
#define MT_PROFILE_FUNCTION() XR_PROFILE_FUNCTION()
/// Like `MT_PROFILE_FUNCTION` but tags the zone with @p CATEGORY for filtering.
/// No-op under the Tracy backend, which has no equivalent concept here.
#define MT_PROFILE_FUNCTION_CATEGORY(CATEGORY) XR_PROFILE_FUNCTION_CATEGORY(CATEGORY)
/// Scoped zone with a static string @p NAME (must be a string literal).
#define MT_PROFILE_EVENT(NAME) XR_PROFILE_EVENT(NAME)
/// Scoped zone with a runtime-computed @p NAME. Prefer `MT_PROFILE_EVENT` when
/// a literal works; the dynamic variant has higher overhead.
#define MT_PROFILE_EVENT_DYNAMIC(NAME) XR_PROFILE_EVENT_DYNAMIC(NAME)
/// Scoped zone tagged with both a @p NAME and a @p CATEGORY. No-op under Tracy.
#define MT_PROFILE_CATEGORY(NAME, CATEGORY) XR_PROFILE_CATEGORY(NAME, CATEGORY)
/// Declares the per-scope state used by `MT_PROFILE_PUSH` / `MT_PROFILE_POP`.
/// Must appear in the enclosing scope before any push/pop pairs.
#define MT_PROFILE_PREPARE_PUSH_POP() XR_PROFILE_PREPARE_PUSH_POP()
/// Manually opens a zone with @p NAME. Each push must be balanced by a matching
/// `MT_PROFILE_POP()` in the same scope, after a `MT_PROFILE_PREPARE_PUSH_POP()`.
#define MT_PROFILE_PUSH(NAME) XR_PROFILE_PUSH(NAME)
/// Closes the most recently opened push zone. Must be balanced with
/// `MT_PROFILE_PUSH`.
#define MT_PROFILE_POP() XR_PROFILE_POP()
/// Registers the calling thread with the profiler under @p THREAD_NAME.
///
/// Side effect: assigns a human-readable name to the current thread in the
/// profiler capture. Typically called once per thread on startup.
#define MT_PROFILE_THREAD(THREAD_NAME) XR_PROFILE_THREAD(THREAD_NAME)
/// Drives any per-update bookkeeping the backend requires. No-op under Tracy.
#define MT_PROFILE_UPDATE() XR_PROFILE_UPDATE()
/// Marks the beginning of a frame. Pair with `MT_PROFILE_END_FRAME()` under
/// the XR backend; under Tracy this emits `FrameMark` and `MT_PROFILE_END_FRAME`
/// is a no-op.
#define MT_PROFILE_BEGIN_FRAME() XR_PROFILE_BEGIN_FRAME()
/// Marks the end of a frame. No-op under Tracy.
#define MT_PROFILE_END_FRAME() XR_PROFILE_END_FRAME()
/// Attaches a @p NAME / @p DATA metadata pair to the current zone or capture.
/// No-op under Tracy.
#define MT_PROFILE_METADATA(NAME, DATA) XR_PROFILE_METADATA(NAME, DATA)

#elif defined(MOMENTUM_WITH_TRACY_PROFILER)

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <stack>

#define _MT_PROFILE_CONCATENATE_DETAIL(x, y) x##y
#define _MT_PROFILE_CONCATENATE(x, y) _MT_PROFILE_CONCATENATE_DETAIL(x, y)
#define _MT_PROFILE_MAKE_UNIQUE(x) _MT_PROFILE_CONCATENATE(x, __LINE__)

#define MT_PROFILE_FUNCTION() ZoneScoped
#define MT_PROFILE_FUNCTION_CATEGORY(CATEGORY)
#define MT_PROFILE_EVENT(NAME) ZoneNamedN(_MT_PROFILE_MAKE_UNIQUE(__tracy), NAME, true)
#define MT_PROFILE_EVENT_DYNAMIC(NAME) ZoneTransientN(_MT_PROFILE_MAKE_UNIQUE(__tracy), NAME, true)
#define MT_PROFILE_CATEGORY(NAME, CATEGORY)
// TODO: Rename `___tracy_xr_stack` — the `xr_` prefix is a leftover from the XR
// backend and is misleading inside the Tracy block. The variable is a per-scope
// stack of open zones; thread-safety is not required because each scope's
// push/pop pairs are local to one thread.
#define MT_PROFILE_PREPARE_PUSH_POP() std::stack<TracyCZoneCtx> ___tracy_xr_stack
#define MT_PROFILE_PUSH(NAME)                                                                    \
  static const struct ___tracy_source_location_data TracyConcat(                                 \
      __tracy_source_location, TracyLine) = {NAME, __func__, TracyFile, (uint32_t)TracyLine, 0}; \
  ___tracy_xr_stack.push(                                                                        \
      ___tracy_emit_zone_begin(&TracyConcat(__tracy_source_location, TracyLine), true));
#define MT_PROFILE_POP()                  \
  TracyCZoneEnd(___tracy_xr_stack.top()); \
  ___tracy_xr_stack.pop()
// Wrap THREAD_NAME in a std::string so the macro accepts both std::string and
// C-string arguments; TracyCSetThreadName requires a C string.
#define MT_PROFILE_THREAD(THREAD_NAME)          \
  {                                             \
    std::string threadNameStr(THREAD_NAME);     \
    TracyCSetThreadName(threadNameStr.c_str()); \
  }
#define MT_PROFILE_UPDATE()
#define MT_PROFILE_BEGIN_FRAME() FrameMark
#define MT_PROFILE_END_FRAME()
#define MT_PROFILE_METADATA(NAME, DATA)

#else

// No profiler backend selected: every macro expands to nothing, so
// instrumentation has zero runtime cost in production builds.
#define MT_PROFILE_FUNCTION()
#define MT_PROFILE_FUNCTION_CATEGORY(CATEGORY)
#define MT_PROFILE_EVENT(NAME)
#define MT_PROFILE_EVENT_DYNAMIC(NAME)
#define MT_PROFILE_CATEGORY(NAME, CATEGORY)
#define MT_PROFILE_PREPARE_PUSH_POP()
#define MT_PROFILE_PUSH(NAME)
#define MT_PROFILE_POP()
#define MT_PROFILE_THREAD(THREAD_NAME)
#define MT_PROFILE_UPDATE()
#define MT_PROFILE_BEGIN_FRAME()
#define MT_PROFILE_END_FRAME()
#define MT_PROFILE_METADATA(NAME, DATA)

#endif
