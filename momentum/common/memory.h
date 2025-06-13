/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

// Define smart pointers for a type
#ifndef MOMENTUM_DEFINE_POINTERS
#define MOMENTUM_DEFINE_POINTERS(x)               \
  using x##_p = ::std::shared_ptr<x>;             \
  using x##_u = ::std::unique_ptr<x>;             \
  using x##_w = ::std::weak_ptr<x>;               \
  using x##_const_p = ::std::shared_ptr<const x>; \
  using x##_const_u = ::std::unique_ptr<const x>; \
  using x##_const_w = ::std::weak_ptr<const x>;
#endif // MOMENTUM_DEFINE_POINTERS

// Forward-declare a struct, define smart pointers
#ifndef MOMENTUM_FWD_DECLARE_STRUCT
#define MOMENTUM_FWD_DECLARE_STRUCT(x) \
  struct x;                            \
  MOMENTUM_DEFINE_POINTERS(x);
#endif // MOMENTUM_FWD_DECLARE_STRUCT

// Forward-declare a class, define smart pointers
#ifndef MOMENTUM_FWD_DECLARE_CLASS
// NOLINTBEGIN(bugprone-macro-parentheses)
#define MOMENTUM_FWD_DECLARE_CLASS(x) \
  class x;                            \
  MOMENTUM_DEFINE_POINTERS(x)
// NOLINTEND(bugprone-macro-parentheses)
#endif // MOMENTUM_FWD_DECLARE_CLASS

// Forward-declare a templated class, define smart pointers for the class and its variants
// Use the standard backward-compatible naming scheme:
//    template <typename T> class BarT;
//    using Bar = BarT<float>;
//    using Bard = BarT<double>;
#ifndef MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS
// NOLINTBEGIN(bugprone-macro-parentheses)
#define MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(x) \
  template <typename T>                        \
  class x##T;                                  \
  using x = x##T<float>;                       \
  using x##d = x##T<double>;                   \
  MOMENTUM_DEFINE_POINTERS(x)                  \
  MOMENTUM_DEFINE_POINTERS(x##d)
// NOLINTEND(bugprone-macro-parentheses)
#endif // MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS

// Forward-declare a templated struct, define smart pointers for the struct and its variants
// Use the standard backward-compatible naming scheme:
//    template <typename T> struct BarT;
//    using Bar = BarT<float>;
//    using Bard = BarT<double>;
#ifndef MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT
// NOLINTBEGIN(bugprone-macro-parentheses)
#define MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(x) \
  template <typename T>                         \
  struct x##T;                                  \
  using x = x##T<float>;                        \
  using x##d = x##T<double>;                    \
  MOMENTUM_DEFINE_POINTERS(x)                   \
  MOMENTUM_DEFINE_POINTERS(x##d)
// NOLINTEND(bugprone-macro-parentheses)
#endif // MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT
