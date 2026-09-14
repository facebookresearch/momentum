/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Include urdfdom headers only through this file: urdfdom uses near/far as
// member names, which collide with Windows SDK macros, and the push/undef/pop
// below keeps that workaround scoped to these includes.
#ifdef _WIN32
#pragma push_macro("near")
#pragma push_macro("far")
#undef near
#undef far
#endif

#include <urdf_model/link.h>
#include <urdf_model/model.h>
#include <urdf_model/pose.h>
#include <urdf_parser/urdf_parser.h>

#ifdef _WIN32
#pragma pop_macro("far")
#pragma pop_macro("near")
#endif
