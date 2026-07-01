/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace pymomentum {

/// Registers the support-contact and support-polygon helper bindings on the
/// given module.
///
/// Adds the following functions to @p m:
/// - @c plane_collision_contacts_by_parent: deepest plane-collision
///   contacts grouped by parent joint.
/// - @c support_contacts: combined floor-locator and collision contacts used
///   for support polygons.
/// - @c support_polygon_from_world_points: support-plane convex hull of
///   world-space support points.
/// - @c support_polygon: support polygon for a character skeleton state.
///
/// @param m The pybind11 module to register the bindings on.
void addSupportContactBindings(pybind11::module_& m);

} // namespace pymomentum
