/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/sdf_collision_error_function.h"

namespace momentum {

// Explicit template instantiations for the default SDFColliderT<float> collider type.
template class SDFCollisionErrorFunctionT<float, SDFColliderT<float>>;
template class SDFCollisionErrorFunctionT<double, SDFColliderT<float>>;

} // namespace momentum
