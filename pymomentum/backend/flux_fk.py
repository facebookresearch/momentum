# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math
import os
import threading
import weakref
from typing import Any

import torch
from torch.utils import dlpack as torch_dlpack

try:
    from arvr.libraries.flux import (  # @manual=//arvr/libraries/flux:flux_python_init
        flux as fx,
    )
except (AttributeError, ImportError):
    fx = None


_LN2 = math.log(2.0)
_TensorMetadataKey = tuple[
    int,
    int,
    tuple[int, ...],
    torch.dtype,
    str,
    int | None,
    int,
]
_WeakTensorAndValue = tuple[Any, Any]
_MAX_METADATA_CACHE_ENTRIES = 128
_MAX_VALIDATION_CACHE_ENTRIES = 256
_MAX_INDEX_CACHE_ENTRIES = 64
_MAX_SMALL_CACHE_ENTRIES = 128
_MAX_FLUX_TENSOR_CACHE_ENTRIES = 32

_METADATA_CACHE: dict[_TensorMetadataKey, _WeakTensorAndValue] = {}
_VALIDATION_CACHE: set[tuple[Any, ...]] = set()
_INDEX_CACHE: dict[tuple[int, int, int, str], tuple[Any, ...]] = {}
_STRIDED_OFFSET_CACHE: dict[tuple[int, int], tuple[int, ...]] = {}
_ACTIVE_ALL_CACHE: dict[int, tuple[bool, ...]] = {}
_FLUX_TENSOR_CACHE: dict[_TensorMetadataKey, _WeakTensorAndValue] = {}
_FORWARD_JIT_CACHE: dict[Any, Any] = {}
_FORWARD_JOINT_JIT_CACHE: dict[Any, Any] = {}
_BACKWARD_JIT_CACHE: dict[Any, Any] = {}
_STRIDED_CALL_ARGS_CACHE: dict[
    tuple[int, int], tuple[list[int], list[int], list[int]]
] = {}
_STRIDED_CALL_LAYOUT_CACHE: dict[tuple[int, int], Any] = {}
_BACKWARD_STRIDED_CALL_ARGS_CACHE: dict[
    tuple[int, tuple[int, ...], int, tuple[int, ...], int],
    tuple[list[int], tuple[int, ...], list[int]],
] = {}
_BACKWARD_STRIDED_CALL_LAYOUT_CACHE: dict[
    tuple[int, tuple[int, ...], int, tuple[int, ...], int],
    Any,
] = {}
_FLUX_CUDA_READY = False
_FLUX_ROW_MAJOR_STATE = "row_major"
_FLUX_COMPONENT_MAJOR_STATE = "component_major"
_CACHE_LOCK = threading.Lock()


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) != "0"


def _bounded_cache_set(
    cache: dict[Any, Any],
    key: Any,
    value: Any,
    max_entries: int,
) -> None:
    with _CACHE_LOCK:
        if key not in cache and len(cache) >= max_entries:
            cache.clear()
        cache[key] = value


def _bounded_set_add(cache: set[Any], key: Any, max_entries: int) -> None:
    with _CACHE_LOCK:
        if key not in cache and len(cache) >= max_entries:
            cache.clear()
        cache.add(key)


def _weak_tensor_ref(tensor: torch.Tensor) -> Any:
    try:
        return weakref.ref(tensor)
    except TypeError:
        return None


def _weak_tensor_matches(tensor_ref: Any, tensor: torch.Tensor) -> bool:
    return tensor_ref is None or tensor_ref() is tensor


def _use_forward_jit() -> bool:
    return _env_flag("PYMOMENTUM_FLUX_FORWARD_JIT", "1")


def _use_forward_joint_jit() -> bool:
    return _env_flag("PYMOMENTUM_FLUX_FORWARD_JOINT_JIT", "1")


def _backward_jit_mode() -> str:
    return os.environ.get("PYMOMENTUM_FLUX_BACKWARD_JIT", "auto").lower()


def _backward_jit_max_joints() -> int:
    try:
        return int(os.environ.get("PYMOMENTUM_FLUX_BACKWARD_JIT_MAX_JOINTS", "8"))
    except ValueError:
        return 8


def _use_backward_joint_jit() -> bool:
    return _env_flag("PYMOMENTUM_FLUX_BACKWARD_JOINT_JIT", "1")


def _use_strided_gather() -> bool:
    return _env_flag("PYMOMENTUM_FLUX_STRIDED_GATHER", "1")


def _use_strided_scatter() -> bool:
    return _env_flag("PYMOMENTUM_FLUX_STRIDED_SCATTER", "0")


def _use_packed_jit_output() -> bool:
    return _env_flag("PYMOMENTUM_FLUX_PACKED_JIT_OUTPUT", "1")


def _require_flux() -> None:
    if fx is None or not hasattr(fx, "from_dlpack") or not hasattr(fx, "gather"):
        raise RuntimeError(
            "Flux FK requested, but Flux is not available in this build."
        )


def _require_flux_cuda() -> None:
    global _FLUX_CUDA_READY
    _require_flux()
    if _FLUX_CUDA_READY:
        return
    if hasattr(fx, "is_cuda_available") and not fx.is_cuda_available():
        reason = ""
        if hasattr(fx, "cuda_runtime_disable_reason"):
            reason = fx.cuda_runtime_disable_reason()
        message = "Flux FK requested, but Flux CUDA runtime is not available."
        if reason:
            message = f"{message} Reason: {reason}"
        raise RuntimeError(message)
    _FLUX_CUDA_READY = True


def _should_use_backward_jit(num_joints: int) -> bool:
    mode = _backward_jit_mode()
    if mode in ("0", "false", "off"):
        return False
    if mode in ("1", "true", "on", "force"):
        return True
    return num_joints <= _backward_jit_max_joints()


def _metadata_key(tensor: torch.Tensor) -> _TensorMetadataKey:
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
        int(tensor._version),
    )


def _float_tuple_2d(tensor: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    key = _metadata_key(tensor)
    cached = _METADATA_CACHE.get(key)
    if cached is not None and _weak_tensor_matches(cached[0], tensor):
        return cached[1]
    result = tuple(
        tuple(float(value) for value in row) for row in tensor.detach().cpu().tolist()
    )
    _bounded_cache_set(
        _METADATA_CACHE,
        key,
        (_weak_tensor_ref(tensor), result),
        _MAX_METADATA_CACHE_ENTRIES,
    )
    return result


def _int_tuple_1d(tensor: torch.Tensor) -> tuple[int, ...]:
    key = _metadata_key(tensor)
    cached = _METADATA_CACHE.get(key)
    if cached is not None and _weak_tensor_matches(cached[0], tensor):
        return cached[1]
    result = tuple(int(value) for value in tensor.detach().cpu().tolist())
    _bounded_cache_set(
        _METADATA_CACHE,
        key,
        (_weak_tensor_ref(tensor), result),
        _MAX_METADATA_CACHE_ENTRIES,
    )
    return result


def _active_tuple(
    active_joints: torch.Tensor | None,
    num_joints: int,
) -> tuple[bool, ...]:
    if active_joints is None:
        cached = _ACTIVE_ALL_CACHE.get(num_joints)
        if cached is None:
            cached = (True,) * num_joints
            _bounded_cache_set(
                _ACTIVE_ALL_CACHE,
                num_joints,
                cached,
                _MAX_SMALL_CACHE_ENTRIES,
            )
        return cached
    return tuple(bool(value) for value in active_joints.detach().cpu().tolist())


def _torch_to_flux(tensor: torch.Tensor, *, cache: bool = False) -> Any:
    assert fx is not None
    if cache:
        key = _metadata_key(tensor)
        cached = _FLUX_TENSOR_CACHE.get(key)
        if cached is not None and _weak_tensor_matches(cached[0], tensor):
            return cached[1]
    if tensor.is_cuda:
        device_index = tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        fx.set_device(int(device_index))
    result = fx.from_dlpack(tensor.detach().__dlpack__())
    if cache:
        _bounded_cache_set(
            _FLUX_TENSOR_CACHE,
            key,
            (_weak_tensor_ref(tensor), result),
            _MAX_FLUX_TENSOR_CACHE_ENTRIES,
        )
    return result


def _flux_outputs_to_torch(outputs: Any) -> torch.Tensor:
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    if fx is not None and hasattr(fx, "eval"):
        fx.eval(*outputs)
    return torch.stack(
        [torch_dlpack.from_dlpack(output.__dlpack__()) for output in outputs],
        dim=1,
    )


def _packed_columns_to_torch(
    packed: Any,
    *,
    batch_size: int,
    num_joints: int,
    components_per_joint: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    del num_joints
    packed_tensor = torch_dlpack.from_dlpack(packed.__dlpack__())
    prefix_shape = shape[:-2]
    prefix_strides = []
    stride = 1
    for dim in reversed(prefix_shape):
        prefix_strides.insert(0, stride)
        stride *= dim
    if stride != batch_size:
        raise ValueError(
            f"packed output batch mismatch: shape implies {stride}, got {batch_size}"
        )
    return packed_tensor.as_strided(
        shape,
        (
            *prefix_strides,
            components_per_joint * batch_size,
            batch_size,
        ),
    )


def _row_major_packed_to_torch(packed: Any, *, shape: tuple[int, ...]) -> torch.Tensor:
    return torch_dlpack.from_dlpack(packed.__dlpack__()).reshape(shape)


def _strided_offsets(
    *,
    num_joints: int,
    components_per_joint: int,
) -> tuple[int, ...]:
    key = (num_joints, components_per_joint)
    cached = _STRIDED_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached
    result = tuple(
        joint * components_per_joint + component
        for joint in range(num_joints)
        for component in range(components_per_joint)
    )
    _bounded_cache_set(
        _STRIDED_OFFSET_CACHE,
        key,
        result,
        _MAX_SMALL_CACHE_ENTRIES,
    )
    return result


def _component_major_offsets(
    *,
    batch_size: int,
    num_joints: int,
    components_per_joint: int,
) -> tuple[int, ...]:
    return tuple(
        (joint * components_per_joint + component) * batch_size
        for joint in range(num_joints)
        for component in range(components_per_joint)
    )


def _component_indices(
    *,
    batch_size: int,
    num_joints: int,
    components_per_joint: int,
    device: torch.device,
) -> tuple[Any, ...]:
    key = (batch_size, num_joints, components_per_joint, str(device))
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    batch_base = torch.arange(batch_size, device=device, dtype=torch.int64) * (
        num_joints * components_per_joint
    )
    result = tuple(
        _torch_to_flux(batch_base + joint * components_per_joint + component)
        for joint in range(num_joints)
        for component in range(components_per_joint)
    )
    _bounded_cache_set(_INDEX_CACHE, key, result, _MAX_INDEX_CACHE_ENTRIES)
    return result


def _gather_strided_components(
    buffer,
    *,
    offsets: tuple[int, ...],
    stride: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Any, ...]:
    assert fx is not None
    gather_strided = getattr(fx, "gather_strided", None)
    if _use_strided_gather() and gather_strided is not None:
        try:
            return tuple(
                gather_strided(
                    [buffer] * len(offsets),
                    offsets,
                    stride,
                    batch_size,
                )
            )
        except (RuntimeError, TypeError):
            pass

    batch_base = torch.arange(batch_size, device=device, dtype=torch.int64) * stride
    return tuple(
        fx.gather(buffer, _torch_to_flux(batch_base + offset)) for offset in offsets
    )


def _gather_components(
    buffer,
    *,
    batch_size: int,
    num_joints: int,
    components_per_joint: int,
    device: torch.device,
) -> tuple[Any, ...]:
    assert fx is not None
    offsets = _strided_offsets(
        num_joints=num_joints,
        components_per_joint=components_per_joint,
    )
    return _gather_strided_components(
        buffer,
        offsets=offsets,
        stride=num_joints * components_per_joint,
        batch_size=batch_size,
        device=device,
    )


def _packed_columns_base(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    num_joints: int,
    components_per_joint: int,
) -> torch.Tensor | None:
    component_count = num_joints * components_per_joint
    if tuple(tensor.shape[-2:]) != (num_joints, components_per_joint):
        return None
    prefix_shape = tuple(tensor.shape[:-2])
    prefix_strides = []
    stride = 1
    for dim in reversed(prefix_shape):
        prefix_strides.insert(0, stride)
        stride *= dim
    if stride != batch_size:
        return None
    expected_strides = (
        *prefix_strides,
        components_per_joint * batch_size,
        batch_size,
    )
    if tuple(tensor.stride()) != expected_strides:
        return None
    base = getattr(tensor, "_base", None)
    if (
        isinstance(base, torch.Tensor)
        and base.is_contiguous()
        and tuple(base.shape) == (component_count, batch_size)
        and tensor.storage_offset() == 0
    ):
        return base
    return tensor.as_strided(
        (component_count, batch_size),
        (batch_size, 1),
    )


def _gather_component(
    components: tuple[Any, ...], joint: int, component: int, stride: int
):
    return components[joint * stride + component]


def _components_to_torch(
    outputs: tuple[Any, ...],
    *,
    batch_size: int,
    num_joints: int,
    components_per_joint: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    assert fx is not None
    scatter_strided = getattr(fx, "scatter_strided", None)
    if _use_strided_scatter() and scatter_strided is not None and outputs:
        offsets = _strided_offsets(
            num_joints=num_joints,
            components_per_joint=components_per_joint,
        )
        packed = fx.empty(
            (batch_size * len(outputs),),
            dtype=outputs[0].dtype,
            device=outputs[0].device,
        )
        try:
            scatter_strided(packed, list(outputs), offsets, len(outputs))
            if hasattr(fx, "eval"):
                fx.eval(packed)
            return torch_dlpack.from_dlpack(packed.__dlpack__()).reshape(shape)
        except (RuntimeError, TypeError):
            pass
    return _flux_outputs_to_torch(outputs).reshape(shape)


def _quat_multiply(x1, y1, z1, w1, x2, y2, z2, w2):
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_multiply_backward(x1, y1, z1, w1, x2, y2, z2, w2, gx, gy, gz, gw):
    return (
        gx * w2 - gy * z2 + gz * y2 - gw * x2,
        gx * z2 + gy * w2 - gz * x2 - gw * y2,
        -gx * y2 + gy * x2 + gz * w2 - gw * z2,
        gx * x2 + gy * y2 + gz * z2 + gw * w2,
        gx * w1 + gy * z1 - gz * y1 - gw * x1,
        -gx * z1 + gy * w1 + gz * x1 - gw * y1,
        gx * y1 - gy * x1 + gz * w1 - gw * z1,
        gx * x1 + gy * y1 + gz * z1 + gw * w1,
    )


def _quat_multiply_backward_parent(x1, y1, z1, w1, x2, y2, z2, w2, gx, gy, gz, gw):
    return (
        gx * w2 - gy * z2 + gz * y2 - gw * x2,
        gx * z2 + gy * w2 - gz * x2 - gw * y2,
        -gx * y2 + gy * x2 + gz * w2 - gw * z2,
        gx * x2 + gy * y2 + gz * z2 + gw * w2,
    )


def _rotate_vector(qx, qy, qz, qw, vx, vy, vz):
    return (
        (1.0 - 2.0 * qy * qy - 2.0 * qz * qz) * vx
        + (2.0 * qx * qy - 2.0 * qz * qw) * vy
        + (2.0 * qx * qz + 2.0 * qy * qw) * vz,
        (2.0 * qx * qy + 2.0 * qz * qw) * vx
        + (1.0 - 2.0 * qx * qx - 2.0 * qz * qz) * vy
        + (2.0 * qy * qz - 2.0 * qx * qw) * vz,
        (2.0 * qx * qz - 2.0 * qy * qw) * vx
        + (2.0 * qy * qz + 2.0 * qx * qw) * vy
        + (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * vz,
    )


def _rotate_vector_backward(qx, qy, qz, qw, vx, vy, vz, gx, gy, gz):
    grad_vx = (
        (1.0 - 2.0 * qy * qy - 2.0 * qz * qz) * gx
        + (2.0 * qx * qy + 2.0 * qz * qw) * gy
        + (2.0 * qx * qz - 2.0 * qy * qw) * gz
    )
    grad_vy = (
        (2.0 * qx * qy - 2.0 * qz * qw) * gx
        + (1.0 - 2.0 * qx * qx - 2.0 * qz * qz) * gy
        + (2.0 * qy * qz + 2.0 * qx * qw) * gz
    )
    grad_vz = (
        (2.0 * qx * qz + 2.0 * qy * qw) * gx
        + (2.0 * qy * qz - 2.0 * qx * qw) * gy
        + (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * gz
    )

    grad_qx = (
        (2.0 * qy * vy + 2.0 * qz * vz) * gx
        + (2.0 * qy * vx - 4.0 * qx * vy - 2.0 * qw * vz) * gy
        + (2.0 * qz * vx + 2.0 * qw * vy - 4.0 * qx * vz) * gz
    )
    grad_qy = (
        (-4.0 * qy * vx + 2.0 * qx * vy + 2.0 * qw * vz) * gx
        + (2.0 * qx * vx + 2.0 * qz * vz) * gy
        + (-2.0 * qw * vx + 2.0 * qz * vy - 4.0 * qy * vz) * gz
    )
    grad_qz = (
        (-4.0 * qz * vx - 2.0 * qw * vy + 2.0 * qx * vz) * gx
        + (2.0 * qw * vx - 4.0 * qz * vy + 2.0 * qy * vz) * gy
        + (2.0 * qx * vx + 2.0 * qy * vy) * gz
    )
    grad_qw = (
        (-2.0 * qz * vy + 2.0 * qy * vz) * gx
        + (2.0 * qz * vx - 2.0 * qx * vz) * gy
        + (-2.0 * qy * vx + 2.0 * qx * vy) * gz
    )
    return grad_qx, grad_qy, grad_qz, grad_qw, grad_vx, grad_vy, grad_vz


def _rotate_vector_backward_parent(qx, qy, qz, qw, vx, vy, vz, gx, gy, gz):
    grad_qx = (
        (2.0 * qy * vy + 2.0 * qz * vz) * gx
        + (2.0 * qy * vx - 4.0 * qx * vy - 2.0 * qw * vz) * gy
        + (2.0 * qz * vx + 2.0 * qw * vy - 4.0 * qx * vz) * gz
    )
    grad_qy = (
        (-4.0 * qy * vx + 2.0 * qx * vy + 2.0 * qw * vz) * gx
        + (2.0 * qx * vx + 2.0 * qz * vz) * gy
        + (-2.0 * qw * vx + 2.0 * qz * vy - 4.0 * qy * vz) * gz
    )
    grad_qz = (
        (-4.0 * qz * vx - 2.0 * qw * vy + 2.0 * qx * vz) * gx
        + (2.0 * qw * vx - 4.0 * qz * vy + 2.0 * qy * vz) * gy
        + (2.0 * qx * vx + 2.0 * qy * vy) * gz
    )
    grad_qw = (
        (-2.0 * qz * vy + 2.0 * qy * vz) * gx
        + (2.0 * qz * vx - 2.0 * qx * vz) * gy
        + (-2.0 * qy * vx + 2.0 * qx * vy) * gz
    )
    return grad_qx, grad_qy, grad_qz, grad_qw


def _euler_to_quaternion(rx, ry, rz):
    assert fx is not None
    cy = fx.cos(rz * 0.5)
    sy = fx.sin(rz * 0.5)
    cp = fx.cos(ry * 0.5)
    sp = fx.sin(ry * 0.5)
    cr = fx.cos(rx * 0.5)
    sr = fx.sin(rx * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _euler_backward(rx, ry, rz, gx, gy, gz, gw):
    assert fx is not None
    cy = fx.cos(rz * 0.5)
    sy = fx.sin(rz * 0.5)
    cp = fx.cos(ry * 0.5)
    sp = fx.sin(ry * 0.5)
    cr = fx.cos(rx * 0.5)
    sr = fx.sin(rx * 0.5)

    grad_rx = (
        gx * (0.5 * cr * cp * cy + 0.5 * sr * sp * sy)
        + gy * (-0.5 * sr * sp * cy + 0.5 * cr * cp * sy)
        + gz * (-0.5 * sr * cp * sy - 0.5 * cr * sp * cy)
        + gw * (-0.5 * sr * cp * cy + 0.5 * cr * sp * sy)
    )
    grad_ry = (
        gx * (-0.5 * sr * sp * cy - 0.5 * cr * cp * sy)
        + gy * (0.5 * cr * cp * cy - 0.5 * sr * sp * sy)
        + gz * (-0.5 * cr * sp * sy - 0.5 * sr * cp * cy)
        + gw * (-0.5 * cr * sp * cy + 0.5 * sr * cp * sy)
    )
    grad_rz = (
        gx * (-0.5 * sr * cp * sy - 0.5 * cr * sp * cy)
        + gy * (-0.5 * cr * sp * sy + 0.5 * sr * cp * cy)
        + gz * (0.5 * cr * cp * cy + 0.5 * sr * sp * sy)
        + gw * (-0.5 * cr * cp * sy + 0.5 * sr * sp * cy)
    )
    return grad_rx, grad_ry, grad_rz


def _load_local_state_from_buffer(
    param_components: tuple[Any, ...],
    joint: int,
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
):
    assert fx is not None
    tx = _gather_component(param_components, joint, 0, 7) + offsets[joint][0]
    ty = _gather_component(param_components, joint, 1, 7) + offsets[joint][1]
    tz = _gather_component(param_components, joint, 2, 7) + offsets[joint][2]

    rx = _gather_component(param_components, joint, 3, 7)
    ry = _gather_component(param_components, joint, 4, 7)
    rz = _gather_component(param_components, joint, 5, 7)
    eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)

    pqx, pqy, pqz, pqw = prerotations[joint]
    qx, qy, qz, qw = _quat_multiply(pqx, pqy, pqz, pqw, eqx, eqy, eqz, eqw)
    scale = fx.exp(_gather_component(param_components, joint, 6, 7) * _LN2)
    return tx, ty, tz, qx, qy, qz, qw, scale


def _compute_active_joint_gradients(
    pqx,
    pqy,
    pqz,
    pqw,
    ltx,
    lty,
    ltz,
    lqx,
    lqy,
    lqz,
    lqw,
    gqx,
    gqy,
    gqz,
    gqw,
    scaled_gtx,
    scaled_gty,
    scaled_gtz,
):
    (
        grad_parent_qx_t,
        grad_parent_qy_t,
        grad_parent_qz_t,
        grad_parent_qw_t,
        grad_local_tx,
        grad_local_ty,
        grad_local_tz,
    ) = _rotate_vector_backward(
        pqx, pqy, pqz, pqw, ltx, lty, ltz, scaled_gtx, scaled_gty, scaled_gtz
    )
    (
        grad_parent_qx_q,
        grad_parent_qy_q,
        grad_parent_qz_q,
        grad_parent_qw_q,
        grad_local_qx,
        grad_local_qy,
        grad_local_qz,
        grad_local_qw,
    ) = _quat_multiply_backward(
        pqx, pqy, pqz, pqw, lqx, lqy, lqz, lqw, gqx, gqy, gqz, gqw
    )
    return (
        grad_parent_qx_t + grad_parent_qx_q,
        grad_parent_qy_t + grad_parent_qy_q,
        grad_parent_qz_t + grad_parent_qz_q,
        grad_parent_qw_t + grad_parent_qw_q,
        grad_local_tx,
        grad_local_ty,
        grad_local_tz,
        grad_local_qx,
        grad_local_qy,
        grad_local_qz,
        grad_local_qw,
    )


def _compute_inactive_joint_gradients(
    pqx,
    pqy,
    pqz,
    pqw,
    ltx,
    lty,
    ltz,
    lqx,
    lqy,
    lqz,
    lqw,
    gqx,
    gqy,
    gqz,
    gqw,
    scaled_gtx,
    scaled_gty,
    scaled_gtz,
):
    grad_parent_qx_t, grad_parent_qy_t, grad_parent_qz_t, grad_parent_qw_t = (
        _rotate_vector_backward_parent(
            pqx, pqy, pqz, pqw, ltx, lty, ltz, scaled_gtx, scaled_gty, scaled_gtz
        )
    )
    grad_parent_qx_q, grad_parent_qy_q, grad_parent_qz_q, grad_parent_qw_q = (
        _quat_multiply_backward_parent(
            pqx, pqy, pqz, pqw, lqx, lqy, lqz, lqw, gqx, gqy, gqz, gqw
        )
    )
    return (
        grad_parent_qx_t + grad_parent_qx_q,
        grad_parent_qy_t + grad_parent_qy_q,
        grad_parent_qz_t + grad_parent_qz_q,
        grad_parent_qw_t + grad_parent_qw_q,
        scaled_gtx * 0.0,
        scaled_gty * 0.0,
        scaled_gtz * 0.0,
        gqx * 0.0,
        gqy * 0.0,
        gqz * 0.0,
        gqw * 0.0,
    )


def _compute_backward_gradients(
    pqx,
    pqy,
    pqz,
    pqw,
    ps,
    ltx,
    lty,
    ltz,
    lqx,
    lqy,
    lqz,
    lqw,
    ls,
    gtx,
    gty,
    gtz,
    gqx,
    gqy,
    gqz,
    gqw,
    gs,
    compute_joint_grad: bool,
):
    rtx, rty, rtz = _rotate_vector(pqx, pqy, pqz, pqw, ltx, lty, ltz)
    scaled_gtx = ps * gtx
    scaled_gty = ps * gty
    scaled_gtz = ps * gtz
    if compute_joint_grad:
        (
            grad_parent_qx,
            grad_parent_qy,
            grad_parent_qz,
            grad_parent_qw,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
        ) = _compute_active_joint_gradients(
            pqx,
            pqy,
            pqz,
            pqw,
            ltx,
            lty,
            ltz,
            lqx,
            lqy,
            lqz,
            lqw,
            gqx,
            gqy,
            gqz,
            gqw,
            scaled_gtx,
            scaled_gty,
            scaled_gtz,
        )
    else:
        (
            grad_parent_qx,
            grad_parent_qy,
            grad_parent_qz,
            grad_parent_qw,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
        ) = _compute_inactive_joint_gradients(
            pqx,
            pqy,
            pqz,
            pqw,
            ltx,
            lty,
            ltz,
            lqx,
            lqy,
            lqz,
            lqw,
            gqx,
            gqy,
            gqz,
            gqw,
            scaled_gtx,
            scaled_gty,
            scaled_gtz,
        )
    return (
        gtx,
        gty,
        gtz,
        grad_parent_qx,
        grad_parent_qy,
        grad_parent_qz,
        grad_parent_qw,
        rtx * gtx + rty * gty + rtz * gtz + ls * gs,
        grad_local_tx,
        grad_local_ty,
        grad_local_tz,
        grad_local_qx,
        grad_local_qy,
        grad_local_qz,
        grad_local_qw,
        ps * gs,
    )


def _validate_inputs(  # noqa: C901
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
    active_joints: torch.Tensor | None,
) -> None:
    validation_key = (
        _metadata_key(joint_parameters),
        _metadata_key(joint_translation_offsets),
        _metadata_key(joint_prerotations),
        _metadata_key(joint_parents),
        None if active_joints is None else _metadata_key(active_joints),
    )
    if validation_key in _VALIDATION_CACHE:
        return

    if not joint_parameters.is_cuda:
        raise RuntimeError("Flux FK requested, but joint_parameters is on CPU.")
    if joint_parameters.dtype != torch.float32:
        raise RuntimeError("Flux FK currently only supports float32 joint parameters.")
    if joint_parameters.ndim < 2 or joint_parameters.shape[-1] != 7:
        raise RuntimeError("joint_parameters must have shape [..., num_joints, 7].")
    if joint_translation_offsets.device != joint_parameters.device:
        raise RuntimeError("joint_translation_offsets must be on the same device.")
    if joint_prerotations.device != joint_parameters.device:
        raise RuntimeError("joint_prerotations must be on the same device.")
    if joint_parents.device != joint_parameters.device:
        raise RuntimeError("joint_parents must be on the same device.")
    if joint_translation_offsets.dtype != torch.float32:
        raise RuntimeError("joint_translation_offsets must be float32.")
    if joint_prerotations.dtype != torch.float32:
        raise RuntimeError("joint_prerotations must be float32.")
    if joint_parents.dtype != torch.int32:
        raise RuntimeError("joint_parents must be int32.")
    num_joints = joint_parameters.shape[-2]
    if joint_translation_offsets.shape != (num_joints, 3):
        raise RuntimeError("joint_translation_offsets must have shape [num_joints, 3].")
    if joint_prerotations.shape != (num_joints, 4):
        raise RuntimeError("joint_prerotations must have shape [num_joints, 4].")
    if joint_parents.shape != (num_joints,):
        raise RuntimeError("joint_parents must have shape [num_joints].")
    if active_joints is not None:
        if active_joints.device != joint_parameters.device:
            raise RuntimeError("active_joints must be on the same device.")
        if active_joints.dtype != torch.bool:
            raise RuntimeError("active_joints must be bool.")
        if active_joints.shape != (num_joints,):
            raise RuntimeError("active_joints must have shape [num_joints].")
    _bounded_set_add(
        _VALIDATION_CACHE,
        validation_key,
        _MAX_VALIDATION_CACHE_ENTRIES,
    )


def _forward_joint_jit(
    *,
    has_parent: bool,
    offset: tuple[float, ...],
    prerotation: tuple[float, ...],
):
    assert fx is not None
    key = (has_parent, offset, prerotation)
    cached = _FORWARD_JOINT_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    ox, oy, oz = offset
    pqx_pre, pqy_pre, pqz_pre, pqw_pre = prerotation

    if has_parent:

        @fx.jit
        def forward_child(
            ptx,
            pty,
            ptz,
            pqx,
            pqy,
            pqz,
            pqw,
            ps,
            txp,
            typ,
            tzp,
            rx,
            ry,
            rz,
            scale_param,
        ):
            ltx = txp + ox
            lty = typ + oy
            ltz = tzp + oz
            eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)
            lqx, lqy, lqz, lqw = _quat_multiply(
                pqx_pre, pqy_pre, pqz_pre, pqw_pre, eqx, eqy, eqz, eqw
            )
            ls = fx.exp(scale_param * _LN2)
            rtx, rty, rtz = _rotate_vector(pqx, pqy, pqz, pqw, ltx, lty, ltz)
            gtx = ptx + ps * rtx
            gty = pty + ps * rty
            gtz = ptz + ps * rtz
            gqx, gqy, gqz, gqw = _quat_multiply(pqx, pqy, pqz, pqw, lqx, lqy, lqz, lqw)
            return gtx, gty, gtz, gqx, gqy, gqz, gqw, ps * ls

        _bounded_cache_set(
            _FORWARD_JOINT_JIT_CACHE,
            key,
            forward_child,
            _MAX_SMALL_CACHE_ENTRIES,
        )
        return forward_child

    @fx.jit
    def forward_root(txp, typ, tzp, rx, ry, rz, scale_param):
        ltx = txp + ox
        lty = typ + oy
        ltz = tzp + oz
        eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)
        gqx, gqy, gqz, gqw = _quat_multiply(
            pqx_pre, pqy_pre, pqz_pre, pqw_pre, eqx, eqy, eqz, eqw
        )
        return ltx, lty, ltz, gqx, gqy, gqz, gqw, fx.exp(scale_param * _LN2)

    _bounded_cache_set(
        _FORWARD_JOINT_JIT_CACHE,
        key,
        forward_root,
        _MAX_SMALL_CACHE_ENTRIES,
    )
    return forward_root


def _forward_joint_eager(
    param_components: tuple[Any, ...],
    *,
    joint: int,
    offset: tuple[float, ...],
    prerotation: tuple[float, ...],
    parent_state: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    assert fx is not None
    ltx = _gather_component(param_components, joint, 0, 7) + offset[0]
    lty = _gather_component(param_components, joint, 1, 7) + offset[1]
    ltz = _gather_component(param_components, joint, 2, 7) + offset[2]
    rx = _gather_component(param_components, joint, 3, 7)
    ry = _gather_component(param_components, joint, 4, 7)
    rz = _gather_component(param_components, joint, 5, 7)
    eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)
    pqx_pre, pqy_pre, pqz_pre, pqw_pre = prerotation
    lqx, lqy, lqz, lqw = _quat_multiply(
        pqx_pre, pqy_pre, pqz_pre, pqw_pre, eqx, eqy, eqz, eqw
    )
    ls = fx.exp(_gather_component(param_components, joint, 6, 7) * _LN2)
    if parent_state is None:
        return ltx, lty, ltz, lqx, lqy, lqz, lqw, ls
    ptx, pty, ptz, pqx, pqy, pqz, pqw, ps = parent_state
    rtx, rty, rtz = _rotate_vector(pqx, pqy, pqz, pqw, ltx, lty, ltz)
    gqx, gqy, gqz, gqw = _quat_multiply(pqx, pqy, pqz, pqw, lqx, lqy, lqz, lqw)
    return ptx + ps * rtx, pty + ps * rty, ptz + ps * rtz, gqx, gqy, gqz, gqw, ps * ls


def _forward_joint(
    param_components: tuple[Any, ...],
    *,
    joint: int,
    offset: tuple[float, ...],
    prerotation: tuple[float, ...],
    parent_state: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    if _use_forward_joint_jit() and hasattr(fx, "jit"):
        try:
            txp = _gather_component(param_components, joint, 0, 7)
            typ = _gather_component(param_components, joint, 1, 7)
            tzp = _gather_component(param_components, joint, 2, 7)
            rx = _gather_component(param_components, joint, 3, 7)
            ry = _gather_component(param_components, joint, 4, 7)
            rz = _gather_component(param_components, joint, 5, 7)
            scale_param = _gather_component(param_components, joint, 6, 7)
            kernel = _forward_joint_jit(
                has_parent=parent_state is not None,
                offset=offset,
                prerotation=prerotation,
            )
            if parent_state is None:
                return tuple(kernel(txp, typ, tzp, rx, ry, rz, scale_param))
            return tuple(kernel(*parent_state, txp, typ, tzp, rx, ry, rz, scale_param))
        except (RuntimeError, TypeError, ValueError):
            pass
    return _forward_joint_eager(
        param_components,
        joint=joint,
        offset=offset,
        prerotation=prerotation,
        parent_state=parent_state,
    )


def _forward_components(
    param_components: tuple[Any, ...],
    *,
    num_joints: int,
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
) -> tuple[Any, ...]:
    assert fx is not None
    global_tx = []
    global_ty = []
    global_tz = []
    global_qx = []
    global_qy = []
    global_qz = []
    global_qw = []
    global_s = []

    for joint in range(num_joints):
        parent = parents[joint]
        if parent >= 0:
            parent_state = (
                global_tx[parent],
                global_ty[parent],
                global_tz[parent],
                global_qx[parent],
                global_qy[parent],
                global_qz[parent],
                global_qw[parent],
                global_s[parent],
            )
        else:
            parent_state = None

        gtx, gty, gtz, gqx, gqy, gqz, gqw, gs = _forward_joint(
            param_components,
            joint=joint,
            offset=offsets[joint],
            prerotation=prerotations[joint],
            parent_state=parent_state,
        )

        global_tx.append(gtx)
        global_ty.append(gty)
        global_tz.append(gtz)
        global_qx.append(gqx)
        global_qy.append(gqy)
        global_qz.append(gqz)
        global_qw.append(gqw)
        global_s.append(gs)

    outputs = []
    for joint in range(num_joints):
        outputs.extend(
            [
                global_tx[joint],
                global_ty[joint],
                global_tz[joint],
                global_qx[joint],
                global_qy[joint],
                global_qz[joint],
                global_qw[joint],
                global_s[joint],
            ]
        )
    return tuple(outputs)


def _forward_jit_key(
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
) -> tuple[Any, ...]:
    return (offsets, prerotations, parents)


def _forward_jit(
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
):
    assert fx is not None
    key = _forward_jit_key(offsets, prerotations, parents)
    cached = _FORWARD_JIT_CACHE.get(key)
    if cached is not None:
        return cached
    num_joints = len(parents)

    @fx.jit(block_dim=128)
    def forward_kernel(*param_components):
        return _forward_components(
            tuple(param_components),
            num_joints=num_joints,
            offsets=offsets,
            prerotations=prerotations,
            parents=parents,
        )

    _bounded_cache_set(
        _FORWARD_JIT_CACHE, key, forward_kernel, _MAX_SMALL_CACHE_ENTRIES
    )
    return forward_kernel


def _strided_call_args(
    *,
    num_joints: int,
    components_per_joint: int,
) -> tuple[list[int], list[int], list[int]]:
    key = (num_joints, components_per_joint)
    cached = _STRIDED_CALL_ARGS_CACHE.get(key)
    if cached is not None:
        return cached
    offsets = _strided_offsets(
        num_joints=num_joints,
        components_per_joint=components_per_joint,
    )
    result = (
        [0] * len(offsets),
        list(offsets),
        [num_joints * components_per_joint] * len(offsets),
    )
    _bounded_cache_set(
        _STRIDED_CALL_ARGS_CACHE,
        key,
        result,
        _MAX_SMALL_CACHE_ENTRIES,
    )
    return result


def _strided_input_layout(base_indices, offsets, strides):
    try:
        from arvr.libraries.flux import (  # @manual=//arvr/libraries/flux:_flux_core
            _flux_core,
        )
    except (AttributeError, ImportError):
        return None
    cuda_jit = getattr(getattr(_flux_core, "tracing", None), "cuda_jit", None)
    layout_cls = getattr(cuda_jit, "StridedInputLayout", None)
    if layout_cls is None:
        return None
    try:
        return layout_cls(base_indices, offsets, strides)
    except (RuntimeError, TypeError, ValueError):
        return None


def _strided_call_layout(
    *,
    num_joints: int,
    components_per_joint: int,
):
    key = (num_joints, components_per_joint)
    cached = _STRIDED_CALL_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    base_indices, offsets, strides = _strided_call_args(
        num_joints=num_joints,
        components_per_joint=components_per_joint,
    )
    layout = _strided_input_layout(base_indices, offsets, strides)
    if layout is not None:
        _bounded_cache_set(
            _STRIDED_CALL_LAYOUT_CACHE,
            key,
            layout,
            _MAX_SMALL_CACHE_ENTRIES,
        )
    return layout


def _backward_strided_call_args(
    *,
    num_joints: int,
    state_component_offsets: tuple[int, ...],
    state_component_stride: int,
    grad_component_offsets: tuple[int, ...],
    grad_component_stride: int,
) -> tuple[list[int], tuple[int, ...], list[int]]:
    key = (
        num_joints,
        state_component_offsets,
        state_component_stride,
        grad_component_offsets,
        grad_component_stride,
    )
    cached = _BACKWARD_STRIDED_CALL_ARGS_CACHE.get(key)
    if cached is not None:
        return cached

    param_offsets = _strided_offsets(
        num_joints=num_joints,
        components_per_joint=7,
    )
    result = (
        [0] * len(param_offsets)
        + [1] * len(state_component_offsets)
        + [2] * len(grad_component_offsets),
        param_offsets + state_component_offsets + grad_component_offsets,
        [num_joints * 7] * len(param_offsets)
        + [state_component_stride] * len(state_component_offsets)
        + [grad_component_stride] * len(grad_component_offsets),
    )
    _bounded_cache_set(
        _BACKWARD_STRIDED_CALL_ARGS_CACHE,
        key,
        result,
        _MAX_SMALL_CACHE_ENTRIES,
    )
    return result


def _backward_strided_call_layout(
    *,
    num_joints: int,
    state_component_offsets: tuple[int, ...],
    state_component_stride: int,
    grad_component_offsets: tuple[int, ...],
    grad_component_stride: int,
):
    key = (
        num_joints,
        state_component_offsets,
        state_component_stride,
        grad_component_offsets,
        grad_component_stride,
    )
    cached = _BACKWARD_STRIDED_CALL_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    base_indices, offsets, strides = _backward_strided_call_args(
        num_joints=num_joints,
        state_component_offsets=state_component_offsets,
        state_component_stride=state_component_stride,
        grad_component_offsets=grad_component_offsets,
        grad_component_stride=grad_component_stride,
    )
    layout = _strided_input_layout(base_indices, offsets, strides)
    if layout is not None:
        _bounded_cache_set(
            _BACKWARD_STRIDED_CALL_LAYOUT_CACHE,
            key,
            layout,
            _MAX_SMALL_CACHE_ENTRIES,
        )
    return layout


def _forward_array_ops(  # noqa: C901
    joint_parameters: torch.Tensor,
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
    *,
    return_flux_output: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Any | None]:
    num_joints = joint_parameters.shape[-2]
    batch_size = joint_parameters.numel() // (num_joints * 7)
    joint_params = _torch_to_flux(joint_parameters, cache=True)
    outputs = None
    param_components = None
    if _use_forward_jit() and hasattr(fx, "jit"):
        kernel = _forward_jit(offsets, prerotations, parents)
        if _use_packed_jit_output():
            layout = _strided_call_layout(
                num_joints=num_joints,
                components_per_joint=7,
            )
            row_major_layout1 = getattr(
                kernel, "call_row_major_outputs_strided_layout1", None
            )
            if layout is not None and row_major_layout1 is not None:
                try:
                    packed = row_major_layout1(
                        joint_params,
                        layout,
                        batch_size,
                    )
                    if packed is not None:
                        output = _row_major_packed_to_torch(
                            packed,
                            shape=(*joint_parameters.shape[:-1], 8),
                        )
                        if return_flux_output:
                            return output, (packed, _FLUX_ROW_MAJOR_STATE)
                        return output
                except (RuntimeError, TypeError, ValueError):
                    pass

            row_major_layout = getattr(
                kernel, "call_row_major_outputs_strided_layout", None
            )
            if layout is not None and row_major_layout is not None:
                try:
                    packed = row_major_layout(
                        [joint_params],
                        layout,
                        batch_size,
                    )
                    if packed is not None:
                        output = _row_major_packed_to_torch(
                            packed,
                            shape=(*joint_parameters.shape[:-1], 8),
                        )
                        if return_flux_output:
                            return output, (packed, _FLUX_ROW_MAJOR_STATE)
                        return output
                except (RuntimeError, TypeError, ValueError):
                    pass

            base_indices, component_offsets, component_strides = _strided_call_args(
                num_joints=num_joints,
                components_per_joint=7,
            )
            row_major_strided = getattr(kernel, "call_row_major_outputs_strided", None)
            if row_major_strided is not None:
                try:
                    packed = row_major_strided(
                        [joint_params],
                        base_indices,
                        component_offsets,
                        component_strides,
                        batch_size,
                    )
                    if packed is not None:
                        output = _row_major_packed_to_torch(
                            packed,
                            shape=(*joint_parameters.shape[:-1], 8),
                        )
                        if return_flux_output:
                            return output, packed
                        return output
                except (RuntimeError, TypeError, ValueError):
                    pass
            packed_layout = getattr(kernel, "call_packed_outputs_strided_layout", None)
            if layout is not None and packed_layout is not None:
                try:
                    packed = packed_layout(
                        [joint_params],
                        layout,
                        batch_size,
                    )
                    if packed is not None:
                        output = _packed_columns_to_torch(
                            packed,
                            batch_size=batch_size,
                            num_joints=num_joints,
                            components_per_joint=8,
                            shape=(*joint_parameters.shape[:-1], 8),
                        )
                        if return_flux_output:
                            return output, (packed, _FLUX_COMPONENT_MAJOR_STATE)
                        return output
                except (RuntimeError, TypeError, ValueError):
                    pass
            packed_strided = getattr(kernel, "call_packed_outputs_strided", None)
            if packed_strided is not None:
                try:
                    packed = packed_strided(
                        [joint_params],
                        base_indices,
                        component_offsets,
                        component_strides,
                        batch_size,
                    )
                    if packed is not None:
                        output = _packed_columns_to_torch(
                            packed,
                            batch_size=batch_size,
                            num_joints=num_joints,
                            components_per_joint=8,
                            shape=(*joint_parameters.shape[:-1], 8),
                        )
                        if return_flux_output:
                            return output, (packed, _FLUX_COMPONENT_MAJOR_STATE)
                        return output
                except (RuntimeError, TypeError, ValueError):
                    pass
            param_components = _gather_components(
                joint_params,
                batch_size=batch_size,
                num_joints=num_joints,
                components_per_joint=7,
                device=joint_parameters.device,
            )
            packed_outputs = getattr(kernel, "call_packed_outputs", None)
            if packed_outputs is not None:
                try:
                    packed = packed_outputs(*param_components)
                    if packed is not None:
                        output = _packed_columns_to_torch(
                            packed,
                            batch_size=batch_size,
                            num_joints=num_joints,
                            components_per_joint=8,
                            shape=(*joint_parameters.shape[:-1], 8),
                        )
                        if return_flux_output:
                            return output, (packed, _FLUX_COMPONENT_MAJOR_STATE)
                        return output
                except (RuntimeError, TypeError, ValueError):
                    pass
        try:
            if param_components is None:
                param_components = _gather_components(
                    joint_params,
                    batch_size=batch_size,
                    num_joints=num_joints,
                    components_per_joint=7,
                    device=joint_parameters.device,
                )
            outputs = tuple(kernel(*param_components))
        except (RuntimeError, TypeError, ValueError):
            outputs = None
    if outputs is None:
        if param_components is None:
            param_components = _gather_components(
                joint_params,
                batch_size=batch_size,
                num_joints=num_joints,
                components_per_joint=7,
                device=joint_parameters.device,
            )
        outputs = _forward_components(
            param_components,
            num_joints=num_joints,
            offsets=offsets,
            prerotations=prerotations,
            parents=parents,
        )
    output = _components_to_torch(
        outputs,
        batch_size=batch_size,
        num_joints=num_joints,
        components_per_joint=8,
        shape=(*joint_parameters.shape[:-1], 8),
    )
    if return_flux_output:
        return output, None
    return output


def _joint_param_grads_from_local(
    rx,
    ry,
    rz,
    eqx,
    eqy,
    eqz,
    eqw,
    ls,
    grad_local_tx,
    grad_local_ty,
    grad_local_tz,
    grad_local_qx,
    grad_local_qy,
    grad_local_qz,
    grad_local_qw,
    grad_local_s,
    prerotation: tuple[float, ...],
) -> tuple[Any, ...]:
    pqx_pre, pqy_pre, pqz_pre, pqw_pre = prerotation
    _, _, _, _, grad_eqx, grad_eqy, grad_eqz, grad_eqw = _quat_multiply_backward(
        pqx_pre,
        pqy_pre,
        pqz_pre,
        pqw_pre,
        eqx,
        eqy,
        eqz,
        eqw,
        grad_local_qx,
        grad_local_qy,
        grad_local_qz,
        grad_local_qw,
    )
    grad_rx, grad_ry, grad_rz = _euler_backward(
        rx, ry, rz, grad_eqx, grad_eqy, grad_eqz, grad_eqw
    )
    return (
        grad_local_tx,
        grad_local_ty,
        grad_local_tz,
        grad_rx,
        grad_ry,
        grad_rz,
        grad_local_s * ls * _LN2,
    )


def _backward_joint_jit(
    *,
    has_parent: bool,
    active: bool,
    offset: tuple[float, ...],
    prerotation: tuple[float, ...],
):
    assert fx is not None
    key = (has_parent, active, offset, prerotation)
    cached = _BACKWARD_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    ox, oy, oz = offset
    pqx_pre, pqy_pre, pqz_pre, pqw_pre = prerotation

    if has_parent:

        @fx.jit
        def backward_child(
            pqx,
            pqy,
            pqz,
            pqw,
            ps,
            txp,
            typ,
            tzp,
            rx,
            ry,
            rz,
            scale_param,
            gtx,
            gty,
            gtz,
            gqx,
            gqy,
            gqz,
            gqw,
            gs,
            parent_grad_tx,
            parent_grad_ty,
            parent_grad_tz,
            parent_grad_qx,
            parent_grad_qy,
            parent_grad_qz,
            parent_grad_qw,
            parent_grad_s,
        ):
            ltx = txp + ox
            lty = typ + oy
            ltz = tzp + oz
            eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)
            lqx, lqy, lqz, lqw = _quat_multiply(
                pqx_pre, pqy_pre, pqz_pre, pqw_pre, eqx, eqy, eqz, eqw
            )
            ls = fx.exp(scale_param * _LN2)
            (
                grad_parent_tx,
                grad_parent_ty,
                grad_parent_tz,
                grad_parent_qx,
                grad_parent_qy,
                grad_parent_qz,
                grad_parent_qw,
                grad_parent_s,
                grad_local_tx,
                grad_local_ty,
                grad_local_tz,
                grad_local_qx,
                grad_local_qy,
                grad_local_qz,
                grad_local_qw,
                grad_local_s,
            ) = _compute_backward_gradients(
                pqx,
                pqy,
                pqz,
                pqw,
                ps,
                ltx,
                lty,
                ltz,
                lqx,
                lqy,
                lqz,
                lqw,
                ls,
                gtx,
                gty,
                gtz,
                gqx,
                gqy,
                gqz,
                gqw,
                gs,
                active,
            )
            if active:
                grad_params = _joint_param_grads_from_local(
                    rx,
                    ry,
                    rz,
                    eqx,
                    eqy,
                    eqz,
                    eqw,
                    ls,
                    grad_local_tx,
                    grad_local_ty,
                    grad_local_tz,
                    grad_local_qx,
                    grad_local_qy,
                    grad_local_qz,
                    grad_local_qw,
                    grad_local_s,
                    prerotation,
                )
            else:
                zero = gtx * 0.0
                grad_params = (zero, zero, zero, zero, zero, zero, zero)
            return (
                parent_grad_tx + grad_parent_tx,
                parent_grad_ty + grad_parent_ty,
                parent_grad_tz + grad_parent_tz,
                parent_grad_qx + grad_parent_qx,
                parent_grad_qy + grad_parent_qy,
                parent_grad_qz + grad_parent_qz,
                parent_grad_qw + grad_parent_qw,
                parent_grad_s + grad_parent_s,
                *grad_params,
            )

        _bounded_cache_set(
            _BACKWARD_JIT_CACHE,
            key,
            backward_child,
            _MAX_SMALL_CACHE_ENTRIES,
        )
        return backward_child

    @fx.jit
    def backward_root(
        txp,
        typ,
        tzp,
        rx,
        ry,
        rz,
        scale_param,
        gtx,
        gty,
        gtz,
        gqx,
        gqy,
        gqz,
        gqw,
        gs,
    ):
        zero = gtx * 0.0
        ltx = txp + ox
        lty = typ + oy
        ltz = tzp + oz
        eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)
        lqx, lqy, lqz, lqw = _quat_multiply(
            pqx_pre, pqy_pre, pqz_pre, pqw_pre, eqx, eqy, eqz, eqw
        )
        ls = fx.exp(scale_param * _LN2)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
            grad_local_s,
        ) = _compute_backward_gradients(
            zero,
            zero,
            zero,
            zero + 1.0,
            zero + 1.0,
            ltx,
            lty,
            ltz,
            lqx,
            lqy,
            lqz,
            lqw,
            ls,
            gtx,
            gty,
            gtz,
            gqx,
            gqy,
            gqz,
            gqw,
            gs,
            active,
        )
        if active:
            return _joint_param_grads_from_local(
                rx,
                ry,
                rz,
                eqx,
                eqy,
                eqz,
                eqw,
                ls,
                grad_local_tx,
                grad_local_ty,
                grad_local_tz,
                grad_local_qx,
                grad_local_qy,
                grad_local_qz,
                grad_local_qw,
                grad_local_s,
                prerotation,
            )
        return zero, zero, zero, zero, zero, zero, zero

    _bounded_cache_set(
        _BACKWARD_JIT_CACHE,
        key,
        backward_root,
        _MAX_SMALL_CACHE_ENTRIES,
    )
    return backward_root


def _backward_joint(
    param_components: tuple[Any, ...],
    state_components: tuple[Any, ...],
    grad_work: tuple[list[Any], ...],
    *,
    joint: int,
    parent: int,
    offset: tuple[float, ...],
    prerotation: tuple[float, ...],
    active: bool,
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
    if _use_backward_joint_jit() and hasattr(fx, "jit"):
        try:
            txp = _gather_component(param_components, joint, 0, 7)
            typ = _gather_component(param_components, joint, 1, 7)
            tzp = _gather_component(param_components, joint, 2, 7)
            rx = _gather_component(param_components, joint, 3, 7)
            ry = _gather_component(param_components, joint, 4, 7)
            rz = _gather_component(param_components, joint, 5, 7)
            scale_param = _gather_component(param_components, joint, 6, 7)
            joint_grad = tuple(component[joint] for component in grad_work)
            kernel = _backward_joint_jit(
                has_parent=parent >= 0,
                active=active,
                offset=offset,
                prerotation=prerotation,
            )
            if parent >= 0:
                parent_state = (
                    _gather_component(state_components, parent, 3, 8),
                    _gather_component(state_components, parent, 4, 8),
                    _gather_component(state_components, parent, 5, 8),
                    _gather_component(state_components, parent, 6, 8),
                    _gather_component(state_components, parent, 7, 8),
                )
                parent_grad = tuple(component[parent] for component in grad_work)
                result = tuple(
                    kernel(
                        *parent_state,
                        txp,
                        typ,
                        tzp,
                        rx,
                        ry,
                        rz,
                        scale_param,
                        *joint_grad,
                        *parent_grad,
                    )
                )
                return result[:8], result[8:]
            return None, tuple(
                kernel(txp, typ, tzp, rx, ry, rz, scale_param, *joint_grad)
            )
        except (RuntimeError, TypeError, ValueError):
            pass
    return None, None


def _backward_components(
    param_components: tuple[Any, ...],
    state_components: tuple[Any, ...],
    grad_output_components: tuple[Any, ...],
    *,
    num_joints: int,
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
    active_joints: tuple[bool, ...],
) -> tuple[Any, ...]:
    assert fx is not None
    grad_work_tx = []
    grad_work_ty = []
    grad_work_tz = []
    grad_work_qx = []
    grad_work_qy = []
    grad_work_qz = []
    grad_work_qw = []
    grad_work_s = []

    for joint in range(num_joints):
        grad_work_tx.append(_gather_component(grad_output_components, joint, 0, 8))
        grad_work_ty.append(_gather_component(grad_output_components, joint, 1, 8))
        grad_work_tz.append(_gather_component(grad_output_components, joint, 2, 8))
        grad_work_qx.append(_gather_component(grad_output_components, joint, 3, 8))
        grad_work_qy.append(_gather_component(grad_output_components, joint, 4, 8))
        grad_work_qz.append(_gather_component(grad_output_components, joint, 5, 8))
        grad_work_qw.append(_gather_component(grad_output_components, joint, 6, 8))
        grad_work_s.append(_gather_component(grad_output_components, joint, 7, 8))

    zero = grad_work_tx[0] * 0.0
    grad_params: list[tuple[Any, ...] | None] = [None] * num_joints
    zero_grad_params = None
    grad_work = (
        grad_work_tx,
        grad_work_ty,
        grad_work_tz,
        grad_work_qx,
        grad_work_qy,
        grad_work_qz,
        grad_work_qw,
        grad_work_s,
    )

    for joint in range(num_joints - 1, -1, -1):
        parent = parents[joint]
        parent_update, joint_grad_params = _backward_joint(
            param_components,
            state_components,
            grad_work,
            joint=joint,
            parent=parent,
            offset=offsets[joint],
            prerotation=prerotations[joint],
            active=active_joints[joint],
        )
        if joint_grad_params is not None:
            if parent >= 0 and parent_update is not None:
                grad_work_tx[parent] = parent_update[0]
                grad_work_ty[parent] = parent_update[1]
                grad_work_tz[parent] = parent_update[2]
                grad_work_qx[parent] = parent_update[3]
                grad_work_qy[parent] = parent_update[4]
                grad_work_qz[parent] = parent_update[5]
                grad_work_qw[parent] = parent_update[6]
                grad_work_s[parent] = parent_update[7]
            grad_params[joint] = joint_grad_params
            continue

        ltx, lty, ltz, lqx, lqy, lqz, lqw, ls = _load_local_state_from_buffer(
            param_components,
            joint,
            offsets,
            prerotations,
        )
        gtx = grad_work_tx[joint]
        gty = grad_work_ty[joint]
        gtz = grad_work_tz[joint]
        gqx = grad_work_qx[joint]
        gqy = grad_work_qy[joint]
        gqz = grad_work_qz[joint]
        gqw = grad_work_qw[joint]
        gs = grad_work_s[joint]

        if parent >= 0:
            pqx = _gather_component(state_components, parent, 3, 8)
            pqy = _gather_component(state_components, parent, 4, 8)
            pqz = _gather_component(state_components, parent, 5, 8)
            pqw = _gather_component(state_components, parent, 6, 8)
            ps = _gather_component(state_components, parent, 7, 8)
        else:
            pqx = zero
            pqy = zero
            pqz = zero
            pqw = zero + 1.0
            ps = zero + 1.0

        (
            grad_parent_tx,
            grad_parent_ty,
            grad_parent_tz,
            grad_parent_qx,
            grad_parent_qy,
            grad_parent_qz,
            grad_parent_qw,
            grad_parent_s,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
            grad_local_s,
        ) = _compute_backward_gradients(
            pqx,
            pqy,
            pqz,
            pqw,
            ps,
            ltx,
            lty,
            ltz,
            lqx,
            lqy,
            lqz,
            lqw,
            ls,
            gtx,
            gty,
            gtz,
            gqx,
            gqy,
            gqz,
            gqw,
            gs,
            active_joints[joint],
        )

        if parent >= 0:
            grad_work_tx[parent] = grad_work_tx[parent] + grad_parent_tx
            grad_work_ty[parent] = grad_work_ty[parent] + grad_parent_ty
            grad_work_tz[parent] = grad_work_tz[parent] + grad_parent_tz
            grad_work_qx[parent] = grad_work_qx[parent] + grad_parent_qx
            grad_work_qy[parent] = grad_work_qy[parent] + grad_parent_qy
            grad_work_qz[parent] = grad_work_qz[parent] + grad_parent_qz
            grad_work_qw[parent] = grad_work_qw[parent] + grad_parent_qw
            grad_work_s[parent] = grad_work_s[parent] + grad_parent_s

        if active_joints[joint]:
            rx = _gather_component(param_components, joint, 3, 7)
            ry = _gather_component(param_components, joint, 4, 7)
            rz = _gather_component(param_components, joint, 5, 7)
            eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)
            pqx_pre, pqy_pre, pqz_pre, pqw_pre = prerotations[joint]
            _, _, _, _, grad_eqx, grad_eqy, grad_eqz, grad_eqw = (
                _quat_multiply_backward(
                    pqx_pre,
                    pqy_pre,
                    pqz_pre,
                    pqw_pre,
                    eqx,
                    eqy,
                    eqz,
                    eqw,
                    grad_local_qx,
                    grad_local_qy,
                    grad_local_qz,
                    grad_local_qw,
                )
            )
            grad_rx, grad_ry, grad_rz = _euler_backward(
                rx, ry, rz, grad_eqx, grad_eqy, grad_eqz, grad_eqw
            )
            grad_params[joint] = (
                grad_local_tx,
                grad_local_ty,
                grad_local_tz,
                grad_rx,
                grad_ry,
                grad_rz,
                grad_local_s * ls * _LN2,
            )

    outputs = []
    for joint in range(num_joints):
        joint_grad_params = grad_params[joint]
        if joint_grad_params is None:
            if zero_grad_params is None:
                zero_grad_params = (zero,) * 7
            joint_grad_params = zero_grad_params
        outputs.extend(joint_grad_params)
    return tuple(outputs)


def _backward_jit_key(
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
    active_joints: tuple[bool, ...],
) -> tuple[Any, ...]:
    return (offsets, prerotations, parents, active_joints)


def _backward_jit(
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
    active_joints: tuple[bool, ...],
):
    assert fx is not None
    key = _backward_jit_key(offsets, prerotations, parents, active_joints)
    cached = _BACKWARD_JIT_CACHE.get(key)
    if cached is not None:
        return cached
    num_joints = len(parents)
    param_component_count = num_joints * 7
    state_component_count = num_joints * 8

    @fx.jit(block_dim=128)
    def backward_kernel(*components):
        param_components = tuple(components[:param_component_count])
        state_components = tuple(
            components[
                param_component_count : param_component_count + state_component_count
            ]
        )
        grad_output_components = tuple(
            components[param_component_count + state_component_count :]
        )
        return _backward_components(
            param_components,
            state_components,
            grad_output_components,
            num_joints=num_joints,
            offsets=offsets,
            prerotations=prerotations,
            parents=parents,
            active_joints=active_joints,
        )

    _bounded_cache_set(
        _BACKWARD_JIT_CACHE, key, backward_kernel, _MAX_SMALL_CACHE_ENTRIES
    )
    return backward_kernel


def _backward_array_ops(  # noqa: C901
    joint_parameters: torch.Tensor,
    global_state: torch.Tensor,
    grad_global_state: torch.Tensor,
    global_state_packed: torch.Tensor | None,
    grad_global_state_packed: torch.Tensor | None,
    global_state_flux: Any | None,
    offsets: tuple[tuple[float, ...], ...],
    prerotations: tuple[tuple[float, ...], ...],
    parents: tuple[int, ...],
    active_joints: tuple[bool, ...],
) -> torch.Tensor:
    assert fx is not None
    num_joints = joint_parameters.shape[-2]
    batch_size = joint_parameters.numel() // (num_joints * 7)
    joint_params = _torch_to_flux(joint_parameters.reshape(-1), cache=True)
    if global_state_flux is not None:
        global_state_layout = _FLUX_ROW_MAJOR_STATE
        if isinstance(global_state_flux, tuple):
            global_state_buffer, global_state_layout = global_state_flux
        else:
            global_state_buffer = global_state_flux
        if global_state_layout == _FLUX_COMPONENT_MAJOR_STATE:
            state_component_offsets = _component_major_offsets(
                batch_size=batch_size,
                num_joints=num_joints,
                components_per_joint=8,
            )
            state_component_stride = 1
        else:
            state_component_offsets = _strided_offsets(
                num_joints=num_joints,
                components_per_joint=8,
            )
            state_component_stride = num_joints * 8
    elif global_state_packed is not None:
        global_state_buffer = _torch_to_flux(global_state_packed.reshape(-1))
        state_component_offsets = _component_major_offsets(
            batch_size=batch_size,
            num_joints=num_joints,
            components_per_joint=8,
        )
        state_component_stride = 1
    else:
        global_state_buffer = _torch_to_flux(global_state.reshape(-1))
        state_component_offsets = _strided_offsets(
            num_joints=num_joints,
            components_per_joint=8,
        )
        state_component_stride = num_joints * 8
    if grad_global_state_packed is not None:
        grad_output_buffer = _torch_to_flux(
            grad_global_state_packed.reshape(-1), cache=True
        )
        grad_component_offsets = _component_major_offsets(
            batch_size=batch_size,
            num_joints=num_joints,
            components_per_joint=8,
        )
        grad_component_stride = 1
    else:
        grad_output_buffer = _torch_to_flux(grad_global_state.reshape(-1), cache=True)
        grad_component_offsets = _strided_offsets(
            num_joints=num_joints,
            components_per_joint=8,
        )
        grad_component_stride = num_joints * 8
    outputs = None
    param_components = None
    state_components = None
    grad_output_components = None
    if _should_use_backward_jit(num_joints) and hasattr(fx, "jit"):
        kernel = _backward_jit(offsets, prerotations, parents, active_joints)
        if _use_packed_jit_output():
            layout = _backward_strided_call_layout(
                num_joints=num_joints,
                state_component_offsets=state_component_offsets,
                state_component_stride=state_component_stride,
                grad_component_offsets=grad_component_offsets,
                grad_component_stride=grad_component_stride,
            )
            row_major_layout = getattr(
                kernel, "call_row_major_outputs_strided_layout", None
            )
            if layout is not None and row_major_layout is not None:
                try:
                    packed = row_major_layout(
                        [joint_params, global_state_buffer, grad_output_buffer],
                        layout,
                        batch_size,
                    )
                    if packed is not None:
                        return _row_major_packed_to_torch(
                            packed,
                            shape=tuple(joint_parameters.shape),
                        )
                except (RuntimeError, TypeError, ValueError):
                    pass

            row_major_strided = getattr(kernel, "call_row_major_outputs_strided", None)
            if row_major_strided is not None:
                base_indices, all_offsets, all_strides = _backward_strided_call_args(
                    num_joints=num_joints,
                    state_component_offsets=state_component_offsets,
                    state_component_stride=state_component_stride,
                    grad_component_offsets=grad_component_offsets,
                    grad_component_stride=grad_component_stride,
                )
                try:
                    packed = row_major_strided(
                        [joint_params, global_state_buffer, grad_output_buffer],
                        base_indices,
                        all_offsets,
                        all_strides,
                        batch_size,
                    )
                    if packed is not None:
                        return _row_major_packed_to_torch(
                            packed,
                            shape=tuple(joint_parameters.shape),
                        )
                except (RuntimeError, TypeError, ValueError):
                    pass
            packed_layout = getattr(kernel, "call_packed_outputs_strided_layout", None)
            if layout is not None and packed_layout is not None:
                try:
                    packed = packed_layout(
                        [joint_params, global_state_buffer, grad_output_buffer],
                        layout,
                        batch_size,
                    )
                    if packed is not None:
                        return _packed_columns_to_torch(
                            packed,
                            batch_size=batch_size,
                            num_joints=num_joints,
                            components_per_joint=7,
                            shape=tuple(joint_parameters.shape),
                        )
                except (RuntimeError, TypeError, ValueError):
                    pass
            packed_strided = getattr(kernel, "call_packed_outputs_strided", None)
            if packed_strided is not None:
                base_indices, all_offsets, all_strides = _backward_strided_call_args(
                    num_joints=num_joints,
                    state_component_offsets=state_component_offsets,
                    state_component_stride=state_component_stride,
                    grad_component_offsets=grad_component_offsets,
                    grad_component_stride=grad_component_stride,
                )
                try:
                    packed = packed_strided(
                        [joint_params, global_state_buffer, grad_output_buffer],
                        base_indices,
                        all_offsets,
                        all_strides,
                        batch_size,
                    )
                    if packed is not None:
                        return _packed_columns_to_torch(
                            packed,
                            batch_size=batch_size,
                            num_joints=num_joints,
                            components_per_joint=7,
                            shape=tuple(joint_parameters.shape),
                        )
                except (RuntimeError, TypeError, ValueError):
                    pass
            param_components = _gather_components(
                joint_params,
                batch_size=batch_size,
                num_joints=num_joints,
                components_per_joint=7,
                device=joint_parameters.device,
            )
            state_components = (
                _gather_components(
                    global_state_buffer,
                    batch_size=batch_size,
                    num_joints=num_joints,
                    components_per_joint=8,
                    device=joint_parameters.device,
                )
                if global_state_packed is None
                else _gather_strided_components(
                    global_state_buffer,
                    offsets=state_component_offsets,
                    stride=state_component_stride,
                    batch_size=batch_size,
                    device=joint_parameters.device,
                )
            )
            grad_output_components = (
                _gather_components(
                    grad_output_buffer,
                    batch_size=batch_size,
                    num_joints=num_joints,
                    components_per_joint=8,
                    device=joint_parameters.device,
                )
                if grad_global_state_packed is None
                else _gather_strided_components(
                    grad_output_buffer,
                    offsets=grad_component_offsets,
                    stride=grad_component_stride,
                    batch_size=batch_size,
                    device=joint_parameters.device,
                )
            )
            packed_outputs = getattr(kernel, "call_packed_outputs", None)
            if packed_outputs is not None:
                try:
                    packed = packed_outputs(
                        *(param_components + state_components + grad_output_components)
                    )
                    if packed is not None:
                        return _packed_columns_to_torch(
                            packed,
                            batch_size=batch_size,
                            num_joints=num_joints,
                            components_per_joint=7,
                            shape=tuple(joint_parameters.shape),
                        )
                except (RuntimeError, TypeError, ValueError):
                    pass
        try:
            if (
                param_components is None
                or state_components is None
                or grad_output_components is None
            ):
                param_components = _gather_components(
                    joint_params,
                    batch_size=batch_size,
                    num_joints=num_joints,
                    components_per_joint=7,
                    device=joint_parameters.device,
                )
                state_components = (
                    _gather_components(
                        global_state_buffer,
                        batch_size=batch_size,
                        num_joints=num_joints,
                        components_per_joint=8,
                        device=joint_parameters.device,
                    )
                    if global_state_packed is None
                    else _gather_strided_components(
                        global_state_buffer,
                        offsets=state_component_offsets,
                        stride=state_component_stride,
                        batch_size=batch_size,
                        device=joint_parameters.device,
                    )
                )
                grad_output_components = (
                    _gather_components(
                        grad_output_buffer,
                        batch_size=batch_size,
                        num_joints=num_joints,
                        components_per_joint=8,
                        device=joint_parameters.device,
                    )
                    if grad_global_state_packed is None
                    else _gather_strided_components(
                        grad_output_buffer,
                        offsets=grad_component_offsets,
                        stride=grad_component_stride,
                        batch_size=batch_size,
                        device=joint_parameters.device,
                    )
                )
            outputs = tuple(
                kernel(*(param_components + state_components + grad_output_components))
            )
        except (RuntimeError, TypeError, ValueError):
            outputs = None
    if outputs is None:
        if (
            param_components is None
            or state_components is None
            or grad_output_components is None
        ):
            param_components = _gather_components(
                joint_params,
                batch_size=batch_size,
                num_joints=num_joints,
                components_per_joint=7,
                device=joint_parameters.device,
            )
            state_components = (
                _gather_components(
                    global_state_buffer,
                    batch_size=batch_size,
                    num_joints=num_joints,
                    components_per_joint=8,
                    device=joint_parameters.device,
                )
                if global_state_packed is None
                else _gather_strided_components(
                    global_state_buffer,
                    offsets=state_component_offsets,
                    stride=state_component_stride,
                    batch_size=batch_size,
                    device=joint_parameters.device,
                )
            )
            grad_output_components = (
                _gather_components(
                    grad_output_buffer,
                    batch_size=batch_size,
                    num_joints=num_joints,
                    components_per_joint=8,
                    device=joint_parameters.device,
                )
                if grad_global_state_packed is None
                else _gather_strided_components(
                    grad_output_buffer,
                    offsets=grad_component_offsets,
                    stride=grad_component_stride,
                    batch_size=batch_size,
                    device=joint_parameters.device,
                )
            )
        outputs = _backward_components(
            param_components,
            state_components,
            grad_output_components,
            num_joints=num_joints,
            offsets=offsets,
            prerotations=prerotations,
            parents=parents,
            active_joints=active_joints,
        )
    return _components_to_torch(
        outputs,
        batch_size=batch_size,
        num_joints=num_joints,
        components_per_joint=7,
        shape=tuple(joint_parameters.shape),
    )


def _launch_forward(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
    *,
    return_flux_output: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Any | None]:
    _validate_inputs(
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
        active_joints=None,
    )
    _require_flux_cuda()
    offsets = _float_tuple_2d(joint_translation_offsets)
    prerotations = _float_tuple_2d(joint_prerotations)
    parents = _int_tuple_1d(joint_parents)
    return _forward_array_ops(
        joint_parameters,
        offsets,
        prerotations,
        parents,
        return_flux_output=return_flux_output,
    )


def _launch_backward(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
    global_state: torch.Tensor,
    grad_global_state: torch.Tensor,
    active_joints: torch.Tensor | None,
    global_state_flux: Any | None,
) -> torch.Tensor:
    _validate_inputs(
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
        active_joints,
    )
    _require_flux_cuda()
    num_joints = joint_parameters.shape[-2]
    packed_global_state = _packed_columns_base(
        global_state,
        batch_size=joint_parameters.numel() // (num_joints * 7),
        num_joints=num_joints,
        components_per_joint=8,
    )
    packed_grad_global_state = _packed_columns_base(
        grad_global_state,
        batch_size=joint_parameters.numel() // (num_joints * 7),
        num_joints=num_joints,
        components_per_joint=8,
    )
    if packed_grad_global_state is None and not grad_global_state.is_contiguous():
        grad_global_state = grad_global_state.contiguous()
    if (
        packed_global_state is None
        and global_state_flux is None
        and not global_state.is_contiguous()
    ):
        global_state = global_state.contiguous()
    offsets = _float_tuple_2d(joint_translation_offsets)
    prerotations = _float_tuple_2d(joint_prerotations)
    parents = _int_tuple_1d(joint_parents)
    active = _active_tuple(active_joints, num_joints)
    return _backward_array_ops(
        joint_parameters,
        global_state,
        grad_global_state,
        packed_global_state,
        packed_grad_global_state,
        global_state_flux,
        offsets,
        prerotations,
        parents,
        active,
    )


class _JointParametersToSkeletonState(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        joint_parameters: torch.Tensor,
        joint_translation_offsets: torch.Tensor,
        joint_prerotations: torch.Tensor,
        joint_parents: torch.Tensor,
        active_joints: torch.Tensor | None,
    ) -> torch.Tensor:
        joint_parameters = joint_parameters.contiguous()
        joint_translation_offsets = joint_translation_offsets.contiguous()
        joint_prerotations = joint_prerotations.contiguous()
        joint_parents = joint_parents.contiguous()
        output, output_flux = _launch_forward(
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            return_flux_output=True,
        )
        ctx.save_for_backward(
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            output,
        )
        ctx.active_joints = (
            active_joints.contiguous() if active_joints is not None else None
        )
        ctx.global_state_flux = output_flux
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None, None, None]:
        (
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            global_state,
        ) = ctx.saved_tensors
        grad_joint_parameters = _launch_backward(
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            global_state,
            grad_output,
            ctx.active_joints,
            ctx.global_state_flux,
        )
        return grad_joint_parameters, None, None, None, None


def joint_parameters_to_skeleton_state(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
    active_joints: torch.Tensor | None = None,
) -> torch.Tensor:
    return _JointParametersToSkeletonState.apply(
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
        active_joints,
    )
