# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import types
from collections.abc import Buffer
from typing import cast

import numpy as np
import numpy.typing as npt
import pymomentum.camera as pym_camera  # @manual=:camera
import pymomentum.renderer as pym_renderer  # @manual=:renderer
import torch

# Keep Torch outputs compatible with the CPU rasterizer's 64-byte SIMD rows.
_BUFFER_ALIGNMENT = 16


def _load_renderer_cuda() -> types.ModuleType:
    try:
        return importlib.import_module("pymomentum._renderer_cuda")
    except ImportError as error:
        raise RuntimeError(
            "CUDA rasterization support is not available in this PyMomentum build"
        ) from error


def _padded_width(camera: pym_camera.Camera) -> int:
    return (
        (camera.image_width + _BUFFER_ALIGNMENT - 1) // _BUFFER_ALIGNMENT
    ) * _BUFFER_ALIGNMENT


def create_z_buffer(
    camera: pym_camera.Camera,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a padded float32 depth buffer on the requested device.

    The buffer is initialized to the largest finite float32 value. Its storage
    width may be larger than ``camera.image_width`` to provide the same
    64-byte-aligned row layout on CPU and CUDA.

    :param camera: Camera defining the output dimensions.
    :param device: Torch device for the result. Defaults to CPU.
    :return: A ``(height, width)`` float32 tensor backed by padded storage.
    """
    storage = torch.full(
        (camera.image_height, _padded_width(camera)),
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
        device=torch.device("cpu") if device is None else device,
    )
    return storage[:, : camera.image_width]


def create_index_buffer(
    camera: pym_camera.Camera,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a padded int32 triangle-index buffer on the requested device.

    :param camera: Camera defining the output dimensions.
    :param device: Torch device for the result. Defaults to CPU.
    :return: A ``(height, width)`` int32 tensor backed by padded storage and initialized to ``-1``.
    """
    storage = torch.full(
        (camera.image_height, _padded_width(camera)),
        -1,
        dtype=torch.int32,
        device=torch.device("cpu") if device is None else device,
    )
    return storage[:, : camera.image_width]


def _validate_mesh_tensor(
    tensor: torch.Tensor,
    name: str,
    dtype: torch.dtype,
) -> None:
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise RuntimeError(f"{name} must have shape (n, 3)")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _validate_output_tensor(
    tensor: torch.Tensor,
    name: str,
    camera: pym_camera.Camera,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on device {device}")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != 2 or tuple(tensor.shape) != (
        camera.image_height,
        camera.image_width,
    ):
        raise RuntimeError(
            f"{name} must have shape ({camera.image_height}, {camera.image_width})"
        )
    if (
        tensor.stride(1) != 1
        or tensor.stride(0) < camera.image_width
        or tensor.stride(0) % _BUFFER_ALIGNMENT != 0
        or tensor.storage_offset() % _BUFFER_ALIGNMENT != 0
    ):
        raise RuntimeError(
            f"{name} must use padded storage with aligned rows and a row stride "
            f"that is a multiple of {_BUFFER_ALIGNMENT}"
        )
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    available_elements = storage_elements - tensor.storage_offset()
    required_elements = camera.image_height * tensor.stride(0)
    if available_elements < required_elements:
        raise RuntimeError(
            f"{name} backing storage has {available_elements} elements from the data pointer; "
            f"at least {required_elements} are required by its shape and strides"
        )


def rasterize_mesh(
    vertex_positions: torch.Tensor,
    triangles: torch.Tensor,
    camera: pym_camera.Camera,
    *,
    z_buffer: torch.Tensor | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize a mesh into Torch depth and triangle-index tensors.

    This operation is non-differentiable. It uses the optimized CPU rasterizer
    for CPU tensors and the native CUDA rasterizer for CUDA tensors. Every Torch
    input and output must be on the same device. CUDA work is enqueued on the
    current PyTorch stream; callers must establish ordering with tensors
    produced on other streams.

    :param vertex_positions: Contiguous float32 tensor with shape ``(n, 3)``.
    :param triangles: Contiguous int32 tensor with shape ``(m, 3)``.
    :param camera: Camera used to transform and project vertices.
    :param z_buffer: Optional padded float32 depth buffer to update in place.
    :param triangle_index_buffer: Optional padded int32 index buffer to update in place.
    :param model_matrix: Optional host-side float32 model transformation matrix.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :return: The depth and triangle-index tensors.
    """
    _validate_mesh_tensor(vertex_positions, "vertex_positions", torch.float32)
    _validate_mesh_tensor(triangles, "triangles", torch.int32)
    if triangles.device != vertex_positions.device:
        raise RuntimeError("triangles must be on the same device as vertex_positions")
    if vertex_positions.device.type not in ("cpu", "cuda"):
        raise RuntimeError("vertex_positions must be on a CPU or CUDA device")
    if near_clip <= 0.0:
        raise RuntimeError("near_clip must be positive")

    device = vertex_positions.device
    depth = z_buffer if z_buffer is not None else create_z_buffer(camera, device=device)
    triangle_index = (
        triangle_index_buffer
        if triangle_index_buffer is not None
        else create_index_buffer(camera, device=device)
    )
    _validate_output_tensor(depth, "z_buffer", camera, device, torch.float32)
    _validate_output_tensor(
        triangle_index, "triangle_index_buffer", camera, device, torch.int32
    )
    if depth.shape != triangle_index.shape or depth.stride() != triangle_index.stride():
        raise RuntimeError(
            "z_buffer and triangle_index_buffer must have the same shape and strides"
        )

    if device.type == "cuda":
        return _load_renderer_cuda().rasterize_mesh(
            vertex_positions,
            triangles,
            camera,
            z_buffer=depth,
            triangle_index_buffer=triangle_index,
            model_matrix=model_matrix,
            back_face_culling=back_face_culling,
            near_clip=near_clip,
            depth_offset=depth_offset,
        )

    pym_renderer.rasterize_mesh(
        vertex_positions.detach().numpy(),
        None,
        triangles.detach().numpy(),
        camera,
        cast(Buffer, depth.detach().numpy()),
        triangle_index_buffer=cast(Buffer, triangle_index.detach().numpy()),
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )
    return depth, triangle_index
