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
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a padded float32 depth buffer on the requested device.

    The buffer is initialized to the largest finite float32 value. Its storage
    width may be larger than ``camera.image_width`` to provide the same
    64-byte-aligned row layout on CPU and CUDA.

    :param camera: Camera defining the output dimensions.
    :param batch_size: Optional positive batch dimension.
    :param device: Torch device for the result. Defaults to CPU.
    :return: A ``(height, width)`` or ``(batch, height, width)`` float32 tensor backed by padded storage.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_shape = (camera.image_height, _padded_width(camera))
    storage_shape = image_shape if batch_size is None else (batch_size, *image_shape)
    storage = torch.full(
        storage_shape,
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
        device=torch.device("cpu") if device is None else device,
    )
    return storage[..., : camera.image_width]


def create_index_buffer(
    camera: pym_camera.Camera,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a padded int32 triangle-index buffer on the requested device.

    :param camera: Camera defining the output dimensions.
    :param batch_size: Optional positive batch dimension.
    :param device: Torch device for the result. Defaults to CPU.
    :return: A ``(height, width)`` or ``(batch, height, width)`` int32 tensor backed by padded storage and initialized to ``-1``.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_shape = (camera.image_height, _padded_width(camera))
    storage_shape = image_shape if batch_size is None else (batch_size, *image_shape)
    storage = torch.full(
        storage_shape,
        -1,
        dtype=torch.int32,
        device=torch.device("cpu") if device is None else device,
    )
    return storage[..., : camera.image_width]


def _validate_vertex_positions(tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise RuntimeError("vertex_positions must have dtype torch.float32")
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != 3:
        raise RuntimeError("vertex_positions must have shape (n, 3) or (batch, n, 3)")
    if tensor.ndim == 3 and tensor.shape[0] == 0:
        raise RuntimeError("vertex_positions batch must be non-empty")
    if not tensor.is_contiguous():
        raise RuntimeError("vertex_positions must be contiguous")


def _validate_triangles(tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise RuntimeError("triangles must have dtype torch.int32")
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise RuntimeError("triangles must have shape (n, 3)")
    if not tensor.is_contiguous():
        raise RuntimeError("triangles must be contiguous")


def _validate_output_tensor(
    tensor: torch.Tensor,
    name: str,
    camera: pym_camera.Camera,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int | None,
) -> None:
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on device {device}")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype}")
    image_shape = (camera.image_height, camera.image_width)
    expected_shape = image_shape if batch_size is None else (batch_size, *image_shape)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(f"{name} must have shape {expected_shape}")
    if (
        tensor.stride(-1) != 1
        or tensor.stride(-2) < camera.image_width
        or tensor.stride(-2) % _BUFFER_ALIGNMENT != 0
        or tensor.storage_offset() % _BUFFER_ALIGNMENT != 0
    ):
        raise RuntimeError(
            f"{name} must use padded storage with aligned rows and a row stride "
            f"that is a multiple of {_BUFFER_ALIGNMENT}"
        )
    row_stride = tensor.stride(-2)
    if batch_size is not None:
        batch_stride = tensor.stride(0)
        if batch_stride < camera.image_height * row_stride:
            raise RuntimeError(f"{name} batch entries overlap")
        if batch_stride % _BUFFER_ALIGNMENT != 0:
            raise RuntimeError(
                f"{name} batch stride must be a multiple of {_BUFFER_ALIGNMENT}"
            )
        required_elements = (
            batch_size - 1
        ) * batch_stride + camera.image_height * row_stride
    else:
        required_elements = camera.image_height * row_stride
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    available_elements = storage_elements - tensor.storage_offset()
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
    """Rasterize one mesh or a batch with shared topology and camera.

    This operation is non-differentiable. It uses the optimized CPU rasterizer
    for CPU tensors and the native CUDA rasterizer for CUDA tensors. A batched
    CUDA input is rendered in three kernel launches. Every Torch input and
    output must be on the same device. CUDA work is enqueued on the current
    PyTorch stream; callers must establish ordering with tensors produced on
    other streams.

    :param vertex_positions: Contiguous float32 tensor with shape ``(n, 3)`` or ``(batch, n, 3)``.
    :param triangles: Contiguous int32 tensor with shape ``(m, 3)`` shared across the batch.
    :param camera: Camera shared across the batch.
    :param z_buffer: Optional padded float32 depth buffer to update in place.
    :param triangle_index_buffer: Optional padded int32 index buffer to update in place.
    :param model_matrix: Optional host-side float32 model transformation matrix shared across the batch.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :return: Depth and triangle-index tensors with leading batch dimension when the input is batched.
    """
    _validate_vertex_positions(vertex_positions)
    _validate_triangles(triangles)
    if triangles.device != vertex_positions.device:
        raise RuntimeError("triangles must be on the same device as vertex_positions")
    if vertex_positions.device.type not in ("cpu", "cuda"):
        raise RuntimeError("vertex_positions must be on a CPU or CUDA device")
    if near_clip <= 0.0:
        raise RuntimeError("near_clip must be positive")

    device = vertex_positions.device
    batch_size = int(vertex_positions.shape[0]) if vertex_positions.ndim == 3 else None
    depth = (
        z_buffer
        if z_buffer is not None
        else create_z_buffer(camera, batch_size=batch_size, device=device)
    )
    triangle_index = (
        triangle_index_buffer
        if triangle_index_buffer is not None
        else create_index_buffer(camera, batch_size=batch_size, device=device)
    )
    _validate_output_tensor(
        depth, "z_buffer", camera, device, torch.float32, batch_size
    )
    _validate_output_tensor(
        triangle_index,
        "triangle_index_buffer",
        camera,
        device,
        torch.int32,
        batch_size,
    )
    if depth.shape != triangle_index.shape or depth.stride() != triangle_index.stride():
        raise RuntimeError(
            "z_buffer and triangle_index_buffer must have the same shape and strides"
        )

    if device.type == "cuda":
        renderer_cuda = _load_renderer_cuda()
        rasterize = (
            renderer_cuda.rasterize_meshes
            if batch_size is not None
            else renderer_cuda.rasterize_mesh
        )
        return rasterize(
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

    triangles_numpy = triangles.detach().numpy()
    if batch_size is not None:
        pym_renderer._rasterize_meshes(
            vertex_positions.detach().numpy(),
            triangles_numpy,
            camera,
            cast(Buffer, depth.detach().numpy()),
            cast(Buffer, triangle_index.detach().numpy()),
            model_matrix=model_matrix,
            back_face_culling=back_face_culling,
            near_clip=near_clip,
            depth_offset=depth_offset,
        )
        return depth, triangle_index

    pym_renderer.rasterize_mesh(
        vertex_positions.detach().numpy(),
        None,
        triangles_numpy,
        camera,
        cast(Buffer, depth.detach().numpy()),
        triangle_index_buffer=cast(Buffer, triangle_index.detach().numpy()),
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )
    return depth, triangle_index
