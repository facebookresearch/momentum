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


def create_surface_normals_buffer(
    camera: pym_camera.Camera,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a padded float32 eye-space surface-normal buffer.

    :param camera: Camera defining the output dimensions.
    :param batch_size: Optional positive batch dimension.
    :param device: Torch device for the result. Defaults to CPU.
    :return: A ``(height, width, 3)`` or ``(batch, height, width, 3)`` tensor backed by padded storage.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_shape = (camera.image_height, _padded_width(camera), 3)
    storage_shape = image_shape if batch_size is None else (batch_size, *image_shape)
    storage = torch.zeros(
        storage_shape,
        dtype=torch.float32,
        device=torch.device("cpu") if device is None else device,
    )
    return storage[..., : camera.image_width, :]


def _validate_vertex_positions(
    tensor: torch.Tensor, name: str = "vertex_positions"
) -> None:
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} must have dtype torch.float32")
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != 3:
        raise RuntimeError(f"{name} must have shape (n, 3) or (batch, n, 3)")
    if tensor.ndim == 3 and tensor.shape[0] == 0:
        raise RuntimeError(f"{name} batch must be non-empty")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


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


def _validate_surface_normals_buffer(
    tensor: torch.Tensor,
    camera: pym_camera.Camera,
    device: torch.device,
    batch_size: int | None,
    depth: torch.Tensor,
) -> None:
    if tensor.device != device:
        raise RuntimeError(f"surface_normals_buffer must be on device {device}")
    if tensor.dtype != torch.float32:
        raise RuntimeError("surface_normals_buffer must have dtype torch.float32")
    image_shape = (camera.image_height, camera.image_width, 3)
    expected_shape = image_shape if batch_size is None else (batch_size, *image_shape)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(f"surface_normals_buffer must have shape {expected_shape}")
    expected_strides = (
        (3 * depth.stride(0), 3, 1)
        if batch_size is None
        else (3 * depth.stride(0), 3 * depth.stride(1), 3, 1)
    )
    if tensor.stride() != expected_strides:
        raise RuntimeError(
            "surface_normals_buffer must use the same padded storage as z_buffer"
        )
    if tensor.storage_offset() % _BUFFER_ALIGNMENT != 0:
        raise RuntimeError("surface_normals_buffer data pointer must be aligned")
    row_stride = depth.stride(-2)
    if batch_size is None:
        required_elements = 3 * camera.image_height * row_stride
    else:
        required_elements = 3 * (
            (batch_size - 1) * depth.stride(0) + camera.image_height * row_stride
        )
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    available_elements = storage_elements - tensor.storage_offset()
    if available_elements < required_elements:
        raise RuntimeError(
            f"surface_normals_buffer backing storage has {available_elements} elements from the data pointer; "
            f"at least {required_elements} are required by its shape and strides"
        )


def rasterize_mesh(  # noqa: C901
    vertex_positions: torch.Tensor,
    triangles: torch.Tensor,
    camera: pym_camera.Camera,
    *,
    vertex_normals: torch.Tensor | None = None,
    z_buffer: torch.Tensor | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    surface_normals_buffer: torch.Tensor | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Rasterize one mesh or a batch with shared topology and camera.

    This operation is non-differentiable. It uses the optimized CPU rasterizer
    for CPU tensors and the native CUDA rasterizer for CUDA tensors. A batched
    CUDA always uses one depth pass. It launches a second full rasterization pass
    only when a triangle-index, vertex-index, or surface-normal output buffer is
    provided.
    The second pass accepts fragments that match finalized depth, so outputs
    are coherent except for unspecified exact-depth ties. Every Torch input and
    output must be on the same device. CUDA work is enqueued on the current
    PyTorch stream; callers must establish ordering with tensors produced on
    other streams.

    :param vertex_positions: Contiguous float32 tensor with shape ``(n, 3)`` or ``(batch, n, 3)``.
    :param triangles: Contiguous int32 tensor with shape ``(m, 3)`` shared across the batch.
    :param camera: Camera shared across the batch.
    :param vertex_normals: Optional contiguous float32 normals matching ``vertex_positions``. Face normals are used when omitted.
    :param z_buffer: Optional padded float32 depth buffer to update in place.
    :param triangle_index_buffer: Optional padded int32 triangle-index output. No triangle-index buffer is allocated when omitted.
    :param vertex_index_buffer: Optional padded int32 closest-vertex buffer to update in place.
    :param surface_normals_buffer: Optional padded float32 eye-space normal buffer with a final dimension of three.
    :param model_matrix: Optional host-side float32 model transformation matrix shared across the batch.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :return: Depth and the caller-provided triangle-index tensor, or ``None`` when triangle indices were not requested.
    """
    _validate_vertex_positions(vertex_positions)
    _validate_triangles(triangles)
    if vertex_normals is not None:
        _validate_vertex_positions(vertex_normals, "vertex_normals")
        if vertex_normals.shape != vertex_positions.shape:
            raise RuntimeError("vertex_normals must match vertex_positions shape")
    if triangles.device != vertex_positions.device:
        raise RuntimeError("triangles must be on the same device as vertex_positions")
    if vertex_normals is not None and vertex_normals.device != vertex_positions.device:
        raise RuntimeError(
            "vertex_normals must be on the same device as vertex_positions"
        )
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
    triangle_index = triangle_index_buffer
    _validate_output_tensor(
        depth, "z_buffer", camera, device, torch.float32, batch_size
    )
    if triangle_index is not None:
        _validate_output_tensor(
            triangle_index,
            "triangle_index_buffer",
            camera,
            device,
            torch.int32,
            batch_size,
        )
        if (
            depth.shape != triangle_index.shape
            or depth.stride() != triangle_index.stride()
        ):
            raise RuntimeError(
                "z_buffer and triangle_index_buffer must have the same shape and strides"
            )
    if vertex_index_buffer is not None:
        _validate_output_tensor(
            vertex_index_buffer,
            "vertex_index_buffer",
            camera,
            device,
            torch.int32,
            batch_size,
        )
        if vertex_index_buffer.stride() != depth.stride():
            raise RuntimeError(
                "vertex_index_buffer must have the same strides as z_buffer"
            )
    if surface_normals_buffer is not None:
        _validate_surface_normals_buffer(
            surface_normals_buffer, camera, device, batch_size, depth
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
            vertex_normals=vertex_normals,
            z_buffer=depth,
            triangle_index_buffer=triangle_index,
            vertex_index_buffer=vertex_index_buffer,
            surface_normals_buffer=surface_normals_buffer,
            model_matrix=model_matrix,
            back_face_culling=back_face_culling,
            near_clip=near_clip,
            depth_offset=depth_offset,
        )

    triangles_numpy = triangles.detach().numpy()
    if batch_size is not None:
        if vertex_index_buffer is not None or surface_normals_buffer is not None:
            for frame in range(batch_size):
                pym_renderer.rasterize_mesh(
                    vertex_positions[frame].detach().numpy(),
                    (
                        vertex_normals[frame].detach().numpy()
                        if vertex_normals is not None
                        else None
                    ),
                    triangles_numpy,
                    camera,
                    cast(Buffer, depth[frame].detach().numpy()),
                    surface_normals_buffer=(
                        cast(Buffer, surface_normals_buffer[frame].detach().numpy())
                        if surface_normals_buffer is not None
                        else None
                    ),
                    vertex_index_buffer=(
                        cast(Buffer, vertex_index_buffer[frame].detach().numpy())
                        if vertex_index_buffer is not None
                        else None
                    ),
                    triangle_index_buffer=(
                        cast(Buffer, triangle_index[frame].detach().numpy())
                        if triangle_index is not None
                        else None
                    ),
                    model_matrix=model_matrix,
                    back_face_culling=back_face_culling,
                    near_clip=near_clip,
                    depth_offset=depth_offset,
                )
            return depth, triangle_index

        pym_renderer._rasterize_meshes(
            vertex_positions.detach().numpy(),
            triangles_numpy,
            camera,
            cast(Buffer, depth.detach().numpy()),
            (
                cast(Buffer, triangle_index.detach().numpy())
                if triangle_index is not None
                else None
            ),
            model_matrix=model_matrix,
            back_face_culling=back_face_culling,
            near_clip=near_clip,
            depth_offset=depth_offset,
        )
        return depth, triangle_index

    pym_renderer.rasterize_mesh(
        vertex_positions.detach().numpy(),
        vertex_normals.detach().numpy() if vertex_normals is not None else None,
        triangles_numpy,
        camera,
        cast(Buffer, depth.detach().numpy()),
        surface_normals_buffer=(
            cast(Buffer, surface_normals_buffer.detach().numpy())
            if surface_normals_buffer is not None
            else None
        ),
        vertex_index_buffer=(
            cast(Buffer, vertex_index_buffer.detach().numpy())
            if vertex_index_buffer is not None
            else None
        ),
        triangle_index_buffer=(
            cast(Buffer, triangle_index.detach().numpy())
            if triangle_index is not None
            else None
        ),
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )
    return depth, triangle_index
