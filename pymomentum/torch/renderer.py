# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import types
from collections import OrderedDict
from collections.abc import Buffer, Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pymomentum.camera as pym_camera  # @manual=:camera
import pymomentum.geometry as pym_geometry  # @manual=:geometry
import pymomentum.quaternion as pym_quaternion  # @manual=:quaternion
import pymomentum.renderer as pym_renderer  # @manual=:renderer
import torch
from pymomentum.torch.character import Character

# Keep Torch outputs compatible with the CPU rasterizer's 64-byte SIMD rows.
_BUFFER_ALIGNMENT = 16
_EXPANDED_TRIANGLE_CACHE_SIZE = 4


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


def _create_vector_buffer(
    camera: pym_camera.Camera,
    *,
    batch_size: int | None,
    device: torch.device | str | None,
    fill_value: float,
) -> torch.Tensor:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_shape = (camera.image_height, _padded_width(camera), 3)
    storage_shape = image_shape if batch_size is None else (batch_size, *image_shape)
    storage = torch.full(
        storage_shape,
        fill_value,
        dtype=torch.float32,
        device=torch.device("cpu") if device is None else device,
    )
    return storage[..., : camera.image_width, :]


def create_rgb_buffer(
    camera: pym_camera.Camera,
    *,
    background_color: torch.Tensor | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a padded float32 RGB buffer on the requested device.

    :param camera: Camera defining the output dimensions.
    :param background_color: Optional float32 tensor with shape ``(3,)`` on the requested device.
    :param batch_size: Optional positive batch dimension.
    :param device: Torch device for the result. Defaults to CPU.
    :return: A padded ``(height, width, 3)`` or ``(batch, height, width, 3)`` tensor.
    """
    result = _create_vector_buffer(
        camera, batch_size=batch_size, device=device, fill_value=0.0
    )
    if background_color is not None:
        if (
            background_color.dtype != torch.float32
            or background_color.device != result.device
            or tuple(background_color.shape) != (3,)
        ):
            raise RuntimeError(
                "background_color must be a float32 tensor with shape (3,) on the output device"
            )
        result.copy_(background_color)
    return result


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
    return _create_vector_buffer(
        camera, batch_size=batch_size, device=device, fill_value=0.0
    )


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


def _validate_vector_output_buffer(
    tensor: torch.Tensor,
    name: str,
    camera: pym_camera.Camera,
    device: torch.device,
    batch_size: int | None,
    depth: torch.Tensor,
) -> None:
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on device {device}")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} must have dtype torch.float32")
    image_shape = (camera.image_height, camera.image_width, 3)
    expected_shape = image_shape if batch_size is None else (batch_size, *image_shape)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(f"{name} must have shape {expected_shape}")
    expected_strides = (
        (3 * depth.stride(0), 3, 1)
        if batch_size is None
        else (3 * depth.stride(0), 3 * depth.stride(1), 3, 1)
    )
    if tensor.stride() != expected_strides:
        raise RuntimeError(f"{name} must use the same padded storage as z_buffer")
    if tensor.storage_offset() % _BUFFER_ALIGNMENT != 0:
        raise RuntimeError(f"{name} data pointer must be aligned")
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
            f"{name} backing storage has {available_elements} elements from the data pointer; "
            f"at least {required_elements} are required by its shape and strides"
        )


def _validate_optional_mesh_attribute(
    tensor: torch.Tensor | None,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape_suffix: tuple[int, ...],
) -> None:
    if tensor is None:
        return
    if tensor.device != device or tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype} on device {device}")
    if tensor.ndim != len(shape_suffix) + 1 or tuple(tensor.shape[1:]) != shape_suffix:
        raise RuntimeError(f"{name} has an invalid shape")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _material_with_device_textures(
    material: pym_renderer.PhongMaterial,
    diffuse_texture: torch.Tensor | None,
    emissive_texture: torch.Tensor | None,
) -> pym_renderer.PhongMaterial:
    return pym_renderer.PhongMaterial(
        diffuse_color=np.asarray(material.diffuse_color, dtype=np.float32),
        specular_color=np.asarray(material.specular_color, dtype=np.float32),
        specular_exponent=material.specular_exponent,
        emissive_color=np.asarray(material.emissive_color, dtype=np.float32),
        diffuse_texture=(
            diffuse_texture.detach().numpy() if diffuse_texture is not None else None
        ),
        emissive_texture=(
            emissive_texture.detach().numpy() if emissive_texture is not None else None
        ),
    )


def _validate_distinct_output_storage(
    outputs: tuple[tuple[str, torch.Tensor | None], ...],
) -> None:
    present: list[tuple[str, torch.Tensor]] = []
    for name, tensor in outputs:
        if tensor is not None:
            present.append((name, tensor))
    for index, (name, tensor) in enumerate(present):
        storage = tensor.untyped_storage().data_ptr()
        for other_name, other in present[index + 1 :]:
            if storage == other.untyped_storage().data_ptr():
                raise RuntimeError(
                    f"{name} and {other_name} must not share backing storage"
                )


@torch.no_grad()
def rasterize_mesh(  # noqa: C901 - validates and dispatches one cohesive public operation
    vertex_positions: torch.Tensor,
    vertex_normals: torch.Tensor | None,
    triangles: torch.Tensor,
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None = None,
    *,
    surface_normals_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    material: pym_renderer.PhongMaterial | None = None,
    texture_coordinates: torch.Tensor | None = None,
    texture_triangles: torch.Tensor | None = None,
    per_vertex_diffuse_color: torch.Tensor | None = None,
    diffuse_texture: torch.Tensor | None = None,
    emissive_texture: torch.Tensor | None = None,
    lights: list[pym_renderer.Light] | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
) -> None:
    """Rasterize a mesh into caller-owned device buffers.

    This operation mutates ``z_buffer`` and any optional output buffers in place and returns
    nothing. Supplying ``rgb_buffer`` enables built-in Phong shading; omitting it runs only the
    depth pass plus the second pass needed by any requested index or normal outputs. CUDA computes
    perspective-correct barycentric coordinates and eye-space normals transiently, without
    materializing full-frame intermediates. Exact-depth CUDA ties are unspecified and may mix
    independently written RGB channels or optional outputs.

    :param vertex_positions: Float32 vertex positions with shape ``(vertices, 3)`` or
        ``(batch, vertices, 3)``.
    :param vertex_normals: Optional float32 normals matching ``vertex_positions``. Geometric normals
        are used when omitted.
    :param triangles: Shared int32 triangle indices with shape ``(triangles, 3)``.
    :param camera: Camera used to project the mesh.
    :param z_buffer: Required caller-owned depth buffer to update in place.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param surface_normals_buffer: Optional caller-owned eye-space surface-normal output.
    :param vertex_index_buffer: Optional caller-owned closest-vertex output.
    :param triangle_index_buffer: Optional caller-owned triangle-index output.
    :param material: Host-side Phong material configuration. Texture maps must be supplied through
        the tensor arguments rather than embedded NumPy arrays.
    :param texture_coordinates: Optional float32 UV coordinates with shape ``(coordinates, 2)``.
    :param texture_triangles: Optional int32 UV indices matching ``triangles``. Mesh triangle indices
        are used when omitted.
    :param per_vertex_diffuse_color: Optional float32 RGB colors with shape ``(vertices, 3)``, shared across a batch.
    :param diffuse_texture: Optional contiguous float32 tensor with shape ``(height, width, 3)``.
    :param emissive_texture: Optional contiguous float32 tensor with shape ``(height, width, 3)``.
    :param lights: Optional host-side lights in world space. CUDA supports at most 16 explicit lights.
    :param model_matrix: Optional host-side model transformation matrix.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Positive eye-space near clipping distance.
    :param depth_offset: Value added to rasterized depth.
    """
    _validate_vertex_positions(vertex_positions)
    _validate_triangles(triangles)
    device = vertex_positions.device
    if device.type not in ("cpu", "cuda"):
        raise RuntimeError("vertex_positions must be on a CPU or CUDA device")
    if near_clip <= 0.0:
        raise ValueError("near_clip must be positive")
    if triangles.device != device:
        raise RuntimeError("triangles must be on the same device as vertex_positions")
    if vertex_normals is not None:
        _validate_vertex_positions(vertex_normals, "vertex_normals")
        if vertex_normals.device != device:
            raise RuntimeError(
                "vertex_normals must be on the same device as vertex_positions"
            )
        if vertex_normals.shape != vertex_positions.shape:
            raise RuntimeError("vertex_normals must match vertex_positions shape")
    for tensor, name, dtype, suffix in (
        (texture_coordinates, "texture_coordinates", torch.float32, (2,)),
        (texture_triangles, "texture_triangles", torch.int32, (3,)),
        (per_vertex_diffuse_color, "per_vertex_diffuse_color", torch.float32, (3,)),
    ):
        _validate_optional_mesh_attribute(tensor, name, device, dtype, suffix)
    for texture, name in (
        (diffuse_texture, "diffuse_texture"),
        (emissive_texture, "emissive_texture"),
    ):
        if texture is not None and (
            texture.device != device
            or texture.dtype != torch.float32
            or texture.ndim != 3
            or texture.shape[0] == 0
            or texture.shape[1] == 0
            or texture.shape[2] != 3
            or not texture.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be a contiguous float32 tensor with shape (height, width, 3) on device {device}"
            )
    if texture_triangles is not None:
        if texture_triangles.shape != triangles.shape:
            raise RuntimeError("texture_triangles must match triangles shape")
        if texture_coordinates is None:
            raise RuntimeError("texture_triangles require texture_coordinates")
        if bool(
            torch.any(
                (texture_triangles < 0)
                | (texture_triangles >= texture_coordinates.shape[0])
            )
        ):
            raise RuntimeError("texture_triangles contain an out-of-range index")
    if (
        per_vertex_diffuse_color is not None
        and per_vertex_diffuse_color.shape != vertex_positions.shape[-2:]
    ):
        raise RuntimeError("per_vertex_diffuse_color must match unbatched vertex shape")
    if per_vertex_diffuse_color is not None and diffuse_texture is not None:
        raise RuntimeError(
            "per_vertex_diffuse_color and diffuse_texture cannot both be provided"
        )
    if (
        texture_coordinates is not None
        and texture_triangles is None
        and texture_coordinates.shape[0] != vertex_positions.shape[-2]
    ):
        raise RuntimeError(
            "texture_coordinates must match vertices when texture_triangles is omitted"
        )
    if (
        diffuse_texture is not None or emissive_texture is not None
    ) and texture_coordinates is None:
        raise RuntimeError("texture maps require texture_coordinates")

    batch_size = int(vertex_positions.shape[0]) if vertex_positions.ndim == 3 else None
    depth = z_buffer
    _validate_output_tensor(
        depth, "z_buffer", camera, device, torch.float32, batch_size
    )
    if triangle_index_buffer is not None:
        _validate_output_tensor(
            triangle_index_buffer,
            "triangle_index_buffer",
            camera,
            device,
            torch.int32,
            batch_size,
        )
        if (
            triangle_index_buffer.shape != depth.shape
            or triangle_index_buffer.stride() != depth.stride()
        ):
            raise RuntimeError(
                "triangle_index_buffer must have the same shape and strides as z_buffer"
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
        if (
            vertex_index_buffer.shape != depth.shape
            or vertex_index_buffer.stride() != depth.stride()
        ):
            raise RuntimeError(
                "vertex_index_buffer must have the same shape and strides as z_buffer"
            )
    if surface_normals_buffer is not None:
        _validate_vector_output_buffer(
            surface_normals_buffer,
            "surface_normals_buffer",
            camera,
            device,
            batch_size,
            depth,
        )

    rgb = rgb_buffer
    if rgb is not None:
        _validate_vector_output_buffer(
            rgb, "rgb_buffer", camera, device, batch_size, depth
        )
    _validate_distinct_output_storage(
        (
            ("z_buffer", depth),
            ("rgb_buffer", rgb),
            ("surface_normals_buffer", surface_normals_buffer),
            ("vertex_index_buffer", vertex_index_buffer),
            ("triangle_index_buffer", triangle_index_buffer),
        )
    )

    current_material = (
        material if material is not None else pym_renderer.PhongMaterial()
    )
    if current_material.has_texture_map:
        raise RuntimeError(
            "PhongMaterial texture maps are not supported by torch.renderer; pass texture maps as device tensors"
        )

    if device.type == "cpu":
        if (
            batch_size is not None
            and rgb is None
            and vertex_index_buffer is None
            and surface_normals_buffer is None
        ):
            pym_renderer._rasterize_meshes(
                vertex_positions.detach().numpy(),
                triangles.detach().numpy(),
                camera,
                cast(Buffer, depth.detach().numpy()),
                (
                    cast(Buffer, triangle_index_buffer.detach().numpy())
                    if triangle_index_buffer is not None
                    else None
                ),
                model_matrix=model_matrix,
                back_face_culling=back_face_culling,
                near_clip=near_clip,
                depth_offset=depth_offset,
            )
            return

        cpu_material = _material_with_device_textures(
            current_material, diffuse_texture, emissive_texture
        )
        positions = vertex_positions.detach().numpy()
        normals = (
            vertex_normals.detach().numpy() if vertex_normals is not None else None
        )

        def frame_buffer(tensor: torch.Tensor | None, frame: int) -> Buffer | None:
            if tensor is None:
                return None
            frame_tensor = tensor if batch_size is None else tensor[frame]
            return cast(Buffer, frame_tensor.detach().numpy())

        for frame in range(1 if batch_size is None else batch_size):
            pym_renderer.rasterize_mesh(
                positions if batch_size is None else positions[frame],
                normals
                if batch_size is None
                else (normals[frame] if normals is not None else None),
                triangles.detach().numpy(),
                camera,
                cast(
                    Buffer,
                    depth.detach().numpy()
                    if batch_size is None
                    else depth[frame].detach().numpy(),
                ),
                frame_buffer(rgb, frame),
                surface_normals_buffer=frame_buffer(surface_normals_buffer, frame),
                vertex_index_buffer=frame_buffer(vertex_index_buffer, frame),
                triangle_index_buffer=frame_buffer(triangle_index_buffer, frame),
                material=cpu_material,
                texture_coordinates=(
                    texture_coordinates.detach().numpy()
                    if texture_coordinates is not None
                    else None
                ),
                texture_triangles=(
                    texture_triangles.detach().numpy()
                    if texture_triangles is not None
                    else None
                ),
                per_vertex_diffuse_color=(
                    per_vertex_diffuse_color.detach().numpy()
                    if per_vertex_diffuse_color is not None
                    else None
                ),
                lights=lights,
                model_matrix=model_matrix,
                back_face_culling=back_face_culling,
                near_clip=near_clip,
                depth_offset=depth_offset,
            )
        return

    renderer_cuda = _load_renderer_cuda()
    renderer_cuda.rasterize_mesh(
        vertex_positions,
        vertex_normals,
        triangles,
        camera,
        depth,
        rgb,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        material=current_material,
        texture_coordinates=texture_coordinates,
        texture_triangles=texture_triangles,
        per_vertex_diffuse_color=per_vertex_diffuse_color,
        diffuse_texture=diffuse_texture,
        emissive_texture=emissive_texture,
        lights=[] if lights is None else lights,
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


@dataclass(frozen=True)
class _PrimitiveMesh:
    vertices: torch.Tensor
    normals: torch.Tensor | None
    triangles: torch.Tensor


@dataclass(frozen=True)
class _SkeletonTopology:
    active_joint_indices: torch.Tensor
    child_joint_indices: torch.Tensor
    parent_joint_indices: torch.Tensor


def _compute_vertex_normals(
    vertices: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    triangle_indices = triangles.to(dtype=torch.int64)
    triangle_vertices = vertices[..., triangle_indices, :]
    face_normals = torch.linalg.cross(
        triangle_vertices[..., 1, :] - triangle_vertices[..., 0, :],
        triangle_vertices[..., 2, :] - triangle_vertices[..., 0, :],
        dim=-1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(-2, triangle_indices[:, corner], face_normals)
    return normals / torch.clamp_min(
        torch.linalg.vector_norm(normals, dim=-1, keepdim=True), 1e-12
    )


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return result


class RenderState:
    """Cache immutable primitive geometry on one Torch device.

    A render state owns canonical primitive meshes and expanded instance-index
    buffers. Output buffers and per-call geometry remain caller-owned.

    :param device: Device on which cached tensors are stored. Defaults to CPU.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = _canonical_device("cpu" if device is None else device)
        self._primitive_meshes: OrderedDict[tuple[object, ...], _PrimitiveMesh] = (
            OrderedDict()
        )
        self._instance_triangles: OrderedDict[
            tuple[tuple[object, ...], int], torch.Tensor
        ] = OrderedDict()
        self._mesh_group_triangles: OrderedDict[
            tuple[tuple[tuple[object, ...], int], ...], torch.Tensor
        ] = OrderedDict()
        self._skeleton_topology_cache: (
            tuple[tuple[tuple[int, ...], tuple[bool, ...]], _SkeletonTopology] | None
        ) = None

    def clear(self) -> None:
        """Release all cached primitive geometry, topology, and index buffers."""
        self._primitive_meshes.clear()
        self._instance_triangles.clear()
        self._mesh_group_triangles.clear()
        self._skeleton_topology_cache = None

    def _primitive_mesh(
        self,
        key: tuple[object, ...],
        factory: Callable[[], pym_geometry.Mesh],
    ) -> _PrimitiveMesh:
        cached = self._primitive_meshes.get(key)
        if cached is not None:
            self._primitive_meshes.move_to_end(key)
            return cached

        mesh = factory()
        vertices = torch.tensor(
            np.asarray(mesh.vertices), dtype=torch.float32, device=self.device
        ).contiguous()
        triangles = torch.tensor(
            np.asarray(mesh.faces), dtype=torch.int32, device=self.device
        ).contiguous()
        mesh_normals = np.asarray(mesh.normals)
        normals = (
            torch.tensor(
                mesh_normals, dtype=torch.float32, device=self.device
            ).contiguous()
            if mesh_normals.shape == mesh.vertices.shape
            else _compute_vertex_normals(vertices, triangles)
        )
        cached = _PrimitiveMesh(vertices, normals, triangles)
        self._primitive_meshes[key] = cached
        self._primitive_meshes.move_to_end(key)
        while len(self._primitive_meshes) > _EXPANDED_TRIANGLE_CACHE_SIZE:
            self._primitive_meshes.popitem(last=False)
        return cached

    def _triangles_for_instances(
        self,
        key: tuple[object, ...],
        mesh: _PrimitiveMesh,
        instance_count: int,
    ) -> torch.Tensor:
        cache_key = (key, instance_count)
        cached = self._instance_triangles.get(cache_key)
        if cached is not None:
            self._instance_triangles.move_to_end(cache_key)
            return cached

        offsets = (
            torch.arange(instance_count, dtype=torch.int32, device=self.device)
            * mesh.vertices.shape[0]
        )
        cached = (mesh.triangles[None] + offsets[:, None, None]).flatten(0, 1)
        self._instance_triangles[cache_key] = cached
        if len(self._instance_triangles) > _EXPANDED_TRIANGLE_CACHE_SIZE:
            self._instance_triangles.popitem(last=False)
        return cached

    def _triangles_for_mesh_groups(
        self,
        groups: tuple[tuple[tuple[object, ...], _PrimitiveMesh, int], ...],
    ) -> torch.Tensor:
        cache_key = tuple((key, count) for key, _, count in groups)
        cached = self._mesh_group_triangles.get(cache_key)
        if cached is not None:
            self._mesh_group_triangles.move_to_end(cache_key)
            return cached

        triangles = []
        vertex_offset = 0
        for key, mesh, count in groups:
            triangles.append(
                self._triangles_for_instances(key, mesh, count) + vertex_offset
            )
            vertex_offset += count * mesh.vertices.shape[0]
        cached = torch.cat(triangles, dim=0)
        self._mesh_group_triangles[cache_key] = cached
        if len(self._mesh_group_triangles) > _EXPANDED_TRIANGLE_CACHE_SIZE:
            self._mesh_group_triangles.popitem(last=False)
        return cached

    def _skeleton_topology(self, character: Character) -> _SkeletonTopology:
        joint_parents = cast(torch.Tensor, character.skeleton.joint_parents)
        parent_values = tuple(int(parent) for parent in joint_parents.tolist())
        active = tuple(name != "body_world" for name in character.skeleton.joint_names)
        cache_key = (parent_values, active)
        if (
            self._skeleton_topology_cache is not None
            and self._skeleton_topology_cache[0] == cache_key
        ):
            return self._skeleton_topology_cache[1]

        active_joint_indices = [
            index for index, is_active in enumerate(active) if is_active
        ]
        child_joint_indices = []
        parent_joint_indices = []
        for child in active_joint_indices:
            parent = parent_values[child]
            while parent >= 0 and not active[parent]:
                parent = parent_values[parent]
            if parent >= 0:
                child_joint_indices.append(child)
                parent_joint_indices.append(parent)

        topology = _SkeletonTopology(
            active_joint_indices=torch.tensor(
                active_joint_indices, dtype=torch.int64, device=self.device
            ),
            child_joint_indices=torch.tensor(
                child_joint_indices, dtype=torch.int64, device=self.device
            ),
            parent_joint_indices=torch.tensor(
                parent_joint_indices, dtype=torch.int64, device=self.device
            ),
        )
        self._skeleton_topology_cache = (cache_key, topology)
        return topology


def _resolve_render_state(
    device: torch.device, state: RenderState | None
) -> RenderState:
    canonical_device = _canonical_device(device)
    if state is None:
        return RenderState(canonical_device)
    if state.device != canonical_device:
        raise RuntimeError(
            f"RenderState is on device {state.device}, but inputs are on {canonical_device}"
        )
    return state


def _validate_instance_positions(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} must have dtype torch.float32")
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != 3:
        raise RuntimeError(f"{name} must have shape (n, 3) or (batch, n, 3)")
    if tensor.shape[-2] == 0:
        raise RuntimeError(f"{name} must contain at least one instance")


def _broadcast_instance_value(
    value: torch.Tensor | float,
    shape: torch.Size,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32:
            raise RuntimeError(f"{name} must have dtype torch.float32")
        if value.device != device:
            raise RuntimeError(f"{name} must be on device {device}")
        tensor = value
    else:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
    try:
        return torch.broadcast_to(tensor, shape)
    except RuntimeError as error:
        raise RuntimeError(
            f"{name} must be broadcastable to shape {tuple(shape)}"
        ) from error


def _normal_matrix(linear: torch.Tensor) -> torch.Tensor:
    first = linear[..., :, 0]
    second = linear[..., :, 1]
    third = linear[..., :, 2]
    cofactors = torch.stack(
        (
            torch.linalg.cross(second, third),
            torch.linalg.cross(third, first),
            torch.linalg.cross(first, second),
        ),
        dim=-1,
    )
    orientation = torch.where(
        torch.linalg.det(linear) < 0.0,
        -torch.ones((), dtype=linear.dtype, device=linear.device),
        torch.ones((), dtype=linear.dtype, device=linear.device),
    )
    return cofactors * orientation[..., None, None]


def _transform_primitive_mesh(
    mesh: _PrimitiveMesh,
    transforms: torch.Tensor,
    valid_instances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    linear = transforms[..., :3, :3]
    translation = transforms[..., :3, 3]
    vertices = torch.einsum("...ij,vj->...vi", linear, mesh.vertices)
    vertices = vertices + translation[..., None, :]
    vertices = vertices.masked_fill(~valid_instances[..., None, None], float("nan"))

    normals = None
    if mesh.normals is not None:
        normal_matrix = _normal_matrix(linear)
        normals = torch.einsum("...ij,vj->...vi", normal_matrix, mesh.normals)
        normal_length = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        normals = normals / torch.clamp_min(normal_length, 1e-12)
    return vertices.flatten(-3, -2).contiguous(), (
        normals.flatten(-3, -2).contiguous() if normals is not None else None
    )


def _rasterize_primitive_instances(
    key: tuple[object, ...],
    factory: Callable[[], pym_geometry.Mesh],
    transforms: torch.Tensor,
    valid_instances: torch.Tensor,
    camera: pym_camera.Camera,
    *,
    state: RenderState | None,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None,
    surface_normals_buffer: torch.Tensor | None,
    vertex_index_buffer: torch.Tensor | None,
    triangle_index_buffer: torch.Tensor | None,
    color: torch.Tensor | None,
    material: pym_renderer.PhongMaterial | None,
    lights: list[pym_renderer.Light] | None,
    model_matrix: npt.NDArray[np.float32] | None,
    back_face_culling: bool,
    near_clip: float,
    depth_offset: float,
) -> None:
    render_state = _resolve_render_state(transforms.device, state)
    mesh = render_state._primitive_mesh(key, factory)
    vertices, normals = _transform_primitive_mesh(mesh, transforms, valid_instances)
    triangles = render_state._triangles_for_instances(
        key, mesh, int(transforms.shape[-3])
    )
    per_vertex_color = None
    if color is not None:
        expected_shape = (transforms.shape[-3], 3)
        if color.shape != expected_shape:
            raise RuntimeError(f"color must have shared batch shape {expected_shape}")
        if color.dtype != torch.float32 or color.device != transforms.device:
            raise RuntimeError(
                "color must have dtype torch.float32 on the primitive input device"
            )
        if rgb_buffer is not None:
            per_vertex_color = (
                color[:, None, :]
                .expand(color.shape[0], mesh.vertices.shape[0], 3)
                .flatten(0, 1)
                .contiguous()
            )
    rasterize_mesh(
        vertices,
        normals,
        triangles,
        camera,
        z_buffer,
        rgb_buffer,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        material=material,
        per_vertex_diffuse_color=per_vertex_color,
        lights=lights,
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


@torch.no_grad()
def rasterize_spheres(
    center: torch.Tensor,
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None = None,
    surface_normals_buffer: torch.Tensor | None = None,
    radius: torch.Tensor | float = 1.0,
    color: torch.Tensor | None = None,
    material: pym_renderer.PhongMaterial | None = None,
    *,
    state: RenderState | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    lights: list[pym_renderer.Light] | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
    subdivision_level: int = 2,
) -> None:
    """Rasterize spheres using cached canonical geometry.

    Centers may contain one set of instances or a batch of instance sets. The
    canonical sphere is created by :func:`pymomentum.renderer.create_sphere_mesh`
    and copied to the render state's device at most once.

    :param center: Float32 centers with shape ``(n, 3)`` or ``(batch, n, 3)``.
    :param camera: Camera shared by every sphere and batch entry.
    :param z_buffer: Required caller-owned padded depth buffer.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param surface_normals_buffer: Optional padded eye-space normal output buffer.
    :param radius: Float32 scalar or tensor broadcastable to the center shape without its final coordinate dimension.
    :param color: Optional float32 RGB color per sphere with shape ``(n, 3)``, shared across a batch.
    :param material: Optional Phong material. ``color`` overrides its diffuse color per sphere.
    :param state: Optional caller-owned cache. Reuse one across calls to retain device geometry; when omitted, resources are call-local.
    :param triangle_index_buffer: Optional padded triangle-index output buffer.
    :param vertex_index_buffer: Optional padded closest-vertex output buffer.
    :param lights: Optional host-side lights in world space.
    :param model_matrix: Optional host-side model transformation matrix.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :param subdivision_level: Number of recursive icosphere subdivisions.
    """
    _validate_instance_positions(center, "center")
    radii = _broadcast_instance_value(
        radius, center.shape[:-1], center.device, "radius"
    )
    valid = radii > 1e-8
    safe_radii = torch.where(valid, radii, torch.ones_like(radii))
    transforms = (
        torch.eye(4, dtype=torch.float32, device=center.device)
        .expand(*center.shape[:-1], 4, 4)
        .clone()
    )
    transforms[..., :3, :3] *= safe_radii[..., None, None]
    transforms[..., :3, 3] = center
    key = ("sphere", subdivision_level)
    _rasterize_primitive_instances(
        key,
        lambda: pym_renderer.create_sphere_mesh(subdivision_level),
        transforms,
        valid,
        camera,
        state=state,
        z_buffer=z_buffer,
        rgb_buffer=rgb_buffer,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        color=color,
        material=material,
        lights=lights,
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


def _rotation_from_x_axis(
    direction: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    x_axis = direction / length[..., None]
    quaternion_scale = torch.sqrt(torch.clamp_min(2.0 * (1.0 + x_axis[..., 0]), 1e-12))
    quaternion = torch.stack(
        (
            torch.zeros_like(quaternion_scale),
            -x_axis[..., 2] / quaternion_scale,
            x_axis[..., 1] / quaternion_scale,
            0.5 * quaternion_scale,
        ),
        dim=-1,
    )
    quaternion /= torch.clamp_min(
        torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), 1e-12
    )
    rotation = pym_quaternion.to_rotation_matrix(quaternion)

    near_opposite = x_axis[..., 0] < -0.9999
    fallback_y = torch.stack(
        (-x_axis[..., 1], x_axis[..., 0], torch.zeros_like(x_axis[..., 0])),
        dim=-1,
    )
    fallback_y /= torch.clamp_min(
        torch.linalg.vector_norm(fallback_y, dim=-1, keepdim=True), 1e-12
    )
    fallback_z = torch.linalg.cross(x_axis, fallback_y, dim=-1)
    fallback_rotation = torch.stack((x_axis, fallback_y, fallback_z), dim=-1)
    return torch.where(near_opposite[..., None, None], fallback_rotation, rotation)


@torch.no_grad()
def rasterize_cylinders(
    start_position: torch.Tensor,
    end_position: torch.Tensor,
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None = None,
    surface_normals_buffer: torch.Tensor | None = None,
    radius: torch.Tensor | float = 1.0,
    color: torch.Tensor | None = None,
    material: pym_renderer.PhongMaterial | None = None,
    *,
    state: RenderState | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    lights: list[pym_renderer.Light] | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
    length_subdivisions: int = 16,
    radius_subdivisions: int = 16,
) -> None:
    """Rasterize cylinders using cached canonical geometry.

    :param start_position: Float32 starts with shape ``(n, 3)`` or ``(batch, n, 3)``.
    :param end_position: Float32 ends matching ``start_position``.
    :param camera: Camera shared by every cylinder and batch entry.
    :param z_buffer: Required caller-owned padded depth buffer.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param surface_normals_buffer: Optional padded eye-space normal output buffer.
    :param radius: Float32 scalar or tensor broadcastable to the instance dimensions.
    :param color: Optional float32 RGB color per cylinder with shape ``(n, 3)``, shared across a batch.
    :param material: Optional Phong material. ``color`` overrides its diffuse color per cylinder.
    :param state: Optional caller-owned cache. Reuse one across calls to retain device geometry; when omitted, resources are call-local.
    :param triangle_index_buffer: Optional padded triangle-index output buffer.
    :param vertex_index_buffer: Optional padded closest-vertex output buffer.
    :param lights: Optional host-side lights in world space.
    :param model_matrix: Optional host-side model transformation matrix.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :param length_subdivisions: Number of subdivisions along the cylinder length.
    :param radius_subdivisions: Number of subdivisions around the cylinder radius.
    """
    _validate_instance_positions(start_position, "start_position")
    _validate_instance_positions(end_position, "end_position")
    if end_position.shape != start_position.shape:
        raise RuntimeError("end_position must match start_position shape")
    if end_position.device != start_position.device:
        raise RuntimeError("end_position and start_position must be on the same device")

    radii = _broadcast_instance_value(
        radius, start_position.shape[:-1], start_position.device, "radius"
    )
    direction = end_position - start_position
    length = torch.linalg.vector_norm(direction, dim=-1)
    valid = (length > 1e-8) & (radii > 1e-8)
    safe_length = torch.where(valid, length, torch.ones_like(length))
    safe_radius = torch.where(valid, radii, torch.ones_like(radii))
    rotation = _rotation_from_x_axis(direction, safe_length)

    transforms = (
        torch.eye(4, dtype=torch.float32, device=start_position.device)
        .expand(*start_position.shape[:-1], 4, 4)
        .clone()
    )
    scale = torch.stack((safe_length, safe_radius, safe_radius), dim=-1)
    transforms[..., :3, :3] = rotation * scale[..., None, :]
    transforms[..., :3, 3] = start_position
    key = ("cylinder", length_subdivisions, radius_subdivisions)
    _rasterize_primitive_instances(
        key,
        lambda: pym_renderer.create_cylinder_mesh(
            length_subdivisions, radius_subdivisions
        ),
        transforms,
        valid,
        camera,
        state=state,
        z_buffer=z_buffer,
        rgb_buffer=rgb_buffer,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        color=color,
        material=material,
        lights=lights,
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


def _capsule_local_geometry(
    mesh: _PrimitiveMesh,
    radii: torch.Tensor,
    length: torch.Tensor,
    length_subdivisions: int,
    radius_subdivisions: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    body_count = (length_subdivisions + 1) * radius_subdivisions
    cap_count = radius_subdivisions * (radius_subdivisions // 2 + 1)
    referenced_vertex_count = body_count + 2 * cap_count
    if mesh.vertices.shape[0] not in (
        referenced_vertex_count,
        referenced_vertex_count + cap_count,
    ):
        raise RuntimeError(
            "Capsule template vertex layout does not match the requested subdivisions"
        )
    body_template = mesh.vertices[:body_count]
    start_template = mesh.vertices[body_count : body_count + cap_count]
    # makeCapsule appends an unreferenced duplicate of the upper-cap vertices;
    # transform every trailing vertex so this stays aligned with its topology.
    end_template = mesh.vertices[body_count + cap_count :]

    body_fraction = body_template[:, 0]
    body_radius = radii[..., :1] + (radii[..., 1:] - radii[..., :1]) * body_fraction
    body_vertices = torch.stack(
        (
            length[..., None] * body_fraction,
            body_radius * body_template[:, 1],
            body_radius * body_template[:, 2],
        ),
        dim=-1,
    )
    start_vertices = radii[..., :1, None] * start_template
    end_center = torch.zeros_like(end_template)
    end_center[:, 0] = 1.0
    end_vertices = radii[..., 1:, None] * (end_template - end_center)
    end_vertices[..., 0] += length[..., None]
    vertices = torch.cat((body_vertices, start_vertices, end_vertices), dim=-2)

    if mesh.normals is None:
        return vertices, None
    normal_length = torch.where(
        torch.abs(length) > 1e-8, length, torch.ones_like(length)
    )
    normal_x = -(radii[..., 1] - radii[..., 0]) / normal_length
    body_normals = torch.cat(
        (
            normal_x[..., None, None].expand(*normal_x.shape, body_count, 1),
            mesh.normals[:body_count, 1:].expand(*normal_x.shape, body_count, 2),
        ),
        dim=-1,
    )
    body_normals /= torch.clamp_min(
        torch.linalg.vector_norm(body_normals, dim=-1, keepdim=True), 1e-12
    )
    start_normals = mesh.normals[body_count : body_count + cap_count].expand(
        *normal_x.shape, cap_count, 3
    )
    end_normals = mesh.normals[body_count + cap_count :].expand(
        *normal_x.shape, end_template.shape[0], 3
    )
    return vertices, torch.cat((body_normals, start_normals, end_normals), dim=-2)


@torch.no_grad()
def rasterize_capsules(
    transformation: torch.Tensor,
    radius: torch.Tensor,
    length: torch.Tensor,
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None = None,
    surface_normals_buffer: torch.Tensor | None = None,
    material: pym_renderer.PhongMaterial | None = None,
    *,
    state: RenderState | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    lights: list[pym_renderer.Light] | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
    cylinder_length_subdivisions: int = 16,
    cylinder_radius_subdivisions: int = 16,
) -> None:
    """Rasterize parameterized capsules without host-device geometry transfers.

    A unit capsule template supplies cached topology. Vertex positions and
    normals for each capsule are generated on the input tensor's device.

    :param transformation: Float32 local-to-world transforms with shape ``(n, 4, 4)`` or ``(batch, n, 4, 4)``.
    :param radius: Float32 start/end radii matching the instance dimensions and ending in two.
    :param length: Float32 distances between cap centers matching the instance dimensions.
    :param camera: Camera shared by every capsule and batch entry.
    :param z_buffer: Required caller-owned padded depth buffer.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param surface_normals_buffer: Optional padded eye-space normal output buffer.
    :param material: Optional Phong material.
    :param state: Optional caller-owned cache. Reuse one across calls to retain device geometry; when omitted, resources are call-local.
    :param triangle_index_buffer: Optional padded triangle-index output buffer.
    :param vertex_index_buffer: Optional padded closest-vertex output buffer.
    :param lights: Optional host-side lights in world space.
    :param model_matrix: Optional host-side model transformation matrix.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :param cylinder_length_subdivisions: Number of subdivisions along the central body.
    :param cylinder_radius_subdivisions: Number of subdivisions around the capsule radius.
    """
    if transformation.dtype != torch.float32:
        raise RuntimeError("transformation must have dtype torch.float32")
    if transformation.ndim not in (3, 4) or transformation.shape[-2:] != (4, 4):
        raise RuntimeError(
            "transformation must have shape (n, 4, 4) or (batch, n, 4, 4)"
        )
    instance_shape = transformation.shape[:-2]
    if instance_shape[-1] == 0:
        raise RuntimeError("transformation must contain at least one capsule")
    if radius.dtype != torch.float32 or length.dtype != torch.float32:
        raise RuntimeError("radius and length must have dtype torch.float32")
    if radius.device != transformation.device or length.device != transformation.device:
        raise RuntimeError(
            "transformation, radius, and length must be on the same device"
        )
    if radius.shape != (*instance_shape, 2):
        raise RuntimeError(f"radius must have shape {(*instance_shape, 2)}")
    if length.shape != instance_shape:
        raise RuntimeError(f"length must have shape {tuple(instance_shape)}")

    valid = (
        torch.isfinite(length)
        & torch.isfinite(radius).all(dim=-1)
        & (length > 1e-8)
        & (radius >= 0.0).all(dim=-1)
    )
    safe_length = torch.where(valid, length, torch.ones_like(length))
    safe_radius = torch.where(valid[..., None], radius, torch.ones_like(radius))
    render_state = _resolve_render_state(transformation.device, state)
    key = ("capsule", cylinder_length_subdivisions, cylinder_radius_subdivisions)
    mesh = render_state._primitive_mesh(
        key,
        lambda: pym_renderer.create_capsule_mesh(
            cylinder_length_subdivisions, cylinder_radius_subdivisions, 1.0, 1.0, 1.0
        ),
    )
    local_vertices, local_normals = _capsule_local_geometry(
        mesh,
        safe_radius,
        safe_length,
        cylinder_length_subdivisions,
        cylinder_radius_subdivisions,
    )
    linear = transformation[..., :3, :3]
    translation = transformation[..., :3, 3]
    vertices = torch.einsum("...ij,...vj->...vi", linear, local_vertices)
    vertices += translation[..., None, :]
    vertices = vertices.masked_fill(~valid[..., None, None], float("nan"))

    normals = None
    if local_normals is not None:
        normal_matrix = _normal_matrix(linear)
        normals = torch.einsum("...ij,...vj->...vi", normal_matrix, local_normals)
        normals /= torch.clamp_min(
            torch.linalg.vector_norm(normals, dim=-1, keepdim=True), 1e-12
        )
        normals = normals.flatten(-3, -2).contiguous()
    vertices = vertices.flatten(-3, -2).contiguous()
    triangles = render_state._triangles_for_instances(
        key, mesh, int(transformation.shape[-3])
    )
    rasterize_mesh(
        vertices,
        normals,
        triangles,
        camera,
        z_buffer,
        rgb_buffer,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        material=material,
        lights=lights,
        model_matrix=model_matrix,
        back_face_culling=True,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


@torch.no_grad()
def rasterize_transforms(
    transforms: torch.Tensor,
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    *,
    rgb_buffer: torch.Tensor | None = None,
    surface_normals_buffer: torch.Tensor | None = None,
    scale: float = 1.0,
    material: pym_renderer.PhongMaterial | None = None,
    lights: list[pym_renderer.Light] | None = None,
    state: RenderState | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
    length_subdivisions: int = 16,
    radius_subdivisions: int = 16,
) -> None:
    """Rasterize the three axes of each transform as cached arrow meshes.

    The three axes use red, green, and blue diffuse colors by default. Supplying
    ``material`` uses that material for every axis instead.

    :param transforms: Float32 transforms with shape ``(n, 4, 4)`` or ``(batch, n, 4, 4)``.
    :param camera: Camera shared by every transform and batch entry.
    :param z_buffer: Required caller-owned padded depth buffer.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param surface_normals_buffer: Optional padded eye-space normal output buffer.
    :param scale: Uniform arrow scale relative to each transform axis length.
    :param material: Optional shared Phong material, replacing the default axis colors.
    :param lights: Optional host-side lights in world space.
    :param state: Optional caller-owned cache. Reuse one across calls to retain device geometry; when omitted, resources are call-local.
    :param triangle_index_buffer: Optional padded triangle-index output buffer.
    :param vertex_index_buffer: Optional padded closest-vertex output buffer.
    :param model_matrix: Optional host-side model transformation matrix.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :param length_subdivisions: Number of subdivisions along each arrow shaft.
    :param radius_subdivisions: Number of subdivisions around each arrow radius.
    """
    if transforms.dtype != torch.float32:
        raise RuntimeError("transforms must have dtype torch.float32")
    if transforms.ndim not in (3, 4) or transforms.shape[-2:] != (4, 4):
        raise RuntimeError("transforms must have shape (n, 4, 4) or (batch, n, 4, 4)")
    if transforms.shape[-3] == 0:
        raise RuntimeError("transforms must contain at least one transform")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    axes = transforms[..., :3, :3].transpose(-1, -2)
    origin = transforms[..., :3, 3]
    axis_length = torch.linalg.vector_norm(axes, dim=-1)
    valid = axis_length > 1e-8
    safe_length = torch.where(valid, axis_length, torch.ones_like(axis_length))
    rotation = _rotation_from_x_axis(axes, safe_length)
    arrow_scale = scale * safe_length
    arrow_transforms = (
        torch.eye(4, dtype=torch.float32, device=transforms.device)
        .expand(*axes.shape[:-1], 4, 4)
        .clone()
    )
    arrow_transforms[..., :3, :3] = rotation * arrow_scale[..., None, None]
    arrow_transforms[..., :3, 3] = origin[..., None, :]
    arrow_transforms = arrow_transforms.flatten(-4, -3)
    valid = valid.flatten(-2, -1)
    axis_colors = None
    if material is None and rgb_buffer is not None:
        axis_colors = (
            torch.tensor(
                [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]],
                dtype=torch.float32,
                device=transforms.device,
            )
            .expand(transforms.shape[-3], 3, 3)
            .flatten(0, 1)
            .contiguous()
        )

    key = ("arrow", length_subdivisions, radius_subdivisions)
    _rasterize_primitive_instances(
        key,
        lambda: pym_renderer.create_arrow_mesh(
            length_subdivisions, radius_subdivisions
        ),
        arrow_transforms,
        valid,
        camera,
        state=state,
        z_buffer=z_buffer,
        rgb_buffer=rgb_buffer,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        color=axis_colors,
        material=material,
        lights=lights,
        model_matrix=model_matrix,
        back_face_culling=True,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


def _validate_skeleton_inputs(
    character: Character, model_parameters: torch.Tensor
) -> None:
    if not hasattr(character, "parameter_transform"):
        raise RuntimeError("Character has no parameter transform")
    if not hasattr(character, "skeleton"):
        raise RuntimeError("Character has no skeleton")
    if model_parameters.ndim not in (1, 2):
        raise RuntimeError(
            "model_parameters must have shape (n_parameters,) or (batch, n_parameters)"
        )
    if model_parameters.ndim == 2 and model_parameters.shape[0] == 0:
        raise RuntimeError("model_parameters batch must be non-empty")
    if model_parameters.dtype != torch.float32:
        raise RuntimeError("model_parameters must have dtype torch.float32")
    if model_parameters.shape[-1] != len(character.parameter_transform.parameter_names):
        raise RuntimeError("model_parameters has the wrong number of parameters")
    if character.skeleton.joint_parents.device != model_parameters.device:
        raise RuntimeError("Character and model_parameters must be on the same device")


def _sphere_instance_transforms(centers: torch.Tensor, radius: float) -> torch.Tensor:
    transforms = (
        torch.eye(4, dtype=torch.float32, device=centers.device)
        .expand(*centers.shape[:-1], 4, 4)
        .clone()
    )
    transforms[..., :3, :3] *= radius
    transforms[..., :3, 3] = centers
    return transforms


def _skeleton_bone_transforms(
    skeleton_state: torch.Tensor,
    topology: _SkeletonTopology,
    style: pym_renderer.SkeletonStyle,
    cylinder_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    child_state = skeleton_state.index_select(-2, topology.child_joint_indices)
    parent_state = skeleton_state.index_select(-2, topology.parent_joint_indices)
    direction = child_state[..., :3] - parent_state[..., :3]
    length = torch.linalg.vector_norm(direction, dim=-1)
    valid = length >= 1e-6
    safe_length = torch.where(valid, length, torch.ones_like(length))
    x_axis = direction / safe_length[..., None]

    parent_rotation = pym_quaternion.to_rotation_matrix(parent_state[..., 3:7])
    parent_z_axis = parent_rotation[..., :, 2]
    y_axis = torch.linalg.cross(x_axis, parent_z_axis)
    y_length = torch.linalg.vector_norm(y_axis, dim=-1, keepdim=True)
    global_z = torch.zeros_like(x_axis)
    global_z[..., 2] = 1.0
    global_y = torch.zeros_like(x_axis)
    global_y[..., 1] = 1.0
    fallback_reference = torch.where(
        (torch.abs(x_axis[..., 2]) < 0.9)[..., None], global_z, global_y
    )
    fallback_y_axis = torch.linalg.cross(x_axis, fallback_reference)
    y_axis = torch.where(y_length > 1e-6, y_axis, fallback_y_axis)
    y_axis /= torch.clamp_min(
        torch.linalg.vector_norm(y_axis, dim=-1, keepdim=True), 1e-12
    )
    z_axis = torch.linalg.cross(x_axis, y_axis)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)

    transforms = (
        torch.eye(4, dtype=torch.float32, device=skeleton_state.device)
        .expand(*direction.shape[:-1], 4, 4)
        .clone()
    )
    if style == pym_renderer.SkeletonStyle.Octahedrons:
        scale = safe_length[..., None].expand(*safe_length.shape, 3)
    else:
        scale = torch.stack(
            (
                safe_length,
                torch.full_like(safe_length, cylinder_radius),
                torch.full_like(safe_length, cylinder_radius),
            ),
            dim=-1,
        )
    transforms[..., :3, :3] = rotation * scale[..., None, :]
    transforms[..., :3, 3] = parent_state[..., :3]
    return transforms, valid


@torch.no_grad()
def rasterize_skeleton(
    character: Character,
    model_parameters: torch.Tensor,
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None = None,
    *,
    surface_normals_buffer: torch.Tensor | None = None,
    sphere_material: pym_renderer.PhongMaterial | None = None,
    cylinder_material: pym_renderer.PhongMaterial | None = None,
    lights: list[pym_renderer.Light] | None = None,
    backend: str = "auto",
    state: RenderState | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
    sphere_radius: float = 2.0,
    cylinder_radius: float = 1.0,
    sphere_subdivision_level: int = 2,
    cylinder_length_subdivisions: int = 16,
    cylinder_radius_subdivisions: int = 16,
    style: pym_renderer.SkeletonStyle = pym_renderer.SkeletonStyle.Pipes,
) -> None:
    """Pose and rasterize a Torch character skeleton as one combined mesh.

    ``Pipes`` and ``Octahedrons`` use cached canonical sphere and bone meshes.
    All joints except ``body_world`` are active. The mesh backend omits the
    CPU renderer's line overlay for ``Octahedrons`` and does not support the
    screen-space ``Lines`` style.

    :param character: Torch character containing a skeleton and parameter transform.
    :param model_parameters: Float32 tensor with shape ``(n_parameters,)`` or ``(batch, n_parameters)``.
    :param camera: Camera shared by every pose.
    :param z_buffer: Required caller-owned padded depth buffer.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param surface_normals_buffer: Optional padded eye-space normal output buffer.
    :param sphere_material: Optional Phong material for joint spheres.
    :param cylinder_material: Optional Phong material for bones.
    :param lights: Optional host-side lights in world space.
    :param backend: Forward-kinematics backend, normally ``"auto"``, ``"torch"``, or ``"triton"``.
    :param state: Optional caller-owned cache. Reuse one across calls to retain device geometry; when omitted, resources are call-local.
    :param triangle_index_buffer: Optional padded triangle-index output buffer; not supported together with ``rgb_buffer``.
    :param vertex_index_buffer: Optional padded closest-vertex output buffer; not supported together with ``rgb_buffer``.
    :param model_matrix: Optional host-side model transformation matrix shared across the batch.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    :param sphere_radius: Radius of the sphere placed at each active joint.
    :param cylinder_radius: Radius of pipe-style bones.
    :param sphere_subdivision_level: Number of recursive sphere subdivisions.
    :param cylinder_length_subdivisions: Number of subdivisions along pipe bones.
    :param cylinder_radius_subdivisions: Number of subdivisions around pipe bones.
    :param style: ``Pipes`` or ``Octahedrons`` mesh rendering style.
    """
    _validate_skeleton_inputs(character, model_parameters)
    if style == pym_renderer.SkeletonStyle.Lines:
        raise NotImplementedError(
            "SkeletonStyle.Lines requires the future CUDA line-rasterization backend"
        )
    if sphere_radius <= 0.0 or cylinder_radius <= 0.0:
        raise ValueError("sphere_radius and cylinder_radius must be positive")

    render_state = _resolve_render_state(model_parameters.device, state)
    topology = render_state._skeleton_topology(character)
    unbatched = model_parameters.ndim == 1
    batched_parameters = (
        model_parameters.unsqueeze(0) if unbatched else model_parameters
    )
    skeleton_state = character.model_parameters_to_skeleton_state(
        batched_parameters,
        use_double_precision=False,
        backend=backend,
    )
    if unbatched:
        skeleton_state = skeleton_state[0]

    sphere_key = ("sphere", sphere_subdivision_level)
    sphere_mesh = render_state._primitive_mesh(
        sphere_key,
        lambda: pym_renderer.create_sphere_mesh(sphere_subdivision_level),
    )
    active_positions = skeleton_state[..., :3].index_select(
        -2, topology.active_joint_indices
    )
    sphere_transforms = _sphere_instance_transforms(active_positions, sphere_radius)
    sphere_valid = torch.ones(
        sphere_transforms.shape[:-2], dtype=torch.bool, device=model_parameters.device
    )
    sphere_vertices, sphere_normals = _transform_primitive_mesh(
        sphere_mesh, sphere_transforms, sphere_valid
    )

    if style == pym_renderer.SkeletonStyle.Octahedrons:
        bone_key: tuple[object, ...] = ("octahedron", 0.15, 0.1)
        bone_mesh = render_state._primitive_mesh(
            bone_key,
            lambda: pym_renderer.create_octahedron_mesh(0.15, 0.1),
        )
    else:
        bone_key = (
            "cylinder",
            cylinder_length_subdivisions,
            cylinder_radius_subdivisions,
        )
        bone_mesh = render_state._primitive_mesh(
            bone_key,
            lambda: pym_renderer.create_cylinder_mesh(
                cylinder_length_subdivisions, cylinder_radius_subdivisions
            ),
        )
    bone_transforms, bone_valid = _skeleton_bone_transforms(
        skeleton_state, topology, style, cylinder_radius
    )
    bone_vertices, bone_normals = _transform_primitive_mesh(
        bone_mesh, bone_transforms, bone_valid
    )

    if rgb_buffer is not None:
        if triangle_index_buffer is not None or vertex_index_buffer is not None:
            raise RuntimeError(
                "rasterize_skeleton does not support index outputs together with rgb_buffer"
            )
        sphere_triangles = render_state._triangles_for_instances(
            sphere_key, sphere_mesh, int(sphere_transforms.shape[-3])
        )
        rasterize_mesh(
            sphere_vertices,
            sphere_normals,
            sphere_triangles,
            camera,
            z_buffer,
            rgb_buffer,
            surface_normals_buffer=surface_normals_buffer,
            material=sphere_material,
            lights=lights,
            model_matrix=model_matrix,
            back_face_culling=back_face_culling,
            near_clip=near_clip,
            depth_offset=depth_offset,
        )
        bone_triangles = render_state._triangles_for_instances(
            bone_key, bone_mesh, int(bone_transforms.shape[-3])
        )
        rasterize_mesh(
            bone_vertices,
            bone_normals,
            bone_triangles,
            camera,
            z_buffer,
            rgb_buffer,
            surface_normals_buffer=surface_normals_buffer,
            material=cylinder_material,
            lights=lights,
            model_matrix=model_matrix,
            back_face_culling=back_face_culling,
            near_clip=near_clip,
            depth_offset=depth_offset,
        )
        return

    vertices = torch.cat((sphere_vertices, bone_vertices), dim=-2)
    normals = (
        torch.cat((sphere_normals, bone_normals), dim=-2)
        if sphere_normals is not None and bone_normals is not None
        else None
    )
    triangles = render_state._triangles_for_mesh_groups(
        (
            (sphere_key, sphere_mesh, int(sphere_transforms.shape[-3])),
            (bone_key, bone_mesh, int(bone_transforms.shape[-3])),
        )
    )
    rasterize_mesh(
        vertices,
        normals,
        triangles,
        camera,
        z_buffer,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


def _validate_character_inputs(
    character: Character, model_parameters: torch.Tensor
) -> None:
    if model_parameters.ndim not in (1, 2):
        raise RuntimeError(
            "model_parameters must have shape (n_parameters,) or (batch, n_parameters)"
        )
    if model_parameters.ndim == 2 and model_parameters.shape[0] == 0:
        raise RuntimeError("model_parameters batch must be non-empty")
    if model_parameters.dtype != torch.float32:
        raise RuntimeError("model_parameters must have dtype torch.float32")
    for attribute, description in (
        ("parameter_transform", "parameter transform"),
        ("skeleton", "skeleton"),
        ("mesh", "rest mesh"),
        ("linear_blend_skinning", "skinning information"),
    ):
        if not hasattr(character, attribute):
            raise RuntimeError(f"Character has no {description}")
    if model_parameters.shape[-1] != len(character.parameter_transform.parameter_names):
        raise RuntimeError("model_parameters has the wrong number of parameters")
    if cast(torch.Tensor, character.mesh.faces).device != model_parameters.device:
        raise RuntimeError("Character and model_parameters must be on the same device")


def _character_rest_vertices(
    character: Character,
    model_parameters: torch.Tensor,
    rest_vertex_positions: torch.Tensor | None,
) -> torch.Tensor:
    if rest_vertex_positions is None:
        if hasattr(character, "blend_shape") or hasattr(
            character, "face_expression_blend_shape"
        ):
            raise RuntimeError(
                "Character has blend shapes; pass fully shaped rest_vertex_positions "
                "explicitly so shape parameters are not ignored"
            )
        rest_vertex_positions = cast(torch.Tensor, character.mesh.rest_vertices)

    if rest_vertex_positions.dtype != torch.float32:
        raise RuntimeError("rest_vertex_positions must have dtype torch.float32")
    if rest_vertex_positions.device != model_parameters.device:
        raise RuntimeError(
            "rest_vertex_positions and model_parameters must be on the same device"
        )
    if rest_vertex_positions.ndim not in (2, 3) or rest_vertex_positions.shape[-1] != 3:
        raise RuntimeError(
            "rest_vertex_positions must have shape (n_vertices, 3) or "
            "(batch, n_vertices, 3)"
        )
    expected_batch_size = 1 if model_parameters.ndim == 1 else model_parameters.shape[0]
    if (
        rest_vertex_positions.ndim == 3
        and rest_vertex_positions.shape[0] != expected_batch_size
    ):
        raise RuntimeError("rest_vertex_positions batch must match model_parameters")
    return rest_vertex_positions


@torch.no_grad()
def rasterize_character(
    character: Character,
    model_parameters: torch.Tensor,
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None = None,
    *,
    backend: str = "auto",
    rest_vertex_positions: torch.Tensor | None = None,
    surface_normals_buffer: torch.Tensor | None = None,
    vertex_index_buffer: torch.Tensor | None = None,
    triangle_index_buffer: torch.Tensor | None = None,
    material: pym_renderer.PhongMaterial | None = None,
    per_vertex_diffuse_color: torch.Tensor | None = None,
    diffuse_texture: torch.Tensor | None = None,
    emissive_texture: torch.Tensor | None = None,
    lights: list[pym_renderer.Light] | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
) -> None:
    """Pose, skin, and rasterize a :class:`pymomentum.torch.character.Character`.

    Model parameters may describe one pose or a batch of poses. Forward
    kinematics and skinning use the requested backend; ``"auto"`` selects the
    Triton implementations for compatible CUDA float32 tensors. Rasterization
    is non-differentiable, so this function does not construct an autograd graph.

    :param character: Torch character containing the skeleton, mesh, parameter transform, and skinning data.
    :param model_parameters: Float32 tensor with shape ``(n_parameters,)`` or ``(batch, n_parameters)``.
    :param camera: Camera shared by every pose.
    :param z_buffer: Required caller-owned padded depth buffer.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param backend: Forward-kinematics and skinning backend, normally ``"auto"``, ``"torch"``, or ``"triton"``.
    :param rest_vertex_positions: Optional fully shaped float32 rest vertices with shape ``(n_vertices, 3)`` or ``(batch, n_vertices, 3)``. Required for characters with body or face blend shapes.
    :param surface_normals_buffer: Optional caller-owned interpolated smooth eye-space normal output.
    :param vertex_index_buffer: Optional caller-owned closest-vertex output.
    :param triangle_index_buffer: Optional caller-owned triangle-index output.
    :param material: Host-side Phong material configuration.
    :param per_vertex_diffuse_color: Optional device-resident per-vertex RGB colors.
    :param diffuse_texture: Optional device-resident diffuse RGB texture.
    :param emissive_texture: Optional device-resident emissive RGB texture.
    :param lights: Optional host-side lights in world space.
    :param model_matrix: Optional host-side float32 model transformation matrix shared across the batch.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Per-pixel near clipping distance.
    :param depth_offset: Value added to rasterized depth before depth testing.
    """
    _validate_character_inputs(character, model_parameters)
    unbatched = model_parameters.ndim == 1
    batched_parameters = (
        model_parameters.unsqueeze(0) if unbatched else model_parameters
    )
    rest_vertices = _character_rest_vertices(
        character, model_parameters, rest_vertex_positions
    )

    skeleton_state = character.model_parameters_to_skeleton_state(
        batched_parameters,
        use_double_precision=False,
        backend=backend,
    )
    vertices = character.skin_points(
        skeleton_state,
        rest_vertices,
        backend=backend,
    ).contiguous()
    if unbatched:
        vertices = vertices[0]

    faces = cast(torch.Tensor, character.mesh.faces)
    needs_normals = rgb_buffer is not None or surface_normals_buffer is not None
    vertex_normals = _compute_vertex_normals(vertices, faces) if needs_normals else None
    texture_coordinates = (
        cast(torch.Tensor, character.mesh.texcoords) if rgb_buffer is not None else None
    )
    texture_triangles = (
        cast(torch.Tensor, character.mesh.texcoord_faces)
        if rgb_buffer is not None
        else None
    )
    rasterize_mesh(
        vertices,
        vertex_normals,
        faces,
        camera,
        z_buffer,
        rgb_buffer,
        surface_normals_buffer=surface_normals_buffer,
        vertex_index_buffer=vertex_index_buffer,
        triangle_index_buffer=triangle_index_buffer,
        material=material,
        texture_coordinates=(
            texture_coordinates
            if texture_coordinates is not None and texture_coordinates.numel() > 0
            else None
        ),
        texture_triangles=(
            texture_triangles
            if texture_triangles is not None and texture_triangles.numel() > 0
            else None
        ),
        per_vertex_diffuse_color=per_vertex_diffuse_color,
        diffuse_texture=diffuse_texture,
        emissive_texture=emissive_texture,
        lights=lights,
        model_matrix=model_matrix,
        back_face_culling=back_face_culling,
        near_clip=near_clip,
        depth_offset=depth_offset,
    )


def _checkerboard_mesh(
    width: float,
    num_checks: int,
    subdivisions: int,
    parity: int,
    device: torch.device,
) -> _PrimitiveMesh:
    positions = []
    triangles = []
    grid_size = num_checks * subdivisions
    width_float32 = np.float32(width)
    coordinates = [
        float(
            np.float32(
                width_float32
                * np.float32(
                    np.float32(index) / np.float32(grid_size) - np.float32(0.5)
                )
            )
        )
        for index in range(grid_size + 1)
    ]
    for check_x in range(num_checks):
        for check_z in range(num_checks):
            if (check_x + check_z) % 2 != parity:
                continue
            offset = len(positions)
            row_size = subdivisions + 1
            for sub_x in range(row_size):
                for sub_z in range(row_size):
                    positions.append(
                        (
                            coordinates[check_x * subdivisions + sub_x],
                            0.0,
                            coordinates[check_z * subdivisions + sub_z],
                        )
                    )
                    if sub_x > 0 and sub_z > 0:
                        current = offset + sub_x * row_size + sub_z
                        previous_row = current - row_size
                        triangles.append((current, previous_row - 1, previous_row))
                        triangles.append((current, current - 1, previous_row - 1))
    vertices = torch.tensor(positions, dtype=torch.float32, device=device)
    normals = torch.zeros_like(vertices)
    normals[:, 1] = 1.0
    return _PrimitiveMesh(
        vertices=vertices,
        normals=normals,
        triangles=torch.tensor(triangles, dtype=torch.int32, device=device),
    )


@torch.no_grad()
def rasterize_checkerboard(
    camera: pym_camera.Camera,
    z_buffer: torch.Tensor,
    rgb_buffer: torch.Tensor | None = None,
    *,
    surface_normals_buffer: torch.Tensor | None = None,
    state: RenderState | None = None,
    material1: pym_renderer.PhongMaterial | None = None,
    material2: pym_renderer.PhongMaterial | None = None,
    lights: list[pym_renderer.Light] | None = None,
    model_matrix: npt.NDArray[np.float32] | None = None,
    back_face_culling: bool = True,
    near_clip: float = 0.1,
    depth_offset: float = 0.0,
    width: float = 50.0,
    num_checks: int = 10,
    subdivisions: int = 1,
) -> None:
    """Rasterize a checkerboard floor into caller-owned buffers.

    Static checker geometry is cached in ``state``. CPU tensors delegate shading to the existing
    renderer; CUDA tensors use native fused Phong shading.

    :param camera: Camera used to project the checkerboard.
    :param z_buffer: Required caller-owned depth buffer.
    :param rgb_buffer: Optional caller-owned RGB buffer. Its presence enables shading.
    :param surface_normals_buffer: Optional caller-owned eye-space surface-normal output.
    :param state: Optional per-device cache for checker geometry.
    :param material1: Material for the first checker parity.
    :param material2: Material for the second checker parity.
    :param lights: Optional host-side lights in world space.
    :param model_matrix: Optional host-side model transformation matrix.
    :param back_face_culling: Whether to discard back-facing triangles.
    :param near_clip: Positive eye-space near clipping distance.
    :param depth_offset: Value added to rasterized depth.
    :param width: Full checkerboard width in model units.
    :param num_checks: Number of checks along each axis.
    :param subdivisions: Triangle subdivisions per check edge.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    if num_checks <= 0:
        raise ValueError("num_checks must be positive")
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")
    output_device = _canonical_device(z_buffer.device)
    if rgb_buffer is not None and rgb_buffer.device != output_device:
        raise RuntimeError("rgb_buffer must be on the same device as z_buffer")
    render_state = _resolve_render_state(output_device, state)
    materials = (
        material1
        if material1 is not None
        else pym_renderer.PhongMaterial(np.full(3, 0.8, dtype=np.float32)),
        material2
        if material2 is not None
        else pym_renderer.PhongMaterial(np.full(3, 0.5, dtype=np.float32)),
    )
    for parity, material in enumerate(materials):
        key = ("checkerboard", width, num_checks, subdivisions, parity)
        mesh = render_state._primitive_meshes.get(key)
        if mesh is None:
            mesh = _checkerboard_mesh(
                width, num_checks, subdivisions, parity, output_device
            )
            render_state._primitive_meshes[key] = mesh
            render_state._primitive_meshes.move_to_end(key)
            while len(render_state._primitive_meshes) > _EXPANDED_TRIANGLE_CACHE_SIZE:
                render_state._primitive_meshes.popitem(last=False)
        else:
            render_state._primitive_meshes.move_to_end(key)
        rasterize_mesh(
            mesh.vertices,
            mesh.normals,
            mesh.triangles,
            camera,
            z_buffer,
            rgb_buffer,
            surface_normals_buffer=surface_normals_buffer,
            material=material,
            lights=lights,
            model_matrix=model_matrix,
            back_face_culling=back_face_culling,
            near_clip=near_clip,
            depth_offset=depth_offset,
        )


@torch.no_grad()
def apply_hard_shadow(
    shadow_depth: torch.Tensor,
    rgb_buffer: torch.Tensor,
    *,
    opacity: float = 0.4,
) -> None:
    """Darken pixels covered by a projected-shadow depth render in place.

    :param shadow_depth: Depth-only rendering of geometry transformed by a shadow projection.
    :param rgb_buffer: Caller-owned RGB buffer to darken.
    :param opacity: Fraction of covered RGB intensity to remove.
    """
    if shadow_depth.device != rgb_buffer.device:
        raise RuntimeError("shadow_depth and rgb_buffer must be on the same device")
    if shadow_depth.dtype != torch.float32 or rgb_buffer.dtype != torch.float32:
        raise RuntimeError("shadow_depth and rgb_buffer must have dtype torch.float32")
    if rgb_buffer.shape != (*shadow_depth.shape, 3):
        raise RuntimeError(
            "rgb_buffer shape must equal shadow_depth shape plus three channels"
        )
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in [0, 1]")
    covered = shadow_depth < torch.finfo(torch.float32).max
    rgb_buffer[covered] *= 1.0 - opacity
