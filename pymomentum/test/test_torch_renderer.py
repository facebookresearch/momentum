# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
import pymomentum.camera as pym_camera  # @manual=:camera
import pymomentum.geometry_test_utils as pym_test_utils  # @manual=:geometry_test_utils
import pymomentum.renderer as pym_renderer  # @manual=:renderer
import pymomentum.torch.character as pym_character  # @manual=:gpu_character
import pymomentum.torch.renderer as torch_renderer  # @manual=:torch_renderer
import torch

try:
    import pymomentum._renderer_cuda as pym_renderer_cuda  # @manual=:_renderer_cuda

    _CUDA_BACKEND_AVAILABLE = pym_renderer_cuda is not None
except ImportError:
    _CUDA_BACKEND_AVAILABLE = False


def _make_camera() -> pym_camera.Camera:
    return pym_camera.Camera(
        pym_camera.PinholeIntrinsicsModel(
            image_width=63,
            image_height=48,
            fx=40.0,
            fy=38.0,
            cx=31.5,
            cy=24.0,
        )
    )


def _make_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-1.0, -0.8, 2.0],
            [-1.0, 0.8, 2.0],
            [1.0, -0.8, 2.0],
            [-0.4, -0.4, 1.0],
            [-0.4, 0.4, 1.0],
            [0.4, -0.4, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return vertices, triangles


def _make_character_inputs(
    device: torch.device | str,
) -> tuple[pym_character.Character, torch.Tensor, npt.NDArray[np.float32]]:
    character = pym_character.Character(pym_test_utils.create_test_character()).to(
        device
    )
    parameter_transform = character.parameter_transform.parameter_transform
    active_parameters = torch.nonzero(parameter_transform.ne(0).any(dim=0)).flatten()
    if active_parameters.numel() == 0:
        raise RuntimeError("Test character has no active model parameters")
    model_parameters = torch.zeros(
        (2, parameter_transform.shape[1]), dtype=torch.float32, device=device
    )
    model_parameters[1, int(active_parameters[0])] = 0.25
    model_matrix = np.eye(4, dtype=np.float32)
    model_matrix[2, 3] = 5.0
    return character, model_parameters, model_matrix


class TestTorchRendererCpu(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = _make_camera()
        self.vertices, self.triangles = _make_mesh()

    def test_factories_and_cpu_rasterization(self) -> None:
        depth = torch_renderer.create_z_buffer(self.camera, device="cpu")
        triangle_ids = torch_renderer.create_index_buffer(
            self.camera, device=torch.device("cpu")
        )
        surface_normals = torch_renderer.create_surface_normals_buffer(
            self.camera, device="cpu"
        )
        self.assertEqual(tuple(depth.shape), (48, 63))
        self.assertEqual(tuple(triangle_ids.shape), (48, 63))
        self.assertEqual(tuple(surface_normals.shape), (48, 63, 3))
        self.assertEqual(depth.stride(), (64, 1))
        self.assertEqual(triangle_ids.stride(), (64, 1))
        self.assertEqual(surface_normals.stride(), (64 * 3, 3, 1))
        self.assertEqual(depth.dtype, torch.float32)
        self.assertEqual(triangle_ids.dtype, torch.int32)
        self.assertEqual(float(depth[0, 0]), torch.finfo(torch.float32).max)

        depth[16, 24] = 0.5
        triangle_ids[16, 24] = 17
        returned_depth, returned_triangle_ids = torch_renderer.rasterize_mesh(
            torch.from_numpy(self.vertices),
            torch.from_numpy(self.triangles),
            self.camera,
            z_buffer=depth,
            triangle_index_buffer=triangle_ids,
        )

        self.assertEqual(returned_depth.data_ptr(), depth.data_ptr())
        self.assertIs(returned_triangle_ids, triangle_ids)
        self.assertEqual(int(triangle_ids[20, 32]), 1)
        self.assertAlmostEqual(float(depth[20, 32]), 1.0, places=5)
        self.assertEqual(int(triangle_ids[16, 24]), 17)
        self.assertEqual(float(depth[16, 24]), 0.5)

        wide_depth = torch.full(
            (48, 80), torch.finfo(torch.float32).max, dtype=torch.float32
        )[:, :63]
        wide_triangle_ids = torch.full((48, 80), -1, dtype=torch.int32)[:, :63]
        torch_renderer.rasterize_mesh(
            torch.from_numpy(self.vertices),
            torch.from_numpy(self.triangles),
            self.camera,
            z_buffer=wide_depth,
            triangle_index_buffer=wide_triangle_ids,
        )
        self.assertEqual(wide_depth.stride(), (80, 1))
        self.assertEqual(int(wide_triangle_ids[20, 32]), 1)
        self.assertAlmostEqual(float(wide_depth[20, 32]), 1.0, places=5)

        depth_only, no_triangle_ids = torch_renderer.rasterize_mesh(
            torch.from_numpy(self.vertices),
            torch.from_numpy(self.triangles),
            self.camera,
        )
        self.assertIsNone(no_triangle_ids)
        self.assertGreater(
            int(torch.count_nonzero(depth_only < torch.finfo(torch.float32).max)), 0
        )

    def test_character_render_matches_fk_skinning_and_explicit_rest_shape(
        self,
    ) -> None:
        character, model_parameters, model_matrix = _make_character_inputs("cpu")
        model_parameters.requires_grad_(True)
        skeleton_state = character.model_parameters_to_skeleton_state(
            model_parameters,
            use_double_precision=False,
            backend="torch",
        )
        vertices = character.skin_points(
            skeleton_state,
            backend="torch",
        ).contiguous()
        self.assertFalse(torch.equal(vertices[0], vertices[1]))
        expected_depth, _ = torch_renderer.rasterize_mesh(
            vertices,
            character.mesh.faces,
            self.camera,
            model_matrix=model_matrix,
            back_face_culling=False,
        )

        actual_depth, triangle_ids = torch_renderer.rasterize_character(
            character,
            model_parameters,
            self.camera,
            backend="torch",
            model_matrix=model_matrix,
            back_face_culling=False,
        )

        self.assertIsNone(triangle_ids)
        self.assertFalse(actual_depth.requires_grad)
        torch.testing.assert_close(actual_depth, expected_depth)
        covered = actual_depth < torch.finfo(torch.float32).max
        self.assertTrue(bool(torch.all(torch.count_nonzero(covered, dim=(1, 2)) > 0)))

        shaped_rest_vertices = character.mesh.rest_vertices + torch.tensor(
            [0.35, 0.0, 0.0]
        )
        shaped_vertices = character.skin_points(
            skeleton_state,
            shaped_rest_vertices,
            backend="torch",
        ).contiguous()
        expected_shaped_depth, _ = torch_renderer.rasterize_mesh(
            shaped_vertices,
            character.mesh.faces,
            self.camera,
            model_matrix=model_matrix,
            back_face_culling=False,
        )
        actual_shaped_depth, _ = torch_renderer.rasterize_character(
            character,
            model_parameters,
            self.camera,
            backend="torch",
            rest_vertex_positions=shaped_rest_vertices,
            model_matrix=model_matrix,
            back_face_culling=False,
        )

        torch.testing.assert_close(actual_shaped_depth, expected_shaped_depth)
        shaped_covered = actual_shaped_depth < torch.finfo(torch.float32).max
        self.assertGreater(int(torch.count_nonzero(covered != shaped_covered)), 0)

    def test_cpu_vertex_indices_and_surface_normals(self) -> None:
        vertices = torch.from_numpy(self.vertices[:3].copy())
        normals = torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        vertex_ids = torch_renderer.create_index_buffer(self.camera, device="cpu")
        surface_normals = torch_renderer.create_surface_normals_buffer(
            self.camera, device="cpu"
        )

        depth, _ = torch_renderer.rasterize_mesh(
            vertices,
            triangles,
            self.camera,
            vertex_normals=normals,
            vertex_index_buffer=vertex_ids,
            surface_normals_buffer=surface_normals,
        )

        covered = depth < torch.finfo(torch.float32).max
        self.assertGreater(int(torch.count_nonzero(covered)), 0)
        self.assertEqual(set(vertex_ids[covered].tolist()), {0, 1, 2})
        torch.testing.assert_close(
            surface_normals[covered],
            torch.tensor([0.0, 0.0, -1.0]).expand(int(torch.count_nonzero(covered)), 3),
        )

    def test_batched_cpu_matches_individual_renders(self) -> None:
        vertices = torch.from_numpy(self.vertices)
        triangles = torch.from_numpy(self.triangles)
        batched_vertices = torch.stack(
            [vertices, vertices + torch.tensor([0.1, 0.0, 0.0])]
        )

        batch_triangle_ids = torch_renderer.create_index_buffer(
            self.camera, batch_size=2, device="cpu"
        )
        batch_depth, returned_batch_triangle_ids = torch_renderer.rasterize_mesh(
            batched_vertices,
            triangles,
            self.camera,
            triangle_index_buffer=batch_triangle_ids,
        )

        self.assertIs(returned_batch_triangle_ids, batch_triangle_ids)

        self.assertEqual(tuple(batch_depth.shape), (2, 48, 63))
        self.assertEqual(batch_depth.stride(), (48 * 64, 64, 1))
        self.assertEqual(batch_triangle_ids.stride(), (48 * 64, 64, 1))
        for frame in range(2):
            triangle_ids = torch_renderer.create_index_buffer(self.camera, device="cpu")
            depth, returned_triangle_ids = torch_renderer.rasterize_mesh(
                batched_vertices[frame].contiguous(),
                triangles,
                self.camera,
                triangle_index_buffer=triangle_ids,
            )
            self.assertIs(returned_triangle_ids, triangle_ids)
            torch.testing.assert_close(batch_depth[frame], depth)
            torch.testing.assert_close(batch_triangle_ids[frame], triangle_ids)

    def test_cpu_factory_aligns_odd_height_images(self) -> None:
        camera = pym_camera.Camera(
            pym_camera.PinholeIntrinsicsModel(
                image_width=17,
                image_height=47,
                fx=10.0,
                fy=30.0,
                cx=8.5,
                cy=23.5,
            )
        )
        single_depth = torch_renderer.create_z_buffer(camera, device="cpu")
        single_triangle_ids = torch_renderer.create_index_buffer(camera, device="cpu")
        for name, buffer in (
            ("single_depth", single_depth),
            ("single_triangle_ids", single_triangle_ids),
        ):
            with self.subTest(name=name):
                self.assertEqual(buffer.stride(-1), 1)
                self.assertGreaterEqual(buffer.stride(-2), camera.image_width)
                self.assertEqual(buffer.stride(-2) * buffer.element_size() % 64, 0)

        depth = torch_renderer.create_z_buffer(camera, batch_size=2, device="cpu")
        triangle_ids = torch_renderer.create_index_buffer(
            camera, batch_size=2, device="cpu"
        )

        for name, buffer in (("depth", depth), ("triangle_ids", triangle_ids)):
            with self.subTest(name=name):
                self.assertEqual(buffer.stride(-1), 1)
                self.assertGreaterEqual(buffer.stride(-2), camera.image_width)
                self.assertEqual(buffer.stride(-2) * buffer.element_size() % 64, 0)
                self.assertEqual(buffer.stride(0) * buffer.element_size() % 64, 0)

        returned_depth, returned_triangle_ids = torch_renderer.rasterize_mesh(
            torch.stack(
                (torch.from_numpy(self.vertices), torch.from_numpy(self.vertices))
            ),
            torch.from_numpy(self.triangles),
            camera,
            z_buffer=depth,
            triangle_index_buffer=triangle_ids,
        )
        self.assertEqual(returned_depth.data_ptr(), depth.data_ptr())
        self.assertEqual(returned_triangle_ids.data_ptr(), triangle_ids.data_ptr())

    def test_rejects_invalid_output_layouts(self) -> None:
        vertices = torch.from_numpy(self.vertices)
        triangles = torch.from_numpy(self.triangles)
        with self.assertRaisesRegex(RuntimeError, "padded storage"):
            torch_renderer.rasterize_mesh(
                vertices,
                triangles,
                self.camera,
                z_buffer=torch.full((48, 63), torch.finfo(torch.float32).max),
            )

        storage = torch.full((48, 128), torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(RuntimeError, "aligned rows"):
            torch_renderer.rasterize_mesh(
                vertices,
                triangles,
                self.camera,
                z_buffer=storage[:, 4:67],
            )

        undersized_storage = torch.empty(48 * 64 - 1, dtype=torch.float32)
        undersized_depth = undersized_storage.as_strided((48, 63), (64, 1))
        with self.assertRaisesRegex(RuntimeError, "backing storage"):
            torch_renderer.rasterize_mesh(
                vertices,
                triangles,
                self.camera,
                z_buffer=undersized_depth,
            )

        batch_stride = 48 * 64 + 1
        batched_storage = torch.full(
            (batch_stride + 48 * 64,), torch.finfo(torch.float32).max
        )
        misaligned_batch = batched_storage.as_strided(
            (2, 48, 63), (batch_stride, 64, 1)
        )
        with self.assertRaisesRegex(RuntimeError, "batch stride"):
            torch_renderer.rasterize_mesh(
                torch.stack((vertices, vertices)),
                triangles,
                self.camera,
                z_buffer=misaligned_batch,
            )

    def test_native_cpu_rejects_undersized_backing_storage(self) -> None:
        row_stride = 64
        image_stride = self.camera.image_height * row_stride
        required_elements = 2 * image_stride
        output_shape = (2, self.camera.image_height, self.camera.image_width)
        output_strides = (image_stride * 4, row_stride * 4, 4)

        for name, depth_elements, index_elements in (
            ("z_buffer", required_elements - 1, required_elements),
            ("triangle_index_buffer", required_elements, required_elements - 1),
        ):
            with self.subTest(name=name):
                depth_storage = torch.full(
                    (depth_elements,), torch.finfo(torch.float32).max
                ).numpy()
                index_storage = torch.full(
                    (index_elements,), -1, dtype=torch.int32
                ).numpy()
                depth = np.lib.stride_tricks.as_strided(
                    depth_storage,
                    shape=output_shape,
                    strides=output_strides,
                )
                triangle_ids = np.lib.stride_tricks.as_strided(
                    index_storage,
                    shape=output_shape,
                    strides=output_strides,
                )
                with self.assertRaisesRegex(RuntimeError, "backing storage"):
                    pym_renderer._rasterize_meshes(
                        np.stack((self.vertices, self.vertices)),
                        self.triangles,
                        self.camera,
                        depth,
                        triangle_ids,
                    )

    def test_native_cpu_rejects_cyclic_buffer_base(self) -> None:
        class CyclicBaseArray(np.ndarray):
            @property
            def base(self) -> np.ndarray:
                return self

        storage_shape = (2, self.camera.image_height, 64)
        depth = np.full(storage_shape, np.finfo(np.float32).max, dtype=np.float32)[
            ..., : self.camera.image_width
        ].view(CyclicBaseArray)
        triangle_ids = np.full(storage_shape, -1, dtype=np.int32)[
            ..., : self.camera.image_width
        ]

        with self.assertRaisesRegex(RuntimeError, "base chain contains a cycle"):
            pym_renderer._rasterize_meshes(
                np.stack((self.vertices, self.vertices)),
                self.triangles,
                self.camera,
                depth,
                triangle_ids,
            )


if _CUDA_BACKEND_AVAILABLE and torch.cuda.is_available():

    class TestTorchRendererCuda(unittest.TestCase):
        def setUp(self) -> None:
            self.camera = _make_camera()
            self.vertices, self.triangles = _make_mesh()

        def test_cuda_matches_cpu_with_model_transform(self) -> None:
            model_matrix = np.eye(4, dtype=np.float32)
            model_matrix[0, 3] = 0.1
            model_matrix[2, 3] = 0.3
            model_matrix[3, 0] = 0.02
            cpu_triangle_ids = torch_renderer.create_index_buffer(
                self.camera, device="cpu"
            )
            cpu_depth, _ = torch_renderer.rasterize_mesh(
                torch.from_numpy(self.vertices),
                torch.from_numpy(self.triangles),
                self.camera,
                triangle_index_buffer=cpu_triangle_ids,
                model_matrix=model_matrix,
            )
            gpu_triangle_ids = torch_renderer.create_index_buffer(
                self.camera, device="cuda"
            )
            gpu_depth, _ = torch_renderer.rasterize_mesh(
                torch.from_numpy(self.vertices).cuda(),
                torch.from_numpy(self.triangles).cuda(),
                self.camera,
                triangle_index_buffer=gpu_triangle_ids,
                model_matrix=model_matrix,
            )
            gpu_depth = gpu_depth.cpu()
            gpu_triangle_ids = gpu_triangle_ids.cpu()

            cpu_covered = cpu_depth < torch.finfo(torch.float32).max
            gpu_covered = gpu_depth < torch.finfo(torch.float32).max
            self.assertGreater(int(torch.count_nonzero(cpu_covered)), 0)
            self.assertEqual(set(cpu_triangle_ids[cpu_covered].tolist()), {0, 1})
            self.assertLessEqual(
                int(torch.count_nonzero(cpu_covered != gpu_covered)), 4
            )
            common = cpu_covered & gpu_covered
            torch.testing.assert_close(
                cpu_depth[common],
                gpu_depth[common],
                atol=2e-4,
                rtol=0.0,
            )
            self.assertTrue(
                bool(
                    torch.all(
                        (gpu_triangle_ids[gpu_covered] >= 0)
                        & (gpu_triangle_ids[gpu_covered] < 2)
                    )
                )
            )

        def test_cuda_character_render_matches_explicit_fk_and_skinning(self) -> None:
            character, model_parameters, model_matrix = _make_character_inputs("cuda")
            skeleton_state = character.model_parameters_to_skeleton_state(
                model_parameters,
                use_double_precision=False,
                backend="auto",
            )
            vertices = character.skin_points(
                skeleton_state,
                backend="auto",
            ).contiguous()
            expected_depth, _ = torch_renderer.rasterize_mesh(
                vertices,
                character.mesh.faces,
                self.camera,
                model_matrix=model_matrix,
                back_face_culling=False,
            )

            actual_depth, triangle_ids = torch_renderer.rasterize_character(
                character,
                model_parameters,
                self.camera,
                backend="auto",
                model_matrix=model_matrix,
                back_face_culling=False,
            )

            self.assertIsNone(triangle_ids)
            torch.testing.assert_close(actual_depth, expected_depth)
            covered = actual_depth < torch.finfo(torch.float32).max
            self.assertTrue(
                bool(torch.all(torch.count_nonzero(covered, dim=(1, 2)) > 0))
            )

        def test_cuda_vertex_indices_and_surface_normals_match_cpu(self) -> None:
            vertices_cpu = torch.from_numpy(self.vertices[:3].copy())
            triangles_cpu = torch.tensor([[0, 1, 2]], dtype=torch.int32)
            normals_cpu = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
            normals_cpu = torch.nn.functional.normalize(normals_cpu, dim=1).repeat(3, 1)
            model_matrix = np.eye(4, dtype=np.float32)
            model_matrix[0, 0] = 1.5
            model_matrix[1, 1] = 0.75
            model_matrix[2, 2] = 1.2
            model_matrix[0, 1] = 0.2

            cpu_vertex_ids = torch_renderer.create_index_buffer(
                self.camera, device="cpu"
            )
            cpu_surface_normals = torch_renderer.create_surface_normals_buffer(
                self.camera, device="cpu"
            )
            cpu_depth, _ = torch_renderer.rasterize_mesh(
                vertices_cpu,
                triangles_cpu,
                self.camera,
                vertex_normals=normals_cpu,
                vertex_index_buffer=cpu_vertex_ids,
                surface_normals_buffer=cpu_surface_normals,
                model_matrix=model_matrix,
            )

            vertices_gpu = vertices_cpu.cuda()
            triangles_gpu = triangles_cpu.cuda()
            normals_gpu = normals_cpu.cuda()
            gpu_vertex_ids = torch_renderer.create_index_buffer(
                self.camera, device="cuda"
            )
            gpu_surface_normals = torch_renderer.create_surface_normals_buffer(
                self.camera, device="cuda"
            )
            gpu_depth, _ = torch_renderer.rasterize_mesh(
                vertices_gpu,
                triangles_gpu,
                self.camera,
                vertex_normals=normals_gpu,
                vertex_index_buffer=gpu_vertex_ids,
                surface_normals_buffer=gpu_surface_normals,
                model_matrix=model_matrix,
            )
            gpu_depth = gpu_depth.cpu()
            gpu_vertex_ids = gpu_vertex_ids.cpu()
            gpu_surface_normals = gpu_surface_normals.cpu()

            cpu_covered = cpu_depth < torch.finfo(torch.float32).max
            gpu_covered = gpu_depth < torch.finfo(torch.float32).max
            common = cpu_covered & gpu_covered
            self.assertGreater(int(torch.count_nonzero(common)), 0)
            self.assertLessEqual(
                int(torch.count_nonzero(cpu_covered != gpu_covered)), 4
            )
            torch.testing.assert_close(
                cpu_surface_normals[common],
                gpu_surface_normals[common],
                atol=2e-4,
                rtol=0.0,
            )
            self.assertLessEqual(
                int(
                    torch.count_nonzero(
                        cpu_vertex_ids[common] != gpu_vertex_ids[common]
                    )
                ),
                4,
            )

        def test_batched_cuda_matches_individual_renders(self) -> None:
            vertices = torch.from_numpy(self.vertices[:3].copy()).cuda()
            triangles = torch.from_numpy(self.triangles[:1].copy()).cuda()
            normals = torch.tensor([[0.0, 0.0, -1.0]], device="cuda").repeat(3, 1)
            offsets = torch.tensor(
                [[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                device="cuda",
            )
            batched_vertices = vertices.unsqueeze(0) + offsets[:, None, :]
            batched_normals = normals.unsqueeze(0).repeat(3, 1, 1)
            batch_triangle_ids = torch_renderer.create_index_buffer(
                self.camera, batch_size=3, device="cuda"
            )
            batch_vertex_ids = torch_renderer.create_index_buffer(
                self.camera, batch_size=3, device="cuda"
            )
            batch_surface_normals = torch_renderer.create_surface_normals_buffer(
                self.camera, batch_size=3, device="cuda"
            )

            batch_depth, returned_batch_triangle_ids = torch_renderer.rasterize_mesh(
                batched_vertices,
                triangles,
                self.camera,
                vertex_normals=batched_normals,
                triangle_index_buffer=batch_triangle_ids,
                vertex_index_buffer=batch_vertex_ids,
                surface_normals_buffer=batch_surface_normals,
            )

            self.assertIs(returned_batch_triangle_ids, batch_triangle_ids)

            self.assertEqual(tuple(batch_depth.shape), (3, 48, 63))
            self.assertEqual(batch_depth.stride(), (48 * 64, 64, 1))
            self.assertEqual(batch_triangle_ids.stride(), (48 * 64, 64, 1))
            self.assertEqual(
                batch_surface_normals.stride(), (48 * 64 * 3, 64 * 3, 3, 1)
            )
            for frame in range(3):
                triangle_ids = torch_renderer.create_index_buffer(
                    self.camera, device="cuda"
                )
                vertex_ids = torch_renderer.create_index_buffer(
                    self.camera, device="cuda"
                )
                surface_normals = torch_renderer.create_surface_normals_buffer(
                    self.camera, device="cuda"
                )
                depth, returned_triangle_ids = torch_renderer.rasterize_mesh(
                    batched_vertices[frame].contiguous(),
                    triangles,
                    self.camera,
                    vertex_normals=batched_normals[frame].contiguous(),
                    triangle_index_buffer=triangle_ids,
                    vertex_index_buffer=vertex_ids,
                    surface_normals_buffer=surface_normals,
                )
                self.assertIs(returned_triangle_ids, triangle_ids)
                torch.testing.assert_close(batch_depth[frame], depth)
                torch.testing.assert_close(batch_triangle_ids[frame], triangle_ids)
                torch.testing.assert_close(batch_vertex_ids[frame], vertex_ids)
                torch.testing.assert_close(
                    batch_surface_normals[frame], surface_normals
                )

        def test_current_stream_preallocated_outputs_and_device_validation(
            self,
        ) -> None:
            vertices = torch.from_numpy(self.vertices).cuda()
            triangles = torch.from_numpy(self.triangles).cuda()
            depth = torch_renderer.create_z_buffer(self.camera, device=vertices.device)
            triangle_ids = torch_renderer.create_index_buffer(
                self.camera, device=vertices.device
            )
            depth[16, 24] = 0.5
            triangle_ids[16, 24] = 17

            producer_stream = torch.cuda.current_stream()
            stream = torch.cuda.Stream()

            with torch.cuda.stream(stream):
                stream.wait_stream(producer_stream)
                returned_depth, returned_triangle_ids = torch_renderer.rasterize_mesh(
                    vertices,
                    triangles,
                    self.camera,
                    z_buffer=depth,
                    triangle_index_buffer=triangle_ids,
                )
            stream.synchronize()

            self.assertEqual(returned_depth.data_ptr(), depth.data_ptr())
            self.assertIs(returned_triangle_ids, triangle_ids)
            self.assertIn(int(triangle_ids[20, 32]), (0, 1))
            self.assertAlmostEqual(float(depth[20, 32]), 1.0, places=5)
            self.assertEqual(int(triangle_ids[16, 24]), 17)
            self.assertAlmostEqual(float(depth[16, 24]), 0.5, places=5)

            with self.assertRaisesRegex(RuntimeError, "same device"):
                torch_renderer.rasterize_mesh(
                    vertices,
                    torch.from_numpy(self.triangles),
                    self.camera,
                )

            undersized_storage = torch.empty(
                self.camera.image_height * depth.stride(0) - 1,
                dtype=torch.float32,
                device=vertices.device,
            )
            undersized_depth = undersized_storage.as_strided(
                (self.camera.image_height, self.camera.image_width),
                (depth.stride(0), 1),
            )
            with self.assertRaisesRegex(RuntimeError, "backing storage"):
                torch_renderer.rasterize_mesh(
                    vertices,
                    triangles,
                    self.camera,
                    z_buffer=undersized_depth,
                )

            image_stride = self.camera.image_height * depth.stride(0)
            undersized_batch_storage = torch.empty(
                2 * image_stride - 1,
                dtype=torch.float32,
                device=vertices.device,
            )
            undersized_batch_depth = undersized_batch_storage.as_strided(
                (2, self.camera.image_height, self.camera.image_width),
                (image_stride, depth.stride(0), 1),
            )
            with self.assertRaisesRegex(RuntimeError, "backing storage"):
                torch_renderer.rasterize_mesh(
                    torch.stack((vertices, vertices)),
                    triangles,
                    self.camera,
                    z_buffer=undersized_batch_depth,
                )

            normal_row_stride = 3 * depth.stride(0)
            undersized_normal_storage = torch.empty(
                self.camera.image_height * normal_row_stride - 1,
                dtype=torch.float32,
                device=vertices.device,
            )
            undersized_normals = undersized_normal_storage.as_strided(
                (self.camera.image_height, self.camera.image_width, 3),
                (normal_row_stride, 3, 1),
            )
            with self.assertRaisesRegex(RuntimeError, "backing storage"):
                torch_renderer.rasterize_mesh(
                    vertices,
                    triangles,
                    self.camera,
                    surface_normals_buffer=undersized_normals,
                )

            normal_image_stride = 3 * image_stride
            undersized_batch_normal_storage = torch.empty(
                2 * normal_image_stride - 1,
                dtype=torch.float32,
                device=vertices.device,
            )
            undersized_batch_normals = undersized_batch_normal_storage.as_strided(
                (2, self.camera.image_height, self.camera.image_width, 3),
                (normal_image_stride, normal_row_stride, 3, 1),
            )
            with self.assertRaisesRegex(RuntimeError, "backing storage"):
                torch_renderer.rasterize_mesh(
                    torch.stack((vertices, vertices)),
                    triangles,
                    self.camera,
                    surface_normals_buffer=undersized_batch_normals,
                )

        def test_native_backend_default_output_is_depth_only_and_padded(self) -> None:
            depth, triangle_ids = pym_renderer_cuda.rasterize_mesh(
                torch.from_numpy(self.vertices).cuda(),
                torch.from_numpy(self.triangles).cuda(),
                self.camera,
            )

            self.assertEqual(tuple(depth.shape), (48, 63))
            self.assertIsNone(triangle_ids)
            self.assertEqual(depth.stride(), (64, 1))
            self.assertEqual(float(depth[0, 0]), torch.finfo(torch.float32).max)

        def test_invalid_triangle_indices_are_ignored(self) -> None:
            triangles = torch.tensor(
                [[0, 1, 2], [0, 1, 99], [0, -1, 2]],
                dtype=torch.int32,
                device="cuda",
            )
            triangle_ids = torch_renderer.create_index_buffer(
                self.camera, device="cuda"
            )
            depth, _ = torch_renderer.rasterize_mesh(
                torch.from_numpy(self.vertices[:3]).cuda(),
                triangles,
                self.camera,
                triangle_index_buffer=triangle_ids,
            )

            covered = depth < torch.finfo(torch.float32).max
            self.assertGreater(int(torch.count_nonzero(covered)), 0)
            self.assertEqual(set(triangle_ids[covered].cpu().tolist()), {0})
