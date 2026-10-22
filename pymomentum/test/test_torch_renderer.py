# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pymomentum.camera as pym_camera  # @manual=:camera
import pymomentum.renderer as pym_renderer  # @manual=:renderer
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


class TestTorchRendererCpu(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = _make_camera()
        self.vertices, self.triangles = _make_mesh()

    def test_factories_and_cpu_rasterization(self) -> None:
        depth = torch_renderer.create_z_buffer(self.camera, device="cpu")
        triangle_ids = torch_renderer.create_index_buffer(
            self.camera, device=torch.device("cpu")
        )
        self.assertEqual(tuple(depth.shape), (48, 63))
        self.assertEqual(tuple(triangle_ids.shape), (48, 63))
        self.assertEqual(depth.stride(), (64, 1))
        self.assertEqual(triangle_ids.stride(), (64, 1))
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
        self.assertEqual(returned_triangle_ids.data_ptr(), triangle_ids.data_ptr())
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

    def test_batched_cpu_matches_individual_renders(self) -> None:
        vertices = torch.from_numpy(self.vertices)
        triangles = torch.from_numpy(self.triangles)
        batched_vertices = torch.stack(
            [vertices, vertices + torch.tensor([0.1, 0.0, 0.0])]
        )

        batch_depth, batch_triangle_ids = torch_renderer.rasterize_mesh(
            batched_vertices,
            triangles,
            self.camera,
        )

        self.assertEqual(tuple(batch_depth.shape), (2, 48, 63))
        self.assertEqual(batch_depth.stride(), (48 * 64, 64, 1))
        self.assertEqual(batch_triangle_ids.stride(), (48 * 64, 64, 1))
        for frame in range(2):
            depth, triangle_ids = torch_renderer.rasterize_mesh(
                batched_vertices[frame].contiguous(),
                triangles,
                self.camera,
            )
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
            cpu_depth, cpu_triangle_ids = torch_renderer.rasterize_mesh(
                torch.from_numpy(self.vertices),
                torch.from_numpy(self.triangles),
                self.camera,
                model_matrix=model_matrix,
            )
            gpu_depth, gpu_triangle_ids = torch_renderer.rasterize_mesh(
                torch.from_numpy(self.vertices).cuda(),
                torch.from_numpy(self.triangles).cuda(),
                self.camera,
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
            same_triangle = common & (cpu_triangle_ids == gpu_triangle_ids)
            torch.testing.assert_close(
                cpu_depth[same_triangle],
                gpu_depth[same_triangle],
                atol=2e-4,
                rtol=0.0,
            )
            self.assertLessEqual(int(torch.count_nonzero(common & ~same_triangle)), 4)

        def test_batched_cuda_matches_individual_renders(self) -> None:
            vertices = torch.from_numpy(self.vertices).cuda()
            triangles = torch.from_numpy(self.triangles).cuda()
            offsets = torch.tensor(
                [[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                device="cuda",
            )
            batched_vertices = vertices.unsqueeze(0) + offsets[:, None, :]

            batch_depth, batch_triangle_ids = torch_renderer.rasterize_mesh(
                batched_vertices,
                triangles,
                self.camera,
            )

            self.assertEqual(tuple(batch_depth.shape), (3, 48, 63))
            self.assertEqual(batch_depth.stride(), (48 * 64, 64, 1))
            self.assertEqual(batch_triangle_ids.stride(), (48 * 64, 64, 1))
            for frame in range(3):
                depth, triangle_ids = torch_renderer.rasterize_mesh(
                    batched_vertices[frame].contiguous(),
                    triangles,
                    self.camera,
                )
                torch.testing.assert_close(batch_depth[frame], depth)
                torch.testing.assert_close(batch_triangle_ids[frame], triangle_ids)

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
            self.assertEqual(returned_triangle_ids.data_ptr(), triangle_ids.data_ptr())
            self.assertEqual(int(triangle_ids[20, 32]), 1)
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

        def test_native_backend_default_outputs_are_padded(self) -> None:
            depth, triangle_ids = pym_renderer_cuda.rasterize_mesh(
                torch.from_numpy(self.vertices).cuda(),
                torch.from_numpy(self.triangles).cuda(),
                self.camera,
            )

            self.assertEqual(tuple(depth.shape), (48, 63))
            self.assertEqual(tuple(triangle_ids.shape), (48, 63))
            self.assertEqual(depth.stride(), (64, 1))
            self.assertEqual(triangle_ids.stride(), (64, 1))
            self.assertEqual(float(depth[0, 0]), torch.finfo(torch.float32).max)

        def test_invalid_triangle_indices_are_ignored(self) -> None:
            triangles = torch.tensor(
                [[0, 1, 2], [0, 1, 99], [0, -1, 2]],
                dtype=torch.int32,
                device="cuda",
            )
            depth, triangle_ids = torch_renderer.rasterize_mesh(
                torch.from_numpy(self.vertices[:3]).cuda(),
                triangles,
                self.camera,
            )

            covered = depth < torch.finfo(torch.float32).max
            self.assertGreater(int(torch.count_nonzero(covered)), 0)
            self.assertEqual(set(triangle_ids[covered].cpu().tolist()), {0})
