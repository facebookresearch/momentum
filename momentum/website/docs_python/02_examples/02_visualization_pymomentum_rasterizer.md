# Visualization using pymomentum rasterizer

Visualizing results is an important part of what we do. There are various ways to do it, but many of our pipelines rely on rendering MP4 movies in batch for later playback. This works well because videos work in any browser and are easy to share links, etc.

## Why render on the CPU?

The obvious choice for visualization for anyone with a graphics background is to render using a GPU, which will be super-fast and provide instant feedback. However, this is often a bad idea on the cluster because cloud GPUs are expensive and use significant energy, so using these GPUs to render images can be quite wasteful. Regardless, almost any rendering job using a sufficiently fast rasterizer (such as our own) is more likely to be bottlenecked by I/O bandwidth than rendering speed so you would see minimal speed gains from even the fastest GPU.

## Why a software rasterizer?

There are many options for visualizing data, and it is extremely nonobvious why we would want to have our own software rasterizer.

* **Software or hardware OpenGL**: challenging to work with, little error checking, many silent failures that generate missing output. Global state is challenging to manage. Doesn't support arbitrary camera models unless you implement custom shaders.
* **Blender/professional tools**: Can produce very pretty soft shadows etc but challenging to work with on the cluster and don't natively support fisheye camera models unless you build a complicated lens shader.
* **WebGL-based visualization**: Very nice when working in Jupyter notebooks but totally unsuitable for rendering in batch. Doesn't natively support our camera models OR standard OpenGL shaders so properly rendering with our camera models can be extremely challenging.
* **Pytorch3d**: optimized for differentiable rendering, very slow on CPU.

## What is pymomentum rasterizer?

Pymomentum rasterizer is a fully-featured rasterizer.

* **Cross-platform**: Implemented using drjit's SIMD wrappers so it runs on both Intel and ARM.
* **Fast**: Runs roughly 2x faster than MesaGL's software OpenGL emulation.
* **Threadsafe**: releases the Python GIL so you can easily run it multithreaded (e.g. render multiple images or multiple frames at once). Compare with e.g. OpenGL which has tons of internal state.
* Full **per-pixel lighting and shading** with multiple lights.
* Runs completely on the CPU: zero OpenGL or GPU dependencies.
* **Easy to use**, completely functional interface (no global state as in OpenGL). Good error reporting (e.g. makes sure your indices are reasonable) and sensible defaults (e.g. if you you don't provide a light, a default lighting setup is automatically provided instead of rendering a black frame). Simple to use (just a single import statement).
* Basic support for **texture mapping**.
* Support for ground plane shadows.
* Can render per-pixel triangle or vertex IDs.
* Supports arbitrary camera models, provided you provide an implementation compliant with the interface in momentum/renderer/camera.h.
* Includes 2d primitives (lines, circles) as well with depth buffer support.

## Using the rasterizer

### Getting started

#### Cameras

The first thing you need is a camera. Pymomentum's rasterizer uses the `pymomentum.renderer.Camera` class. You can construct one in a few ways:

1. You can construct a camera with an intrinsics model and optional extrinsics matrix.
2. If you have a pymomentum body model, you can use `pymomentum.renderer.build_cameras_for_body()` to create a camera that looks at the body and frames it in view.
3. You can construct a default camera and set the extrinsics matrix explicitly, using e.g. `camera.look_at()`.
4. You can use `camera.frame()` to frame a set of 3d points in view (this can be handy for ensuring that an entire animation stays in frame).

Note that the camera determines the image resolution, you can always use `camera.upsample()` to scale up the image as needed for better quality.

```python
import pymomentum.renderer as pym_renderer
import numpy as np

image_height, image_width = 800, 1000

# Create a pinhole intrinsics model
intrinsics = pym_renderer.PinholeIntrinsicsModel(
    image_width=image_width,
    image_height=image_height,
    fx=800.0,  # focal length in pixels
    fy=800.0,
    cx=image_width / 2.0,  # principal point
    cy=image_height / 2.0
)

# Create a camera with the intrinsics
camera = pym_renderer.Camera(intrinsics)

# Move the camera along -z and look at the origin
camera = camera.look_at(
    position=np.array([0, 0, 1]), target=np.zeros(3), up=np.array([0, 1, 0])
)

# Make sure the entire object is in view:
camera = camera.frame(vertex_positions)
```

#### Depth/Image buffers

Now you need to create depth and RGB buffers to render onto. This is very easy now that you have a camera.

```python
import pymomentum.renderer as pym_renderer
z_buffer = pym_renderer.create_z_buffer(camera)
rgb_buffer = pym_renderer.create_rgb_buffer(camera)
```

Note: the buffer size will get padded out to the nearest multiple of 8 for better SIMD performance. You can correct this after the rendering is complete using standard slicing:

```python
z_buffer = z_buffer[:,:camera.image_width]
rgb_buffer = rgb_buffer[:,:camera.image_height]
```

### 3d primitives

#### Meshes

Now, rasterizing a mesh onto the image is a single function call.

```python
pym_renderer.rasterize_mesh(vertex_positions,
  vertex_normals, triangles, camera, z_buffer=z_buffer, rgb_buffer=rgb_buffer)
```

If you have multiple meshes to render, you just call `rasterize_mesh` repeatedly using the same z_buffer.

There is a special function to simplify rasterizing posed pymomentum Characters that takes in a skeleton state:

```python
skel_state = pym_geometry.model_parameters_to_skeleton_state(character, model_params)
pym_renderer.rasterize_character(character, skel_state, camera, z_buffer, rgb_buffer)
```

The default render uses a basic material (white diffuse) and a basic but usable lighting setup where the light is co-located with the camera. If you want a shinier setup, you can change the material:

```python
mat = pym_renderer.PhongMaterial(diffuse_color=np.array([0.8, 0.9, 1.0]),
    specular_color=np.ones(3) * 0.3)

pym_renderer.rasterize_mesh(vertex_positions,
  vertex_normals, triangles, camera, z_buffer=z_buffer, rgb_buffer=rgb_buffer,
  material=mat)
```

If you want to render a wireframe on your mesh, you can use this command:

```python
pym_renderer.rasterize_wireframe(vertex_positions,
  triangles, camera, z_buffer=z_buffer, rgb_buffer=rgb_buffer)
```

#### Spheres and cylinders

There is special functionality for rendering spheres and cylinders.

```python
sphere_centers = torch.stack(
    [torch.arange(-10, 10, 3), 5 * torch.ones(7), torch.ones(7)]
).transpose(0, 1)
pym_renderer.rasterize_spheres(
    sphere_centers, camera, z_buffer, rgb_buffer=rgb_buffer, radius=torch.ones(7)
)
pym_renderer.rasterize_cylinders(
    start_position=torch.tensor([[-5, 8, 0]]),
    end_position=torch.tensor([[5, 8, 0]]),
    camera=camera,
    z_buffer=z_buffer,
    rgb_buffer=rgb_buffer
)
```

You also generate a nice checkerboard ground plane (y defaults to up, but you can change this with the model_matrix if needed).

```python
pym_renderer.rasterize_checkerboard(
    camera=camera,
    z_buffer=z_buffer,
    rgb_buffer=rgb_buffer,
)
```

#### Transforms

You can also transform any object by passing a model transform, the rasterizer is capable of dealing with nonuniform scale and shearing:

```python
xf = np.array(
    [[1, 0, 0, 0], [0, 0.3, 0, 5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)

pym_renderer.rasterize_mesh(vertex_positions,
  vertex_normals, triangles, camera, z_buffer=z_buffer, rgb_buffer=rgb_buffer,
  material=mat, model_matrix=xf)
```

#### Skeletons

Because Character skeletons are so important to working with momentum, we have some extra functionality for rendering them.

```python
pym_renderer.rasterize_skeleton(character, skel_state, camera, z_buffer,
    rgb_buffer=rgb_buffer, style=pym_renderer.SkeletonStyle.Pipes,
    image_offset=np.asarray([-600, 0]), sphere_radius=1.0, cylinder_radius=0.5)
pym_renderer.rasterize_skeleton(character, skel_state, camera, z_buffer,
    rgb_buffer=rgb_buffer, style=pym_renderer.SkeletonStyle.Octahedrons,
    sphere_radius=1.0)
pym_renderer.rasterize_skeleton(character, skel_state, camera, z_buffer,
    rgb_buffer=rgb_buffer, style=pym_renderer.SkeletonStyle.Lines,
    image_offset=np.asarray([600, 0]), sphere_radius=5.0, cylinder_radius=2.0,
    sphere_material=pym_renderer.PhongMaterial(np.asarray([1, 0.6, 0.6])))
```

There are three different skeleton "styles": "Pipes" (3d cylinders and spheres), "Octahedrons" (this asymmetric octahedron shape, useful for visualizing rotations) and "Lines" (2d lines and circles).

### 2d primitives

It can be useful to render 2d primitives like circles and lines but have them respect the z buffer. This can be used:

1. To create a nice grid on the ground plane.
2. To render e.g. 3d keypoints (also see the note below about using a depth offset).

Now, spheres and cylinders work pretty well for these needs, but (1) lines and circles are significantly faster since approximating a sphere requires >100 triangles (2) lines and circles have radius/thickness values defined in pixels instead of worldspace units, making tuning their size easier (3) lines and circles can look more aesthetically pleasing depending on the use case.

```python
pym_renderer.rasterize_lines(
    positions=line_positions,
    camera=camera,
    z_buffer=z_buffer,
    rgb_buffer=rgb_buffer,
    thickness=2.0,
    color=np.array([1.0, 0.0, 0.0])  # Red lines
)
pym_renderer.rasterize_circles(
    positions=circle_positions,
    camera=camera,
    z_buffer=z_buffer,
    rgb_buffer=rgb_buffer,
    radius=5.0,
    line_thickness=1.0,
    line_color=np.array([0.0, 1.0, 0.0])  # Green circles
)
```

Note that aliasing can be particularly bad for lines so see the notes about antialiasing below.

### Rendering on top of existing images

If you want to render on top of an existing image, you can use the `alpha_matte` function. This will automatically downsample the image if necessary (if it was upsampled for anti-aliasing reasons) and handle conversions between float- and uint8-valued buffers.

```python
import cv2

tgt_image = cv2.imread(...)
# OpenCV likes to use BGR but we use RGB
tgt_image = tgt_image[..., ::-1]

rgb_buffer = pym_renderer.create_rgb_buffer(camera)
z_buffer = pym_renderer.create_z_buffer(camera)

# Target image is a [height x width x 3] float- or uint8-valued array:
pym_renderer.alpha_matte(z_buffer, rgb_buffer, tgt_image)
```

### Using depth offset for clearer skeleton/keypoint rendering

A classic approach to rendering e.g. 3d keypoints is to render circles on top of the image. The problem with this approach is that because the depth buffer is not respected, the keypoints will be visible through the mesh. This can be very confusing to look at. Notice in the left character how the skeleton of the right hand is visible all the way through the character, which makes it hard to see what is going on.

We can use the z buffer to correct this, but if we try to rasterize the skeleton to the same image where the mesh is the skeleton will be completely hidden. Passing a `depth_offset` to the rasterizer bumps the skeleton forward, allowing you to see the parts of the skeleton that are just below the mesh surface but still hiding parts of the skeleton that are far behind (the right image).

```python
pym_renderer.rasterize_character(character, skel_state, camera, z_buffer,
    rgb_buffer=rgb_buffer, image_offset=np.asarray([300, 0]),
    material=pym_renderer.PhongMaterial(np.asarray([1, 0.6, 0.6])))
# Use depth_offset to bump the skeleton forward so we can see it "through" the mesh:
pym_renderer.rasterize_skeleton(character, skel_state, camera, z_buffer,
    rgb_buffer=rgb_buffer, style=pym_renderer.SkeletonStyle.Pipes, sphere_radius=1.0,
    cylinder_radius=0.5, depth_offset=-15, image_offset=np.asarray([300, 0]))
```

In addition, you can pass an `image_offset` (in pixels) to any rasterizer function and it will displace a mesh in image-space.

```python
# Render body and skeleton side-by-side using image_offset:
pym_renderer.rasterize_character(character, skel_state, camera, z_buffer,
    rgb_buffer=rgb_buffer, image_offset=np.asarray([-300, 0]),
    material=pym_renderer.PhongMaterial(np.asarray([1, 0.6, 0.6])))
pym_renderer.rasterize_skeleton(character, skel_state, camera, z_buffer,
    rgb_buffer=rgb_buffer, style=pym_renderer.SkeletonStyle.Octahedrons,
    sphere_radius=1.0, cylinder_radius=0.5, depth_offset=-15,
    image_offset=np.asarray([300, 0]))
```

### Ground plane shadows

Shadows can be very helpful in debugging lower body motion. The rasterizer does not support fully general shadows but there is a basic old-school OpenGL trick you can use to generate a nice shadow on the ground plane.

Basically, we can rasterize the mesh projected down onto the ground plane. This is done as a two-step process: the first rasterizes the mesh, generating a depth buffer, and the second splats this shadow onto the ground.

```python
# Two lights, the first is above the person and casts shadows while the other
# is co-located with the camera to ensure good fill.
lights = [pym_renderer.Light.create_point_light(
        np.asarray([-20, 200, 30]), color=np.asarray([0.7, 0.7, 0.7])
    ), pym_renderer.Light.create_point_light(
        camera.center_of_projection,
        np.asarray([0.3, 0.3, 0.3]),
    )]

# Create a separate z buffer for the shadows.
shadow_buffer = pym_renderer.create_z_buffer(camera)
# Rasterize the body mesh onto the shadow Z buffer using a projection matrix
# constructed from the first light:
pym_renderer.rasterize_character(
    character,
    skel_state,
    camera,
    z_buffer=shadow_buffer,
    model_matrix=pym_renderer.create_shadow_projection_matrix(lights[0]),
    back_face_culling=False,  # Disable back-face culling in case the project inverts triangles.
)

# Rasterizer the ground plane to our RGB buffer:
pym_renderer.rasterize_checkerboard(camera, z_buffer, rgb_buffer, width=500,
    subdivisions=3)
# Use the shadow z buffer to darken the ground plane wherever the shadow hits:
very_far = 10000.0
rgb_buffer *= (
    torch.logical_or(shadow_buffer > very_far, z_buffer > very_far)
    .to(torch.float)
    .clamp(0.5, 1.0)
    .unsqueeze(-1)
)
# Finally rasterize the character mesh:
pym_renderer.rasterize_character(
    character, skel_state, camera, z_buffer, rgb_buffer=rgb_buffer, lights=lights
)
```

### Generating a video

For video generation, you can use standard video writing libraries like OpenCV or ffmpeg-python. The basic idea is to render each frame to a buffer, and then write the buffer to a video file.

```python
import cv2

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(file_path, fourcc, video_fps, (video_width, video_height))

for i_frame in range(n_frames):
    full_image = np.zeros(shape=(video_height, video_width, 3), dtype=np.uint8)
    # ... render your frame ...
    # Convert RGB to BGR for OpenCV
    bgr_image = full_image[..., ::-1]
    video_writer.write(bgr_image)

video_writer.release()
```

### Multithreading

As noted above, `pymomentum.renderer` works well in a multithreaded setting. The simplest way to leverage this is using `multiprocessing.dummy.Pool()`:

```python
def rasterize_one_frame(frame_idx: int):
    # ... your rendering code here ...
    return rendered_image

n_threads = 4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    os.path.join(out_path, "animation.mp4"),
    fourcc, 30, (image_width, image_height)
)

with multiprocessing.dummy.Pool(n_threads) as pool:
    for idx, image in enumerate(
        pool.imap(rasterize_one_frame, frames_to_write)
    ):
        # Convert RGB to BGR for OpenCV
        bgr_image = image[..., ::-1]
        video_writer.write(bgr_image)

        if idx % 10 == 0:
            print(f"Write frame {idx} of {len(frames_to_write)}")

video_writer.release()
```

Typically the speedup you get is bottlenecked by the serial parts (video encoding, Python overhead) so you won't see a perfectly linear speedup, but in the above code I saw a roughly 3x speedup on 4 threads (60s-20s) and a 4x speedup on 8 threads (60s-15s). Note that this is a sequence of 675 frames (with shadows and 2x supersampling) and we are getting ~45fps on 8 threads.

### Subdivision

One potential drawback of rasterizing is that we only apply camera distortion to vertices, and the interpolation between vertices is linear. If you have very large objects, you will start to notice that triangles aren't "bending" the way you'd expect toward the edge of wide-angle cameras. The way to address this is to break the mesh into smaller triangles, and `pymomentum.renderer` provides functionality to do this with the `subdivide_mesh` function:

```python
subdivided_vertices, subdivided_normals, subdivided_triangles, _, _ = pym_renderer.subdivide_mesh(
    vertices=vertex_positions,
    normals=vertex_normals,
    triangles=triangles,
    levels=2,  # Number of subdivision levels
    max_edge_length=10.0  # Maximum edge length before subdivision
)
```

The function will subdivide triangles based on:

1. **levels**: Number of subdivision iterations to perform
2. **max_edge_length**: Maximum allowed edge length - longer edges will be broken into smaller triangles.

### Other buffers

The rasterizer knows how to render other quantities as well.

* The `vertex_index_buffer` rasterizes the index of the vertex to the buffer, or -1 for empty pixels.
* The `triangle_index_buffer` rasterizes the index of the triangle to the buffer, or -1 for empty pixels.
* The `surface_normals_buffer` rasterizes the direction of the surface normal in eye coordinates, or all zeros if empty pixels.

These last two buffers can be used for things like per-part segmentation (use the rendered vertex indices to look up into a vertex index to part ID mapping).

```python
# Default index buffer is set to -1 everywhere (this is because vertex
# indices start at 0)
vertex_index_buffer = pym_renderer.create_index_buffer(camera)
triangle_index_buffer = pym_renderer.create_index_buffer(camera)
normals_buffer = pym_renderer.create_rgb_buffer(camera)

pym_renderer.rasterize_character(character, skel_state, camera,
    z_buffer=z_buffer,
    surface_normals_buffer=normals_buffer,
    vertex_index_buffer=vertex_index_buffer,
    triangle_index_buffer=triangle_index_buffer,
)

# Generate some random colors:
random_colors = torch.rand(
    max(triangles.shape[0], vertices.shape[0]), 3, dtype=torch.float32
)
# Need to shift by 1 since empty pixels are set to -1 (torch tensor indexing doesn't
# appear to support -1).
triangle_colors = random_colors[triangle_index_buffer.flatten() + 1, :].reshape(
    rgb_buffer.shape
)
vertex_colors = random_colors[vertex_index_buffer.flatten() + 1, :].reshape(
    rgb_buffer.shape
)
```

From left: RGB buffer, normals buffer, triangle index buffer, vertex index buffer (notice the Voronoi regions).

### Antialiasing

The rasterizer doesn't do any antialiasing, so you may see some jagged edges in your renders. This will probably be less important for meshes but is going to be particularly noticeable for thin structures like lines or thin cylinders. This is easy to fix by supersampling the image, just create a larger camera using `camera.upsample()` and then downsample at the end.

```python
import pymomentum.renderer as pym_renderer
sup_samp: int = 2
cam_supersample = cam.upsample(sup_samp)
z_buffer = pym_renderer.create_z_buffer(cam_supersample)
rgb_buffer = pym_renderer.create_rgb_buffer(cam_supersample)

# render
pym_renderer.rasterize_mesh(...)

output_image = np.zeros(shape=(cam.image_height, cam.image_width, 3)
# Alpha_matte function knows how to handle alpha with upsampled cameras (will
# correctly blend along edges using the averaged alpha).
pym_renderer.alpha_matte(z_buffer, rgb_buffer, output_image)
```

No supersampling vs with supersampling provides significantly better visual quality.
