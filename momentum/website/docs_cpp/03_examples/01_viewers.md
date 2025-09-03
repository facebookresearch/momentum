---
sidebar_position: 1
---

# Viewers

Momentum supports various file formats for storing characters and motions. In this section, we demonstrate how to visualize these file formats using Momentum powered by [Rerun](https://rerun.io/).

For simplicity, we have built individual viewers for each file format. Each viewer example may require different arguments. Please use the `--help` option to see the correct usage of each example.

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

## Troubleshooting

If you haven't installed the rerun viewer on your system before running the viewer examples, you may encounter the following error:

```
[2025-09-03T17:30:10Z WARN  rerun_c::error] Error message was too long for C error description buffer. Full message
    Failed to find Rerun Viewer executable in PATH.
```

To resolve this issue, you have two options:

**Option 1:** Install the rerun viewer on your system:

```
pixi global install rerun-sdk==<same_rerun_sdk_version>
```

**Option 2:** Run `buck run rerun` before running the viewer examples.

</FbInternalOnly>

## GLB Viewer

To run the GLB viewer, use the following command:

<OssOnly>
```
pixi run glb_viewer --input <my_file.glb>
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt glb_viewer -- --input <my_file.glb>
```
</FbInternalOnly>

![glb_viewer](/img/glb_viewer.png)

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/glb_viewer)

## FBX Viewer

To run the FBX viewer, use the following command:

<OssOnly>
```
pixi run fbx_viewer --input <my_file.fbx>
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt fbx_viewer -- --input <my_file.fbx>
```
</FbInternalOnly>

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/fbx_viewer)

## C3D Viewer

To run the C3D viewer, use the following command:

<OssOnly>
```
pixi run c3d_viewer --input <my_file.c3d>
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt c3d_viewer -- --input <my_file.c3d>
```
</FbInternalOnly>

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/c3d_viewer)

## URDF Viewer

To run the URDF viewer, use the following command:

<OssOnly>
```
pixi run urdf_viewer --input <my_file.urdf>
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt urdf_viewer -- --input <my_file.urdf>
```
</FbInternalOnly>

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/urdf_viewer)

For example, you can download an Atlas robot from this [link](https://github.com/Daniella1/urdf_files_dataset/blob/main/urdf_files/matlab/Atlas/urdf/atlas.urdf), which may look like:

![urdf_viewer](/img/urdf_viewer.png)
