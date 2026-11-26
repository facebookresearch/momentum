---
sidebar_position: 4
---

# Character Files

Momentum stores a character and its motion in a few file types, and each file type has a role. This page tells what each file holds and which file to change.

## Model package

A model package is the set of files that defines a character:

| File | Holds |
| --- | --- |
| `.fbx`, `.glb` or `.gltf` | The skeleton, the mesh and its skin weights, and optionally blend shapes and collision geometry. |
| `.model` | The rig: the parameter transform, the parameter sets, the pose constraints and the parameter limits. |
| `.locators` | The locators: named points attached to joints. |

The `.model` and `.locators` files are the sources of the rig and the locators: to change a rig, change its `.model` file. For example, [MHR](https://github.com/facebookresearch/MHR) ships its skeleton and mesh as `lod1.fbx` and its rig as `compact_v6_1.model`. An FBX file from another tool stores no rig, so it needs its `.model` file.

Load a model package with `loadFullCharacter()` from `momentum/io/character_io.h` in C++, with `Character.load_fbx()` and its `model_filename` and `locators_filename` arguments in Python, or with the `-m`, `-p` and `-l` options of the [convert_model tools](../03_examples/03_convert_model.md#files-and-options-shared-by-the-tools). The `.model` file replaces the parameter transform and the parameter limits that the character file stores. The `.locators` file replaces the locators of the character file in C++ and in the tools; `Character.load_fbx()` adds its locators to those of the FBX file.

## Files with motion

A `.glb`, `.gltf` or `.fbx` file that momentum writes can also hold the motion of one or more characters, for example a hand and the object it holds. With the motion, the file stores a copy of the rig and of the locators of each character, and optionally marker sequences, so that the file can be read back without its `.model` and `.locators` files. `loadFbxCharacter()` and the other loaders of `momentum/io/fbx/fbx_io.h` read the locators only with `KeepLocators::Yes`; `loadFullCharacter()`, `Character.load_fbx()` and the tools read them.

These copies are snapshots of the rig and the locators at the time of the export, not sources. A change to a `.model` or `.locators` file does not change them, and a rig that is copied into many files is hard to change. To apply a changed rig or changed locators to such a file, load it again with the `.model` or `.locators` file, as for a model package.

A glTF file can store the motion as model parameters. An FBX file stores joint animation curves, and the convert_model tools fit the model parameters to the rig.

## Other files

- A `.bvh` file holds the skeleton and the motion of one character, without a mesh or a rig.
- A `.urdf` file describes a robot. Momentum reads its skeleton with the joint limits, its parameter transform and its meshes.
- Marker sequences from motion capture are in `.c3d` or `.trc` files, or in `.glb`, `.gltf` or `.fbx` files that momentum writes. `loadMarkers()` from `momentum/io/marker/marker_io.h` reads all of them.
