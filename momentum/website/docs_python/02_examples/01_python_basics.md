---
sidebar_position: 1
---

# Load and pose a character

This example loads a character, changes one model parameter, runs forward
kinematics, and skins the character mesh.

## Prerequisites

Download the repository's public sample character and model definition:

```bash
curl -L \
  https://raw.githubusercontent.com/facebookresearch/momentum/main/momentum/examples/convert_model/test_data/character.fbx \
  -o character.fbx
curl -L \
  https://raw.githubusercontent.com/facebookresearch/momentum/main/momentum/examples/convert_model/test_data/character.model \
  -o character.model
```

The `.model` file populates `character.parameter_transform`; loading only the
FBX would produce a character without usable model parameters.

The same pose operations apply to characters loaded from glTF or URDF.

## Load and inspect the character

```python
import numpy as np
import pymomentum.geometry as pym_geometry

character = pym_geometry.Character.load_fbx(
    fbx_filename="character.fbx",
    model_filename="character.model",
)

print(f"joints: {character.skeleton.size}")
print(f"model parameters: {character.parameter_transform.size}")
print(character.skeleton.joint_names[:10])
print(character.parameter_transform.names[:10])
```

Joint and parameter names come from the asset. Prefer looking them up by name
over copying numeric indices between characters:

```python
joint_index = character.skeleton.joint_index("joint1")
parameter_mask = character.parameter_transform.find_parameters(["joint1_rx"])
parameter_index = int(np.flatnonzero(parameter_mask)[0])
```

Use names that exist in your character; rigs do not share a universal naming
scheme.

## Run forward kinematics

A zero vector represents the character's reference model parameters. Here we
change the first pose parameter so the example does not depend on a particular
naming convention:

```python
model_parameters = np.zeros(
    character.parameter_transform.size,
    dtype=np.float32,
)
pose_parameters = np.flatnonzero(
    character.parameter_transform.pose_parameters
)
if len(pose_parameters) == 0:
    raise ValueError("The character has no pose parameters")

model_parameters[pose_parameters[0]] = 0.35
skeleton_state = pym_geometry.model_parameters_to_skeleton_state(
    character,
    model_parameters,
)
```

For a motion sequence, pass an array with shape `(frames, parameters)`. The
result then has shape `(frames, joints, 8)`.

If the character has a mesh, the same skeleton state can drive linear blend
skinning:

```python
if character.has_mesh:
    posed_vertices = character.skin_points(skeleton_state)
    print(posed_vertices.shape)  # (vertices, 3)
```

See [Core concepts](../01_user_guide/02_core_concepts.md) for the relationship
between model parameters, joint parameters, skeleton states, and posed
vertices. The [geometry API reference](https://facebookresearch.github.io/momentum/python_api_doc/geometry.html)
lists the available loaders and conversions. The public
[model-conversion integration test](https://github.com/facebookresearch/momentum/blob/main/momentum/examples/convert_model/convert_model_integration_test.cpp)
uses the same sample assets and model definition.
