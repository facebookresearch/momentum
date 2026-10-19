---
sidebar_position: 2
---

# Core concepts

PyMomentum represents a character at several levels. Most workflows load a
`Character`, manipulate its model parameters, and convert those parameters to
joint transforms or a posed mesh.

## What is in a character?

A `pymomentum.geometry.Character` can contain:

- a skeleton and its joint hierarchy;
- a parameter transform describing how named model parameters drive joints;
- a mesh, skin weights, and blend shapes;
- locators, parameter limits, and collision geometry.

Not every character has every component. For example, a character loaded from
URDF may have a skeleton but no mesh.

## Pose representations

| Representation | Shape | Contents |
| --- | --- | --- |
| Model parameters | `(..., P)` | Rig-specific, named controls. This is usually the best representation to optimize or save. |
| Joint parameters | `(..., J * 7)` or `(..., J, 7)` | Translation, XYZ Euler rotation, and log2 scale for every joint. |
| Skeleton state | `(..., J, 8)` | Global translation, quaternion in `[x, y, z, w]` order, and scale. |
| Posed vertices | `(..., V, 3)` | Mesh vertices after forward kinematics and skinning. |

`P`, `J`, and `V` depend on the character. Leading dimensions are batch
dimensions, so the same functions work for a single pose, a motion sequence, or
a batch of sequences.

The usual data flow is:

```text
model parameters
    -> parameter transform
joint parameters
    -> forward kinematics
skeleton state
    -> linear blend skinning
posed vertices
```

In NumPy code, that flow looks like this:

```python
import numpy as np
import pymomentum.geometry as pym_geometry

model_parameters = np.zeros(
    character.parameter_transform.size,
    dtype=np.float32,
)
joint_parameters = character.parameter_transform.apply(
    model_parameters,
    flatten=False,
)
skeleton_state = pym_geometry.joint_parameters_to_skeleton_state(
    character,
    joint_parameters,
)
if character.has_mesh:
    posed_vertices = character.skin_points(skeleton_state)
```

For convenience,
`pym_geometry.model_parameters_to_skeleton_state(character, model_parameters)`
combines the middle two operations.

:::note
Euler angles and quaternions do not have unique numeric representations. When
checking whether two poses are equivalent, compare their transformation
matrices or transformed points rather than comparing angles or quaternion
components directly.
:::

## Choosing an API

- Use `pymomentum.geometry` for character I/O, NumPy forward kinematics, and
  skinning.
- Use `pymomentum.solver2` for inverse kinematics and sequence optimization.
- Use `pymomentum.torch.character` for batched, differentiable forward
  kinematics and skinning in PyTorch.

The differentiable solver API is experimental. Most inverse-kinematics users
should start with `solver2`.

See the [geometry API reference](https://facebookresearch.github.io/momentum/python_api_doc/geometry.html)
for the complete set of conversions and character operations.
