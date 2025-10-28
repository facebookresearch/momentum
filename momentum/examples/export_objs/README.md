# Export OBJs Example

This example demonstrates how to export animation data from GLB/GLTF or FBX files as per-frame OBJ mesh files.

## Overview

The `export_objs` utility loads a character with animation from either:
- **GLB/GLTF files**: Using the Momentum GLTF loader
- **FBX files**: Using the Momentum FBX loader

It then exports each frame of the animation as a separate OBJ file containing the deformed mesh geometry. The output folder will be created if it doesn't exist yet.

## Building with buck

```bash
buck2 build path/to/momentum/examples:export_objs
```

## Usage

Using buck:
```bash
buck2 run path/to/momentum/examples:export_objs -- \
  -i <input_file> \
  -o <output_folder> \
  [--first <frame_number>] \
  [--last <frame_number>] \
  [--stride <frame_stride>]
```

Using pixi:
```
pixi run export_objs -i <input_file> -o <output_folder> [--first <frame_number>] [--last <frame_number>] [--stride <frame_stride>]
```

### Options

- `-i, --input`: Path to the input animation file (`.fbx`, `.glb`, or `.gltf`) [required]
- `-o, --output`: Path to the output folder where OBJ files will be saved [required]
- `--first`: First frame to export (default: 0)
- `--last`: Last frame to export, inclusive (default: -1 for all frames)
- `--stride`: Frame stride when exporting (default: 1)

### Examples

**Export all frames from a GLB file:**

Using buck:
```bash
buck2 run path/to/momentum/examples:export_objs -- \
  -i test.glb \
  -o /tmp/exported_objs
```

Using pixi:
```
pixi run export_objs -i test.glb -o /tmp/exported_objs
```

**Export frames 10-50 from an FBX file with stride 2:**

Using buck:
```bash
buck2 run path/to/momentum/examples:export_objs -- \
  -i path/to/animation.fbx \
  -o /tmp/exported_objs \
  --first 10 \
  --last 50 \
  --stride 2
```

Using pixi:
```
pixi run export_objs -i path/to/animation.fbx -o /tmp/exported_objs --first 10 --last 50 --stride 2
```

**Export static mesh (no animation):**
If the input file has no animation data, it will export a single OBJ file with the template mesh.

## Output Format

### Animation Sequences
The exported OBJ files are named sequentially:
- `00000.obj`, `00001.obj`, `00002.obj`, etc.

### Static Mesh (No Animation)
If the input file has no animation data, a single OBJ file is exported with the name:
- `<input_filename>.obj` (e.g., `character.obj` for `character.glb`)

### OBJ File Contents
Each OBJ file contains:
- Vertex positions (`v` lines)
- Triangle faces (`f` lines, 1-indexed)

Note: The simple OBJ exporter in this example does not export:
- Texture coordinates
- Normals
- Materials

## Implementation Notes

### GLB/GLTF Files
- Loads character with motion using `loadCharacterWithMotion()`
- Returns motion as a single `MatrixXf` containing **model parameters**
- Also returns a separate identity parameter (stored in `JointParameters`)
- The identity parameter is constant for the character and not time-varying
- Each column represents one frame's model parameters

### FBX Files
- Loads character with motion using `loadFbxCharacterWithMotion()`
- Returns motion as a `std::vector<MatrixXf>` containing **joint parameters**
- If multiple motions are present, only the first one is exported
- FBX files do not store custom parameter information, so only joint parameters are available

### Parameter Handling Difference
The key difference between GLB and FBX files:
- **GLB**: Motion matrix contains model parameters (stored in custom plugin); identity parameters are separate
- **FBX**: Motion matrix contains joint parameters (no custom storage); model parameters are not available

This affects how parameters are set in `CharacterState`:
- For GLB: `params.offsets = id; params.pose = motion.col(iFrame)` (model parameters)
- For FBX: `params.pose.v.setZero(0); params.offsets = motion.col(iFrame)` (joint parameters)

### Per-Frame Export Process
1. Load the character and animation
2. For each frame:
   - Set the character state with the appropriate parameters (model or joint)
   - Compute the deformed mesh geometry
   - Export the mesh to an OBJ file

## See Also

- `examples/glb_viewer/`: Interactive viewer for GLB files with Rerun
- `examples/fbx_viewer/`: Interactive viewer for FBX files with Rerun
- `examples/convert_model/`: Convert between different model formats
