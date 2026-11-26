---
sidebar_position: 3
---

# Convert Model

The `convert_model` example demonstrates how to convert a character model and its associated animation file between FBX, glTF/GLB and BVH formats. Use the `-m` or `--model` option to specify the input model, followed by the path to the model file (`.fbx`, `.glb`, `.gltf`, `.urdf`, or `.bvh`). If no input model is provided, the tool will automatically read the animation from an existing glTF/GLB, FBX or BVH file given with `-d` or `--motion`; one of the two options is required. The input files are read as described in [Files and options shared by the tools](#files-and-options-shared-by-the-tools).

A BVH file holds the skeleton and the motion of one character. When the output is a BVH file, the tool warns about the data that BVH cannot store, such as the mesh, the physical properties and the metadata.

The `--out-locators-local`, `--out-locators-global` and `--out-parameters` options are deprecated: use the `export_rig` example to save the parameter transform, the parameter limits and the locators of a character. The `-m` option with `-d` is also deprecated: use the `apply_motion` example to apply the motion of one file to the characters of another file.

Files with several characters, such as a body and the objects it holds, are converted with every character and its motion. Use `--character` to convert only one of them. The `--out-locators-local`, `--out-locators-global` and `--out-parameters` outputs apply to one character, so they need `--character` when the input has several characters.

## Files and options shared by the tools

The `convert_model`, `export_rig`, `apply_motion`, `split_characters` and `combine_characters` examples share their code and read files in the same way. See [Character Files](../02_user_guide/04_character_files.md) for the role of each file.

- `-p` applies a `.model` file and `-l` a `.locators` file to a character, in `convert_model`, `export_rig` and `apply_motion`. They replace the rig and the locators that the input file stores. They apply to one character, so they need `--character` when the input has several characters.
- Without `-p`, a character gets the rig that its file stores, such as the rig that momentum saves in glTF and FBX files. The stored rig is a copy: to change a rig, change its `.model` file and give it with `-p`. A character of an FBX file without model parameters, such as a file from another tool, gets the identity parameter transform, so that its joint motion is also its model-parameter motion.
- glTF motion maps onto the rig by parameter name, and onto the identity parameter transform of an FBX model through the joint transforms. FBX and BVH motion is fitted to the rig.
- When the model file has several characters, or with `--character`, each character gets the motion of the motion-file character with the same name.

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

```
convert_model [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -m,--model TEXT:FILE        Input model (.fbx/.glb/.gltf/.urdf/.bvh); not required if reading animation from glb/gltf, fbx or bvh. Every character of the file is converted, unless --character selects one. (Deprecated with --motion: use apply_motion) With several characters or with --character, each character gets the motion of the motion-file character with the same name
  -p,--parameters TEXT:FILE   Input model parameter file (.model)
  -l,--locator TEXT:FILE      Input locator file (.locators)
  -d,--motion TEXT:FILE       Input motion data file (.glb/.gltf/.fbx/.bvh); without --model, every character of the file is converted with its motion, and --parameters and --locator apply to it
  --character TEXT            Convert only the character with this name, from the model file and the motion file; required for --parameters, --locator, --out-locators-local, --out-locators-global and --out-parameters when the input has several characters
  -o,--out TEXT REQUIRED      Output file (.fbx/.glb/.gltf/.bvh); a .bvh file holds one character's skeleton and motion
  --out-locators-local TEXT   (Deprecated: use export_rig) Output a locator file (.locators) in local space for transferring between identities
  --out-locators-global TEXT  (Deprecated: use export_rig) Output a locator file (.locators) in global space for authoring a template
  --out-parameters TEXT       (Deprecated: use export_rig) Output a model parameter file (.model) containing the parameter transform and limits
  --fbx-namespace TEXT        Namespace in output fbx file
  --save-markers              Save marker data from input motion file in the output
  -c,--character-mesh         (FBX Output file only) Saves the Character Mesh to the output file.
```

* Example 1: Convert an fbx model to a glb model.

<OssOnly>
```
pixi run convert_model -m character.fbx -p character.model -l character.locators -o character.glb
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -m character.fbx -p character.model -l character.locators -o character.glb
```
</FbInternalOnly>

* Example 2: Convert a glb file to an fbx file with animation curves only (without mesh).

<OssOnly>
```
pixi run convert_model -d animation.glb -o animation.fbx
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -d animation.glb -o animation.fbx
```
</FbInternalOnly>

* Example 3: Convert an fbx animation to glb with a given model parameter file. There is no guarantee the conversion is lossless. We will simply use InverseParameterTransform for a least square fit of the parameters.

<OssOnly>
```
pixi run convert_model -d animation.fbx -p character.model -o animation.glb
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -d animation.fbx -p character.model -o animation.glb
```
</FbInternalOnly>

* Example 4: Convert a glb file with several characters (for example a hand and an object) to an fbx file with every character and its motion.

<OssOnly>
```
pixi run convert_model -d hand_and_object.glb -o hand_and_object.fbx
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -d hand_and_object.glb -o hand_and_object.fbx
```
</FbInternalOnly>

* Example 5: Convert only the character named `hand` from that file.

<OssOnly>
```
pixi run convert_model -d hand_and_object.glb --character hand -o hand.glb
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -d hand_and_object.glb --character hand -o hand.glb
```
</FbInternalOnly>

* Example 6: Convert the motion of a glb file to a BVH file.

<OssOnly>
```
pixi run convert_model -d animation.glb -o animation.bvh
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -d animation.glb -o animation.bvh
```
</FbInternalOnly>
