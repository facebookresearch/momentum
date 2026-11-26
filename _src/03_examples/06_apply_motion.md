---
sidebar_position: 6
---

# Apply Motion

The `apply_motion` example applies the motion of a motion file to the characters of a model file, and saves them in FBX, glTF/GLB or BVH format. Use the `-m` or `--model` option to specify the model file (`.fbx`, `.glb`, `.gltf`, `.urdf`, or `.bvh`), and the `-d` or `--motion` option to specify the motion file (`.glb`, `.gltf`, `.fbx`, or `.bvh`); a motion file of another format is an error. The files are read, and `-p`, `-l` and `--character` work, as described in [Files and options shared by the tools](./03_convert_model.md#files-and-options-shared-by-the-tools); `-p` and `-l` apply to the character of the model file.

This example replaces the `-m` option with `-d` of `convert_model`, which is deprecated; both tools write the same files.

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

```
apply_motion [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -m,--model TEXT:FILE REQUIRED
                              Input character file (.fbx/.glb/.gltf/.urdf/.bvh) that gets the motion. With several characters or with --character, each character gets the motion of the motion-file character with the same name
  -p,--parameters TEXT:FILE   Input model parameter file (.model)
  -l,--locator TEXT:FILE      Input locator file (.locators)
  -d,--motion TEXT:FILE REQUIRED
                              Input motion data file (.glb/.gltf/.fbx/.bvh)
  --character TEXT            Apply the motion only to the character with this name, from the model file and the motion file; required for --parameters and --locator when the model file has several characters
  -o,--out TEXT REQUIRED      Output file (.fbx/.glb/.gltf/.bvh); a .bvh file holds one character's skeleton and motion
  --fbx-namespace TEXT        Namespace in output fbx file
  --save-markers              Save marker data from input motion file in the output
  -c,--character-mesh         (FBX Output file only) Saves the Character Mesh to the output file.
```

* Example 1: Apply animation from an s0 model to a s4 model for high-res rendering (ie. with mesh). With the `.model` file, the fbx model gets the parameter transform, and the glb motion maps onto it by parameter name. Without `-p`, an fbx model that momentum wrote uses the rig that it was saved with.

<OssOnly>
```
pixi run apply_motion -m character.fbx -p character.model -d animation_s0.glb -o animation_s4.fbx -c
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:apply_motion -- -m character.fbx -p character.model -d animation_s0.glb -o animation_s4.fbx -c
```
</FbInternalOnly>

* Example 2: Apply the motion of a BVH file to an fbx model.

<OssOnly>
```
pixi run apply_motion -m character.fbx -p character.model -d animation.bvh -o animation.fbx
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:apply_motion -- -m character.fbx -p character.model -d animation.bvh -o animation.fbx
```
</FbInternalOnly>
