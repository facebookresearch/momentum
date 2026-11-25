---
sidebar_position: 3
---

# Convert Model

The `convert_model` example demonstrates how to convert a character model and its associated animation file between FBX and GLB formats. Use the `-m` or `--model` option to specify the input model, followed by the path to the model file (`.fbx`, `.glb`, `.gltf`, or `.urdf`). If no input model is provided, the tool will automatically read the animation from an existing GLB or FBX file.

Files with several characters, such as a body and the objects it holds, are converted with every character and its motion. Use `--character` to convert only one of them; the parameter and locator inputs and outputs (`-p`, `-l`, `--out-locators-local`, `--out-locators-global` and `--out-parameters`) apply to one character, so they need `--character` when the input has several characters. When the model file has several characters, or with `--character`, each character gets the motion of the motion-file character with the same name.

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

```
convert_model [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -m,--model TEXT:FILE        Input model (.fbx/.glb/.gltf/.urdf); not required if reading animation from glb or fbx. Every character of the file is converted, unless --character selects one. With several characters or with --character, each character gets the motion of the motion-file character with the same name
  -p,--parameters TEXT:FILE   Input model parameter file (.model)
  -l,--locator TEXT:FILE      Input locator file (.locators)
  -d,--motion TEXT:FILE       Input motion data file (.glb/.fbx); without --model, every character of the file is converted with its motion
  --character TEXT            Convert only the character with this name, from the model file and the motion file; required for --parameters, --locator, --out-locators-local, --out-locators-global and --out-parameters when the input has several characters
  -o,--out TEXT REQUIRED      Output file (.fbx/.glb/.gltf)
  --out-locators-local TEXT   Output a locator file (.locators) in local space for transferring between identities
  --out-locators-global TEXT  Output a locator file (.locators) in global space for authoring a template
  --out-parameters TEXT       Output a model parameter file (.model) containing the parameter transform and limits
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

* Example 4: Apply animation from an s0 model to a s4 model for high-res rendering (ie. with mesh). The `.model` file is needed if the target model is in `.fbx` format so we know how to map the input motion.

<OssOnly>
```
pixi run convert_model -m character.fbx -p character.model -d animation_s0.glb -o animation_s4.fbx -c
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -m character.fbx -p character.model -d animation_s0.glb -o animation_s4.fbx -c
```
</FbInternalOnly>

* Example 5: Convert a glb file with several characters (for example a hand and an object) to an fbx file with every character and its motion.

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

* Example 6: Convert only the character named `hand` from that file, and write its locators.

<OssOnly>
```
pixi run convert_model -d hand_and_object.glb --character hand -o hand.glb --out-locators-local hand.locators
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt convert_model -- -d hand_and_object.glb --character hand -o hand.glb --out-locators-local hand.locators
```
</FbInternalOnly>
