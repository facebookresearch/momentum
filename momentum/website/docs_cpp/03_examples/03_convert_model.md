---
sidebar_position: 3
---

# Convert Model

The `convert_model` example demonstrates how to convert a character model and its associated animation file between FBX and GLB formats. Use the `-m` or `--model` option to specify the input model, followed by the path to the model file (`.fbx` or `.glb`). If no input model is provided, the tool will automatically read the animation from an existing GLB or FBX file.

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

```
convert_model [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -m,--model TEXT:FILE        Input model (.fbx/.glb); not required if reading animation from glb or fbx
  -p,--parameters TEXT:FILE   Input model parameter file (.model)
  -l,--locator TEXT:FILE      Input locator file (.locators)
  -d,--motion TEXT:FILE       Input motion data file (.mmo/.glb/.fbx)
  -o,--out TEXT REQUIRED      Output file (.fbx/.glb)
  --out-locator TEXT          Output a locator file (.locators)
  --save-markers              Save marker data from motion file in output (glb only)
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
