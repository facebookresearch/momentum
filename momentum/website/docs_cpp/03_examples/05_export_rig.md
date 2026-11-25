---
sidebar_position: 5
---

# Export Rig

The `export_rig` example saves the parameter transform, the parameter limits and the locators of a character to `.model` and `.locators` files. Use the `-m` or `--model` option to specify the input character file (`.fbx`, `.glb`, `.gltf`, `.urdf`, or `.bvh`). The `-p` and `-l` options apply a model parameter file and a locator file to the character before the export, as in `convert_model`. When the file has several characters, use `--character` to select the one to export.

These outputs replace the deprecated `--out-locators-local`, `--out-locators-global` and `--out-parameters` options of `convert_model`; both tools write the same files.

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

```
export_rig [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -m,--model TEXT:FILE REQUIRED
                              Input character file (.fbx/.glb/.gltf/.urdf/.bvh)
  -p,--parameters TEXT:FILE   Input model parameter file (.model)
  -l,--locator TEXT:FILE      Input locator file (.locators)
  --character TEXT            Export the character with this name; required when the input has several characters
  --out-locators-local TEXT   Output a locator file (.locators) in local space for transferring between identities
  --out-locators-global TEXT  Output a locator file (.locators) in global space for authoring a template
  --out-parameters TEXT       Output a model parameter file (.model) containing the parameter transform and limits
```

* Example 1: Export the parameter transform, the parameter limits and the locators of a glb model.

<OssOnly>
```
pixi run export_rig -m character.glb --out-parameters character.model --out-locators-local character.locators
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:export_rig -- -m character.glb --out-parameters character.model --out-locators-local character.locators
```
</FbInternalOnly>

* Example 2: Export the locators of the character named `hand` in global space, for authoring a template.

<OssOnly>
```
pixi run export_rig -m hand_and_object.glb --character hand --out-locators-global hand.locators
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:export_rig -- -m hand_and_object.glb --character hand --out-locators-global hand.locators
```
</FbInternalOnly>
