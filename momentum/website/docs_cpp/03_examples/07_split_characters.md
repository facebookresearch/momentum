---
sidebar_position: 7
---

# Split Characters

The `split_characters` example saves each character of a file, such as a hand and the object it holds, with its motion to a file of its own. Use the `-i` or `--input` option to specify the input file (`.glb`, `.gltf`, `.fbx`, or `.bvh`), and `--out-dir` for the output directory. Each file has the name of its character: `_` replaces the characters other than letters, digits, `-`, `_` and `.`, a character without a name is saved as `character_<index>`, and a numeric suffix keeps the file names unique in any letter case. The output files have the format of the input file, unless `--format` selects another one (`fbx`, `glb`, `gltf`, or `bvh`). The markers of the input file are not saved.

Each output file has the content that `convert_model -d <input> --character <name>` writes. The input file is read as described in [Files and options shared by the tools](./03_convert_model.md#files-and-options-shared-by-the-tools).

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

```
split_characters [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -i,--input TEXT:FILE REQUIRED
                              Input file (.glb/.gltf/.fbx/.bvh); a .bvh file holds one character
  --out-dir TEXT REQUIRED     Output directory; each character is saved with its motion to <character name>.<format> in it
  --format TEXT               Format of the output files: fbx, glb, gltf or bvh (default: the format of the input file)
  --fbx-namespace TEXT        Namespace in output fbx files
  -c,--character-mesh         (FBX Output files only) Saves the Character Mesh to the output files.
```

* Example 1: Save the hand and the object of a glb file to two glb files.

<OssOnly>
```
pixi run split_characters -i hand_and_object.glb --out-dir characters
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:split_characters -- -i hand_and_object.glb --out-dir characters
```
</FbInternalOnly>

* Example 2: Save them to BVH files, which hold the skeleton and the motion of one character.

<OssOnly>
```
pixi run split_characters -i hand_and_object.glb --out-dir characters --format bvh
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:split_characters -- -i hand_and_object.glb --out-dir characters --format bvh
```
</FbInternalOnly>
