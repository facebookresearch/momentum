---
sidebar_position: 8
---

# Combine Characters

The `combine_characters` example saves the characters of several files with their motion to one file, for example a hand and the object it holds. Use the `-i` or `--input` option for each input file (`.glb`, `.gltf`, `.fbx`, `.bvh`, or `.urdf`), and `-o` or `--out` for the output file. The input files are read as described in [Files and options shared by the tools](./03_convert_model.md#files-and-options-shared-by-the-tools). The motion of each character starts at the first frame of the output. A character with the name of an earlier character is saved with a numeric suffix, and a warning tells the new name. Use `--markers` to save the markers of a marker file (`.c3d`, `.trc`, `.glb`, `.gltf`, or `.fbx`) in the output.

The output has one frame rate. Inputs with motion at different frame rates are an error, unless `--resample` is given: the motion is then resampled to the frame rate of the first input with motion, or to the frame rate given with `--fps`. A warning tells each resampled input: each frame of the output is at the same time from the start, and its model parameters are linearly interpolated between the two nearest frames of the input. The markers must have the frame rate of the output; to keep them, resample the motion to their frame rate with `--resample --fps <frame rate of the markers>`.

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

```
combine_characters [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -i,--input TEXT:FILE ... REQUIRED
                              Input files (.glb/.gltf/.fbx/.bvh/.urdf); every character of each file is saved with its motion, which starts at the first frame of the output
  -o,--out TEXT REQUIRED      Output file (.fbx/.glb/.gltf/.bvh); a .bvh file holds one character's skeleton and motion
  --markers TEXT:FILE         Marker file (.c3d/.trc/.glb/.gltf/.fbx) whose markers are saved in the output; its frame rate must be the frame rate of the output
  --resample                  Resample the motion of the inputs with another frame rate than the output, by linear interpolation of the model parameters; without it, inputs with different frame rates are an error
  --fps FLOAT:POSITIVE Needs: --resample
                              Frame rate of the output with --resample (default: the frame rate of the first input with motion)
  --fbx-namespace TEXT        Namespace in output fbx file
  -c,--character-mesh         (FBX Output file only) Saves the Character Mesh to the output file.
```

* Example 1: Save a hand and an object to one glb file.

<OssOnly>
```
pixi run combine_characters -i hand.glb -i object.glb -o hand_and_object.glb
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:combine_characters -- -i hand.glb -i object.glb -o hand_and_object.glb
```
</FbInternalOnly>

* Example 2: Save a hand with motion at 120 fps and an object with motion at 30 fps to one file at 120 fps, with the markers of a capture at 120 fps.

<OssOnly>
```
pixi run combine_characters -i hand.glb -i object.fbx --resample --markers capture.c3d -o hand_and_object.glb
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt //arvr/libraries/momentum/examples:combine_characters -- -i hand.glb -i object.fbx --resample --markers capture.c3d -o hand_and_object.glb
```
</FbInternalOnly>
