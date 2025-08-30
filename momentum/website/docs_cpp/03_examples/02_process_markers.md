---
sidebar_position: 2
---

# Process Markers

## Optical Marker based Body Tracking

This project provides a set of core functions to solve for body motions based on optical marker inputs. It supports all PC OSes (untested on mobile).
The `marker_tracker` lib contains core functionalities for downstream applications to build on. Demo applications are provided to show how they can be used to build your data processing pipeline. `process_markers_app` solves for body motion given an input marker sequence, with or without an existing calibrated skeleton. `refine_motion` runs smoothing as a post process to fill in missing data from input. They can be used to batch process mocap data in a python script.

:::info

The Momentum ecosystem implicitly operates in *centimeter*. If you are working with c3d files, we will do the unit conversion based on the stored unit on file. However, if you are using our API with your own data, make sure to convert them into cm. We also assume a Y-up coordinate system, which is not the industry convention (i.e., Z-up).

:::

<FbInternalOnly>

:::note
The buck mode `@arvr/mode/win/opt` is for Windows. Use `@arvr/mode/mac-arm/opt` for macOS or `@arvr/mode/platform010/opt` for Linux.
:::

</FbInternalOnly>

## Example use cases

Get the full list of options for each application with `-h` or `--help` argument. `02_01.c3d` is an example input file used by the default config files. Note that a config file can be used together with command line options. The command line overwrites values in the config file.

## Track a marker sequence without a calibrated model.

The first step in tracking a marker file is to calibrate the subject's proportions and the markers' placement. It requires a `.locators` file that defines a template of marker layout on the body. We have a template file with common layouts from Vicon and OptiTrack. There is usually a Range-of-Motion (ROM) sequence captured for this calibration purpose.

Use a config file:

<OssOnly>
```
pixi run process_markers -c process_markers_calib.config
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt process_markers_app -- -c process_markers_calib.config
```
</FbInternalOnly>

Setting the `calibrate` option to true will first calibrate the skeleton and the marker layout, then use the calibrated model for motion tracking.
<FbInternalOnly>
The official template models are in [`momentum/models`](https://www.internalfb.com/code/fbsource/arvr/libraries/momentum/models/).
</FbInternalOnly>

## Track a marker sequence with a calibrated model.

The tracking result from the above calibration step contains the calibrated model, and it can then be used to track other motion data from the same subject, without running the calibration step again. We currently only support saving/loading calibrated models in `.glb` format.

Use a config file:

<OssOnly>
```
pixi run process_markers -c process_markers_tracking.config
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt process_markers_app -- -c process_markers_tracking.config
```
</FbInternalOnly>

Use cli arguments:

<OssOnly>
```
pixi run process_markers -i input.c3d -o tracked.glb --model calibrated_model.glb --calibrate false
```
</OssOnly>

<FbInternalOnly>
```
buck run @arvr/mode/win/opt process_markers_app -- -i input.c3d -o tracked.glb --model calibrated_model.glb --calibrate false
```
</FbInternalOnly>
