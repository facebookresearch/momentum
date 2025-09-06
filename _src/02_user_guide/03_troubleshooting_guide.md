---
sidebar_position: 3
---

# Troubleshooting Guide

Common issues and solutions when working with Momentum.

## FBX Retargeting: Incorrect Bind Pose

**Problem:** FBX shows correctly in `fbx_viewer` but fails to convert to GLB or produces wrong joint transforms.

**Cause:** Retargeting changes joint ordering, making skeleton incompatible with Momentum format.

**Solution:**

The core approach involves creating a joint name mapping between retargeted and expected joint names, then applying this mapping to reorder skeleton states before conversion.

1. **Identify Joint Mapping**: Compare joint names between retargeted FBX and expected character format
2. **Apply Remapping**: Use mapping to reorder skeleton states to match expected format
3. **Convert to Transforms**: Apply `SkeletonState::compare()` to validate and convert to joint transforms

**Validation:**

See `test/io/io_fbx_test.cpp::Fbx_Retargeting_JointRemapping` for a complete working example that:
- Simulates the joint reordering problem (demonstrates failure without remapping)
- Implements the remapping solution (demonstrates success with remapping)
- Validates that remapping significantly improves skeleton state accuracy

**Tools:**
- [`fbx_viewer`](../03_examples/01_viewers.md#fbx-viewer): Visual validation of FBX files
- [`convert_model`](../03_examples/03_convert_model.md): Convert between FBX and GLB formats
- [`glb_viewer`](../03_examples/01_viewers.md#glb-viewer): Validate converted GLB results

**Related Examples:**
- [Viewers Guide](../03_examples/01_viewers.md): Visual validation tools for different file formats
- [Convert Model](../03_examples/03_convert_model.md): Comprehensive FBX/GLB conversion examples
