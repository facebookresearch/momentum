# PyMomentum Examples

This directory contains example scripts demonstrating various features of PyMomentum, particularly working with the official Momentum Human Rig (MHR).

## Prerequisites

Before running examples, ensure you have:

1. Built PyMomentum:
   ```bash
   pixi run build_py
   ```

2. Downloaded the official MHR assets:
   ```bash
   pixi run download_assets
   ```

## Available Examples

### load_character.py

Demonstrates how to:
- Download and use official Momentum Human Rig (MHR) assets
- Load MHR Character objects from FBX files at different LODs
- Inspect skeleton structure (127 joints), parameters, and mesh data
- Understand the multi-resolution character models (LOD 0-6)

The MHR provides professional-quality character models with full skeletal rigs,
multi-resolution meshes, and corrective blend shapes suitable for research and production use.

**Run:**
```bash
pixi run -e py312 python pymomentum/examples/load_character.py
```

**Expected output:**
- List of available MHR model files (7 LODs)
- Character skeleton information (127 joints)
- Parameter counts
- Mesh statistics for different LODs

## About the Momentum Human Rig (MHR)

The MHR is a professional-grade character rig released by Meta Reality Labs Research:
- **7 LOD levels** (lod0.fbx to lod6.fbx) from 73K vertices down to minimal geometry
- **127 skeletal joints** with full body articulation
- **Corrective blend shapes** for natural deformation
- **PyTorch model weights** for ML applications

Learn more: https://github.com/facebookresearch/MHR

## Adding New Examples

When creating new examples:
1. Add clear docstrings explaining what the example demonstrates
2. Include usage instructions in the docstring
3. Use the asset download infrastructure from `scripts/download_test_assets.py`
4. Handle missing assets gracefully with informative error messages
5. Update this README with the new example
6. Consider different LODs and their use cases
