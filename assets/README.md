# Momentum Human Rig (MHR) Assets

This directory contains the official Momentum Human Rig (MHR) assets required for creating and working with the foundational character model. These assets are automatically downloaded from the [MHR GitHub release](https://github.com/facebookresearch/MHR/releases/tag/v1.0.0).

## Contents

After running `pixi run download_assets`, this directory will contain:
- **Character Models (FBX)**: Multi-resolution character models at various LODs (lod0.fbx through lod6.fbx)
- **Corrective Blend Shapes**: NPZ files with blend shape data for natural deformation
- **Model Configuration**: `compact_v6_1.model` - Character configuration file
- **PyTorch Weights**: `mhr_model.pt` - Pre-trained model weights
- **Activation Data**: `corrective_activation.npz` - Blend shape activation patterns

## Management

**Download MHR assets:**
```bash
pixi run download_assets
```

**Clean up:**
```bash
pixi run clean_assets
```

## Usage

These official MHR assets are used by:

1. **Tests** - Tests marked with `@pytest.mark.requires_assets`
   ```bash
   pixi run test_py_with_assets
   ```

2. **Python Examples** - Working examples with real character models
   ```bash
   pixi run -e py312 python pymomentum/examples/load_character.py
   ```

3. **Your Applications** - Load MHR characters in your own code:
   ```python
   from pathlib import Path
   import pymomentum.geometry as pym_geometry

   # Load LOD 0 (highest quality)
   mhr_assets = Path("assets")
   char = pym_geometry.Character.load_fbx(str(mhr_assets / "assets" / "lod0.fbx"))
   ```

## Size

The MHR assets are approximately 100MB compressed and will expand to roughly the same size when extracted.

## Asset Details

### Character LODs (Level of Detail)
- **lod0.fbx** - Highest quality (73,639 vertices, 127 joints)
- **lod1.fbx** - High quality (18,439 vertices, 127 joints)
- **lod2.fbx** - Medium quality
- **lod3.fbx** - Medium-low quality
- **lod4.fbx** - Low quality
- **lod5.fbx** - Very low quality
- **lod6.fbx** - Lowest quality

### Blend Shapes
Corrective blend shapes for each LOD provide natural deformation during animation, improving realism for joint bending and body shape variations.

## Citation

If you use the Momentum Human Rig in your research, please cite:
```
@misc{MHR2024,
  title={Momentum Human Rig},
  author={Meta Reality Labs Research},
  year={2024},
  url={https://github.com/facebookresearch/MHR}
}
```
