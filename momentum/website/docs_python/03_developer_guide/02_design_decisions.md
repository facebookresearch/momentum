---
sidebar_position: 2
---

# Design Decisions

## NumPy vs PyTorch Tensors

PyMomentum mixes NumPy arrays and PyTorch tensors throughout the codebase. This explains the reasoning and current status.

### Why the Mixed Approach?

**Old Reason (Historical):**
* Auto-conversion from Eigen types to NumPy arrays works natively in pybind11
* Can use Python buffer interface to wrap data without copies
* PyTorch tensors are "more painful to work with" in pybind11
* Used PyTorch only where differentiability was needed

**Internal Convention:**
* PyTorch tensor = differentiable operation
* NumPy array = non-differentiable operation

**New Reason (Current):**
The dependence on aten makes building Python code that uses torch.Tensor somewhat challenging compared to code that uses the basic buffer interface:
* Blocking pymomentum usage in downstream projects
* Preventing demonstration projects
* Causing various compatibility problems

### Current State

The codebase is mixed which can be confusing because we're between approaches. Discussions are happening around:

* Stripping all PyTorch from `pymomentum.geometry`
* Isolating differentiability (and hence PyTorch dependencies) in specific libraries, such as `diff_solver`
* Using `GPU_character` for ML workloads instead

**Examples:**
* `solver2` uses NumPy arrays (not differentiable, PyTorch-independent)
* Batching support being removed from non-ML contexts (like rendering)
* Rendering code will strip batch support for open sourcing

**PyTorch advantages being lost:**
* Extra tensor manipulation functionality
* Batching/unbatching support
* But interfaces become more confusing for non-ML use

### Future Directions

**Potential approach:** If you need batching + differentiability + GPU support, use `GPU_character` instead of pymomentum. This would let us:
* Strip enormous amounts of code from pymomentum
* Fix build system issues
* Simplify interfaces

(Obviously could only happen after open sourcing `GPU_character`)

**Current recommendation:** Accept manual conversions between NumPy/PyTorch until the architecture stabilizes.
