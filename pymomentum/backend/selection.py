# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import torch

try:
    import triton
except ImportError:
    triton = None

_HAS_TRITON: bool = triton is not None


def resolve_backend(
    backend: str,
    tensor: torch.Tensor,
    use_double_precision: bool = False,
) -> str:
    if torch.jit.is_scripting():
        return "torch"
    if backend != "auto":
        return backend
    if (
        not use_double_precision
        and tensor.is_cuda
        and tensor.dtype == torch.float32
        and _HAS_TRITON
    ):
        return "triton"
    return "torch"
