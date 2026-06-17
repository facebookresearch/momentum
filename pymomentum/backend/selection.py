# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

import torch

try:
    import triton
except ImportError:
    triton = None

_HAS_TRITON: bool = triton is not None

# Backend override. `is_tracing()` only catches code that runs *inside*
# torch.jit.trace; exporters also run the model eagerly before tracing (a warm-up
# forward to settle device placement), and that eager run would pick Triton on
# CUDA -- which then can't be traced. `force_backend("torch")` lets the exporter
# pin the backend across the whole export (warm-up + trace). A ContextVar (not a
# plain global) keeps the override thread- and async-local, so it can't leak into
# a concurrent thread's resolve_backend (e.g. dataloader/inference) mid-export.
_FORCED_BACKEND: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "pymomentum_forced_backend", default=None
)


@contextlib.contextmanager
def force_backend(backend: str | None) -> Iterator[None]:
    """Force `resolve_backend` to return `backend` within this context.

    Used by model export to keep the entire export (the eager warm-up forward and
    the subsequent jit.trace) on the traceable torch backend, since Triton kernels
    are not traceable. Pass None to clear the override. The override is scoped to
    the current thread / async context.
    """
    token = _FORCED_BACKEND.set(backend)
    try:
        yield
    finally:
        _FORCED_BACKEND.reset(token)


def resolve_backend(
    backend: str,
    tensor: torch.Tensor,
    use_double_precision: bool = False,
) -> str:
    # Triton kernels are not traceable/scriptable, so fall back to pure torch
    # under TorchScript/trace so the graph captures cleanly. Keep is_scripting()
    # as a standalone early-return: the scripting compiler treats it as
    # compile-time True and prunes everything after the return, so the
    # is_tracing() check below is never compiled under torch.jit.script (where it
    # may not be scriptable). is_tracing() then handles torch.jit.trace (export).
    if torch.jit.is_scripting():
        return "torch"
    if torch.jit.is_tracing():
        return "torch"
    # Eager-mode export override (set via force_backend). After the jit gates so
    # the scripting compiler still prunes it (the ContextVar access is not
    # scriptable, but is_scripting() returns above under torch.jit.script).
    forced = _FORCED_BACKEND.get()
    if forced is not None:
        return forced
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


def resolve_fk_backend(
    backend: str,
    tensor: torch.Tensor,
    use_double_precision: bool = False,
) -> str:
    resolved = resolve_backend(backend, tensor, use_double_precision)
    if resolved == "triton" and use_double_precision:
        raise RuntimeError(
            "Triton FK does not support double precision; "
            "pass use_double_precision=False"
        )
    return resolved
