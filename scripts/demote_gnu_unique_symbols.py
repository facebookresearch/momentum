#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Demote STB_GNU_UNIQUE dynamic symbols in an ELF shared library to STB_GLOBAL.

The Linux wheels bundle a private libstdc++. Renaming its SONAME is not enough
to keep it private: libstdc++ defines its locale facet ids and other template
statics as STB_GNU_UNIQUE, and glibc binds each UNIQUE symbol to the first
definition loaded in the process, even across RTLD_LOCAL. If another C++
runtime (for example the one PyTorch loads) is loaded first, the private copy
indexes its own locales with the other copy's facet ids, which can crash.
Demoting the symbols to STB_GLOBAL makes the private copy bind to itself.

Usage: demote_gnu_unique_symbols.py LIB...
"""

import struct
import sys
from pathlib import Path

STB_GLOBAL = 1
STB_GNU_UNIQUE = 10
SHT_DYNSYM = 11
ELF64_LITTLE_ENDIAN = bytes([0x7F, 0x45, 0x4C, 0x46, 2, 1])


def _unique_symbol_offsets(data: bytes) -> list[int]:
    """Return the file offsets of st_info for every UNIQUE dynamic symbol."""
    if data[:6] != ELF64_LITTLE_ENDIAN:
        raise ValueError("not an ELF64 little-endian file")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    offsets = []
    for index in range(shnum):
        header = shoff + index * shentsize
        (sh_type,) = struct.unpack_from("<I", data, header + 4)
        if sh_type != SHT_DYNSYM:
            continue
        sh_offset, sh_size = struct.unpack_from("<QQ", data, header + 0x18)
        (sh_entsize,) = struct.unpack_from("<Q", data, header + 0x38)
        for symbol in range(sh_offset, sh_offset + sh_size, sh_entsize):
            if data[symbol + 4] >> 4 == STB_GNU_UNIQUE:
                offsets.append(symbol + 4)
    return offsets


def count_unique_symbols(data: bytes) -> int:
    return len(_unique_symbol_offsets(data))


def demote_unique_symbols(path: Path) -> int:
    """Rewrite the file's UNIQUE dynamic symbols as GLOBAL; return how many."""
    data = bytearray(path.read_bytes())
    offsets = _unique_symbol_offsets(data)
    for offset in offsets:
        data[offset] = (STB_GLOBAL << 4) | (data[offset] & 0xF)
    if offsets:
        path.write_bytes(data)
    return len(offsets)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in argv:
        count = demote_unique_symbols(Path(arg))
        print(f"{arg}: demoted {count} STB_GNU_UNIQUE symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
