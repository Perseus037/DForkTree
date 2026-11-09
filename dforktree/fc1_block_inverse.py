# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
"""16x16 forward substitution plus 4-block triangular merge.
Diagonal recurrence follows FLA solve_tril (MIT, Songlin Yang et al.).
Adapted to head-major 64-node ancestor-masked matrices.
"""
import triton
import triton.language as tl

@triton.jit
def diagonal_inverse(a):
    r = tl.arange(0, 16)
    lower = r[:, None] > r[None, :]
    inv = -tl.where(lower, a, 0.0)
    for i in range(2, 16):
        row = -tl.sum(tl.where((r == i)[:, None], a, 0.0), axis=0)
        row = tl.where(r < i, row, 0.0)
        row = row + tl.sum(row[:, None] * inv, axis=0)
        inv = tl.where((r == i)[:, None], row[None, :], inv)
    return inv + (r[:, None] == r[None, :]).to(tl.float32)

@triton.jit
def load_block(P, N, bi: tl.constexpr, bj: tl.constexpr):
    r = tl.arange(0, 16)
    return tl.load(P + (bi * 16 + r[:, None]) * N + bj * 16 + r[None, :])

@triton.jit
def store_block(P, N, b, bi: tl.constexpr, bj: tl.constexpr):
    r = tl.arange(0, 16)
    tl.store(P + (bi * 16 + r[:, None]) * N + bj * 16 + r[None, :], b)

@triton.jit
def mm(a, b):
    return tl.dot(a, b, input_precision='tf32x3')

@triton.jit
def invert64(P, N: tl.constexpr):
    P += tl.program_id(0) * N * N
    d0 = diagonal_inverse(load_block(P, N, 0, 0))
    d1 = diagonal_inverse(load_block(P, N, 1, 1))
    d2 = diagonal_inverse(load_block(P, N, 2, 2))
    d3 = diagonal_inverse(load_block(P, N, 3, 3))
    a10 = load_block(P, N, 1, 0)
    a20 = load_block(P, N, 2, 0)
    a21 = load_block(P, N, 2, 1)
    a30 = load_block(P, N, 3, 0)
    a31 = load_block(P, N, 3, 1)
    a32 = load_block(P, N, 3, 2)
    b10 = -mm(d1, mm(a10, d0))
    b20 = -mm(d2, mm(a20, d0) + mm(a21, b10))
    b21 = -mm(d2, mm(a21, d1))
    b30 = -mm(d3, mm(a30, d0) + mm(a31, b10) + mm(a32, b20))
    b31 = -mm(d3, mm(a31, d1) + mm(a32, b21))
    b32 = -mm(d3, mm(a32, d2))
    store_block(P, N, d0, 0, 0)
    store_block(P, N, d1, 1, 1)
    store_block(P, N, d2, 2, 2)
    store_block(P, N, d3, 3, 3)
    store_block(P, N, b10, 1, 0)
    store_block(P, N, b20, 2, 0)
    store_block(P, N, b21, 2, 1)
    store_block(P, N, b30, 3, 0)
    store_block(P, N, b31, 3, 1)
    store_block(P, N, b32, 3, 2)
    for i in tl.static_range(4):
        for j in tl.static_range(i + 1, 4):
            store_block(P, N, tl.full((16, 16), 0.0, tl.float32), i, j)

@triton.jit
def invert32(P, N: tl.constexpr):
    P += tl.program_id(0) * N * N
    d0 = diagonal_inverse(load_block(P, N, 0, 0))
    d1 = diagonal_inverse(load_block(P, N, 1, 1))
    b10 = -mm(d1, mm(load_block(P, N, 1, 0), d0))
    store_block(P, N, d0, 0, 0)
    store_block(P, N, d1, 1, 1)
    store_block(P, N, b10, 1, 0)
    store_block(P, N, tl.full((16, 16), 0.0, tl.float32), 0, 1)

@triton.jit
def invert16(P, N: tl.constexpr):
    P += tl.program_id(0) * N * N
    store_block(P, N, diagonal_inverse(load_block(P, N, 0, 0)), 0, 0)
