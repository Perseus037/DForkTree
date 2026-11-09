# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import torch, triton
import triton.language as tl
import fc1_fused as F

@triton.jit
def conv(CS, X, W, B, CI, Slot, ISlot, IW, Y, C: tl.constexpr, CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr, I3: tl.constexpr, BIAS: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < C
    slot = tl.load(Slot)
    dst = tl.load(ISlot)
    acc = tl.full((BC,), 0, tl.float32)
    for j in tl.static_range(4):
        col = tl.load(CI + n * 4 + j)
        old = col < 3
        x0 = tl.load(CS + slot * CS0 + c * CS1 + col * CS2, mask=m & old, other=0).to(tl.float32)
        x1 = tl.load(X + c * X1 + (col - 3) * X2, mask=m & ~old, other=0).to(tl.float32)
        value = tl.where(old, x0, x1)
        weight = tl.load(W + c * 4 + j, mask=m, other=0).to(tl.float32)
        acc += value * weight
        if j > 0:
            tl.store(IW + dst * I0 + n * I1 + c * I2 + (j - 1) * I3, value, mask=m)
    if BIAS:
        acc += tl.load(B + c, mask=m, other=0).to(tl.float32)
    tl.store(Y + n * C + c, acc / (1 + tl.exp(-acc)), mask=m)

def install():
    from sglang.srt.layers.attention.linear import gdn_backend as backend
    original = backend.causal_conv1d_update

    def call(x, conv_state, weight, bias=None, activation=None, **kw):
        if F.NATIVE_REFERENCE or x.ndim != 3 or x.shape[0] != 1 or (x.shape[-1] != F.NN):
            return original(x, conv_state, weight, bias, activation, **kw)
        assert weight.shape[-1] == 4 and activation in ['silu', 'swish', True]
        if F.CTX is None:
            F.CTX = F.Context()
        if not hasattr(F.CTX, 'cidx'):
            F.CTX.cidx = torch.cat((torch.arange(3, device=x.device).expand(F.NN, 3), torch.arange(3, 3 + F.NN, device=x.device)[:, None]), 1).int()
        out = torch.empty((1, F.NN, x.shape[1]), device=x.device, dtype=x.dtype)
        iw = kw['intermediate_conv_window']
        conv[F.NN, triton.cdiv(x.shape[1], 256)](conv_state, x, weight, bias if bias is not None else x, F.CTX.cidx, kw['conv_state_indices'], kw['intermediate_state_indices'], iw, out, x.shape[1], *conv_state.stride(), x.stride(1), x.stride(2), *iw.stride(), BIAS=bias is not None, BC=256, num_warps=4)
        F.CTX.current_mixed = out[0]
        return out.transpose(1, 2)
    backend.causal_conv1d_update = call
