# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import math
import numpy as np
try:
    import torch
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

def available():
    return _HAS_TRITON and torch.cuda.is_available()
if _HAS_TRITON:

    @triton.jit
    def _tree_conv_kernel(cs_ptr, mx_ptr, cidx_ptr, w_ptr, b_ptr, y_ptr, C, s_cs_c, s_cs_t, s_mx_n, s_mx_c, s_y_n, KC: tl.constexpr, BC: tl.constexpr, HAS_BIAS: tl.constexpr):
        n = tl.program_id(0)
        offs = tl.program_id(1) * BC + tl.arange(0, BC)
        m = offs < C
        acc = tl.zeros([BC], dtype=tl.float32)
        for j in tl.static_range(KC):
            col = tl.load(cidx_ptr + n * KC + j)
            w = tl.load(w_ptr + offs * KC + j, mask=m, other=0.0).to(tl.float32)
            st = col < KC - 1
            xs = tl.load(cs_ptr + offs * s_cs_c + col * s_cs_t, mask=m & st, other=0.0).to(tl.float32)
            xm = tl.load(mx_ptr + (col - (KC - 1)) * s_mx_n + offs * s_mx_c, mask=m & (col >= KC - 1), other=0.0).to(tl.float32)
            acc += w * tl.where(st, xs, xm)
        if HAS_BIAS:
            acc += tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
        y = acc / (1.0 + tl.exp(-acc))
        tl.store(y_ptr + n * s_y_n + offs, y.to(y_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _tree_gdn_kernel(y_ptr, aa_ptr, bb_ptr, alog_ptr, dtb_ptr, s0_ptr, anc_ptr, o_ptr, kn_ptr, vn_ptr, gn_ptr, neg_ptr, N, s_y_n, s_a_n, s_b_n, s_s_h, s_s_k, s_s_v, s_o_n, s_o_h, KD, scale, R: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BN: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr, CACHED: tl.constexpr):
        h = tl.program_id(0)
        hk = h // R
        on = tl.arange(0, BN)
        mn = on < N
        ok_ = tl.arange(0, DK)
        ov = tl.program_id(1) * BV + tl.arange(0, BV)
        q = tl.load(y_ptr + on[:, None] * s_y_n + (hk * DK + ok_)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        k = tl.load(y_ptr + on[:, None] * s_y_n + (KD + hk * DK + ok_)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        v = tl.load(y_ptr + on[:, None] * s_y_n + (2 * KD + h * DV + ov)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        q = q * (1.0 / tl.sqrt(tl.sum(q * q, 1) + 1e-06))[:, None] * scale
        k = k * (1.0 / tl.sqrt(tl.sum(k * k, 1) + 1e-06))[:, None]
        a = tl.load(aa_ptr + on * s_a_n + h, mask=mn, other=0.0).to(tl.float32) + tl.load(dtb_ptr + h).to(tl.float32)
        sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(tl.minimum(a, 20.0))))
        decay = tl.load(neg_ptr + h).to(tl.float32) if CACHED else -tl.exp(tl.load(alog_ptr + h).to(tl.float32))
        gl = tl.where(mn, decay * sp, 0.0)
        bl = tl.load(bb_ptr + on * s_b_n + h, mask=mn, other=0.0).to(tl.float32)
        beta = tl.where(mn, 1.0 / (1.0 + tl.exp(-bl)), 0.0)
        anc = tl.load(anc_ptr + on[:, None] * N + on[None, :], mask=mn[:, None] & mn[None, :], other=0.0)
        G = tl.sum(anc * gl[None, :], 1)
        isanc = anc > 0.5
        D = tl.exp(tl.where(isanc, G[:, None] - G[None, :], float('-inf')))
        offd = on[:, None] != on[None, :]
        kb = k * beta[:, None]
        A = tl.where(isanc & offd, tl.dot(kb, tl.trans(k)) * D, 0.0)
        I = tl.where(offd, 0.0, 1.0)
        X = I - A
        P = tl.dot(A, A)
        X = tl.dot(X, I + P)
        P = tl.dot(P, P)
        X = tl.dot(X, I + P)
        P = tl.dot(P, P)
        X = tl.dot(X, I + P)
        eG = tl.exp(G)
        U = tl.dot(X, v * beta[:, None])
        Wk = tl.dot(X, kb * eG[:, None])
        S0 = tl.load(s0_ptr + h * s_s_h + ok_[:, None] * s_s_k + ov[None, :] * s_s_v)
        vn = U - tl.dot(Wk, S0)
        if SAVE:
            tl.store(vn_ptr + (h * N + on[:, None]) * DV + ov[None, :], vn, mask=mn[:, None])
            if tl.program_id(1) == 0:
                tl.store(kn_ptr + (h * N + on[:, None]) * DK + ok_[None, :], k, mask=mn[:, None])
                tl.store(gn_ptr + h * N + on, G, mask=mn)
        Pq = tl.dot(q, tl.trans(k)) * D
        o = tl.dot(q * eG[:, None], S0) + tl.dot(Pq, vn)
        tl.store(o_ptr + on[:, None] * s_o_n + h * s_o_h + ov[None, :], o, mask=mn[:, None])

def _pow2(n):
    return max(16, 1 << (int(n) - 1).bit_length())

def tree_conv_fused(conv_state, mixed0, cidx, w2, bias, BC=1024):
    N, C = mixed0.shape
    KC = cidx.shape[1]
    assert mixed0.stride(1) == 1 and w2.is_contiguous()
    y = torch.empty(N, C, device=mixed0.device, dtype=mixed0.dtype)
    ci = cidx if cidx.dtype == torch.int32 and cidx.is_contiguous() else cidx.to(torch.int32).contiguous()
    _tree_conv_kernel[N, triton.cdiv(C, BC)](conv_state, mixed0, ci, w2, bias if bias is not None else w2, y, C, conv_state.stride(0), conv_state.stride(1), mixed0.stride(0), mixed0.stride(1), y.stride(0), KC=KC, BC=BC, HAS_BIAS=bias is not None)
    return y

def tree_gdn_fused(y, aa0, bb0, A_log, dt_bias, S0, Anc, Hk, Hv, Dk, Dv, scale=None, BV=16, num_warps=4, aux=None, neg_exp_A=None):
    N = y.shape[0]
    if N > 32:
        return tree_gdn_split(y, aa0, bb0, A_log, dt_bias, S0, Anc, Hk, Hv, Dk, Dv, scale=scale, aux=aux, neg_exp_A=neg_exp_A)
    BN = _pow2(N)
    assert BN <= 64, 'Invalid n-gram input or configuration'
    assert y.stride(1) == 1 and Dv % BV == 0
    scale = Dk ** (-0.5) if scale is None else scale
    aa0 = aa0 if aa0.stride(1) == 1 else aa0.contiguous()
    bb0 = bb0 if bb0.stride(1) == 1 else bb0.contiguous()
    Anc = Anc if Anc.is_contiguous() and Anc.dtype == torch.float32 else Anc.float().contiguous()
    o = torch.empty(N, Hv, Dv, device=y.device, dtype=torch.float32)
    _tree_gdn_kernel[Hv, Dv // BV](y, aa0, bb0, A_log, dt_bias, S0, Anc, o, aux[0] if aux is not None else o, aux[1] if aux is not None else o, aux[2] if aux is not None else o, neg_exp_A if neg_exp_A is not None else A_log, N, y.stride(0), aa0.stride(0), bb0.stride(0), S0.stride(0), S0.stride(1), S0.stride(2), o.stride(0), o.stride(1), Hk * Dk, float(scale), R=Hv // Hk, DK=Dk, DV=Dv, BN=BN, BV=BV, SAVE=aux is not None, CACHED=neg_exp_A is not None, num_warps=num_warps, num_stages=1)
    return o
if _HAS_TRITON:

    @triton.jit
    def _closed_commit(S, K, V, G, Src, Num, H: tl.constexpr, N: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr=32, BV: tl.constexpr=32, BT: tl.constexpr=16):
        lh = tl.program_id(0)
        ik = tl.program_id(1) * BK + tl.arange(0, BK)
        iv = tl.program_id(2) * BV + tl.arange(0, BV)
        t = tl.arange(0, BT)
        n = tl.load(Num)
        u = tl.load(Src + n)
        ids = tl.load(Src + t, mask=t <= n, other=0)
        gn = tl.load(G + lh * N + u)
        gt = tl.load(G + lh * N + ids, mask=t <= n, other=0.0)
        w = tl.exp(tl.where(t <= n, gn - gt, float('-inf')))
        k = tl.load(K + (lh * N + ids[None, :]) * DK + ik[:, None], mask=(ik[:, None] < DK) & (t[None, :] <= n), other=0.0)
        v = tl.load(V + (lh * N + ids[:, None]) * DV + iv[None, :], mask=(t[:, None] <= n) & (iv[None, :] < DV), other=0.0)
        old = tl.load(S + lh * DK * DV + ik[:, None] * DV + iv[None, :], mask=(ik[:, None] < DK) & (iv[None, :] < DV), other=0.0)
        new = old * tl.exp(gn) + tl.dot(k * w[None, :], v, input_precision='tf32x3')
        tl.store(S + lh * DK * DV + ik[:, None] * DV + iv[None, :], new, mask=(ik[:, None] < DK) & (iv[None, :] < DV))

def closed_commit(states, keys, values, gains, src, n):
    L, H, DK, DV = states.shape
    N = keys.shape[2]
    _closed_commit[L * H, triton.cdiv(DK, 32), triton.cdiv(DV, 32)](states, keys, values, gains, src, n, H, N, DK, DV, BT=16, num_warps=4)
if _HAS_TRITON:

    @triton.jit
    def _tree_prepare(y_ptr, aa_ptr, bb_ptr, alog_ptr, dtb_ptr, anc_ptr, qn_ptr, kn_ptr, gn_ptr, beta_ptr, x_ptr, neg_ptr, N, s_y_n, s_a_n, s_b_n, KD, scale, R: tl.constexpr, DK: tl.constexpr, BN: tl.constexpr, CACHED: tl.constexpr):
        h = tl.program_id(0)
        hk = h // R
        on = tl.arange(0, BN)
        mn = on < N
        od = tl.arange(0, DK)
        q = tl.load(y_ptr + on[:, None] * s_y_n + (hk * DK + od)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        k = tl.load(y_ptr + on[:, None] * s_y_n + (KD + hk * DK + od)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, 1) + 1e-06)[:, None] * scale
        k = k * tl.rsqrt(tl.sum(k * k, 1) + 1e-06)[:, None]
        a = tl.load(aa_ptr + on * s_a_n + h, mask=mn, other=0.0).to(tl.float32) + tl.load(dtb_ptr + h).to(tl.float32)
        sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(tl.minimum(a, 20.0))))
        dec = tl.load(neg_ptr + h).to(tl.float32) if CACHED else -tl.exp(tl.load(alog_ptr + h).to(tl.float32))
        gl = tl.where(mn, dec * sp, 0.0)
        b = tl.load(bb_ptr + on * s_b_n + h, mask=mn, other=0.0).to(tl.float32)
        beta = tl.where(mn, 1.0 / (1.0 + tl.exp(-b)), 0.0)
        anc = tl.load(anc_ptr + on[:, None] * N + on[None, :], mask=mn[:, None] & mn[None, :], other=0.0)
        G = tl.sum(anc * gl[None, :], 1)
        isanc = anc > 0.5
        D = tl.exp(tl.where(isanc, G[:, None] - G[None, :], float('-inf')))
        offd = on[:, None] != on[None, :]
        I = tl.where(offd, 0.0, 1.0)
        A = tl.where(isanc & offd, tl.dot(k * beta[:, None], tl.trans(k)) * D, 0.0)
        X = A
        tl.store(qn_ptr + (h * N + on[:, None]) * DK + od[None, :], q, mask=mn[:, None])
        tl.store(kn_ptr + (h * N + on[:, None]) * DK + od[None, :], k, mask=mn[:, None])
        tl.store(gn_ptr + h * N + on, G, mask=mn)
        tl.store(beta_ptr + h * N + on, beta, mask=mn)
        tl.store(x_ptr + (h * N + on[:, None]) * N + on[None, :], X, mask=mn[:, None] & mn[None, :])

    @triton.jit
    def _tree_finish(y_ptr, s0_ptr, anc_ptr, qn_ptr, kn_ptr, gn_ptr, beta_ptr, x_ptr, vn_ptr, o_ptr, slot_ptr, N, s_slot, s_y_n, s_s_h, s_s_k, s_s_v, s_o_n, s_o_h, KD, DK: tl.constexpr, DV: tl.constexpr, BN: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr, HAS_SLOT: tl.constexpr):
        h = tl.program_id(0)
        on = tl.arange(0, BN)
        mn = on < N
        od = tl.arange(0, DK)
        ov = tl.program_id(1) * BV + tl.arange(0, BV)
        q = tl.load(qn_ptr + (h * N + on[:, None]) * DK + od[None, :], mask=mn[:, None], other=0.0)
        k = tl.load(kn_ptr + (h * N + on[:, None]) * DK + od[None, :], mask=mn[:, None], other=0.0)
        G = tl.load(gn_ptr + h * N + on, mask=mn, other=0.0)
        beta = tl.load(beta_ptr + h * N + on, mask=mn, other=0.0)
        X = tl.load(x_ptr + (h * N + on[:, None]) * N + on[None, :], mask=mn[:, None] & mn[None, :], other=0.0)
        anc = tl.load(anc_ptr + on[:, None] * N + on[None, :], mask=mn[:, None] & mn[None, :], other=0.0)
        D = tl.exp(tl.where(anc > 0.5, G[:, None] - G[None, :], float('-inf')))
        eG = tl.exp(G)
        v = tl.load(y_ptr + on[:, None] * s_y_n + (2 * KD + h * DV + ov)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        U = tl.dot(X, v * beta[:, None])
        Wk = tl.dot(X, k * beta[:, None] * eG[:, None])
        slot = tl.maximum(tl.load(slot_ptr), 0) if HAS_SLOT else 0
        S0 = tl.load(s0_ptr + slot * s_slot + h * s_s_h + od[:, None] * s_s_k + ov[None, :] * s_s_v)
        vn = U - tl.dot(Wk, S0)
        if SAVE:
            tl.store(vn_ptr + (h * N + on[:, None]) * DV + ov[None, :], vn, mask=mn[:, None])
        Pq = tl.dot(q, tl.trans(k)) * D
        o = tl.dot(q * eG[:, None], S0) + tl.dot(Pq, vn)
        tl.store(o_ptr + on[:, None] * s_o_n + h * s_o_h + ov[None, :], o, mask=mn[:, None])

def tree_gdn_split(y, aa0, bb0, A_log, dt_bias, S0, Anc, Hk, Hv, Dk, Dv, scale=None, BV=64, num_warps=8, aux=None, neg_exp_A=None, state_indices=None, state_slot_stride=0):
    N = y.shape[0]
    BN = _pow2(N)
    scale = Dk ** (-0.5) if scale is None else scale
    assert BN <= 64 and Dv % BV == 0
    q = torch.empty(Hv, N, Dk, device=y.device, dtype=torch.float32)
    k = aux[0] if aux is not None else torch.empty_like(q)
    G = aux[2] if aux is not None else torch.empty(Hv, N, device=y.device, dtype=torch.float32)
    beta = torch.empty_like(G)
    X = torch.empty(Hv, N, N, device=y.device, dtype=torch.float32)
    o = torch.empty(N, Hv, Dv, device=y.device, dtype=y.dtype)
    _tree_prepare[Hv,](y, aa0, bb0, A_log, dt_bias, Anc, q, k, G, beta, X, neg_exp_A if neg_exp_A is not None else A_log, N, y.stride(0), aa0.stride(0), bb0.stride(0), Hk * Dk, float(scale), R=Hv // Hk, DK=Dk, BN=BN, CACHED=neg_exp_A is not None, num_warps=num_warps, num_stages=1)
    from fc1_block_inverse import invert64, invert32, invert16
    {16: invert16, 32: invert32, 64: invert64}[N][Hv,](X, N, num_warps=4, num_stages=1)
    _tree_finish[Hv, Dv // BV](y, S0, Anc, q, k, G, beta, X, aux[1] if aux is not None else o, o, state_indices if state_indices is not None else G, N, state_slot_stride, y.stride(0), S0.stride(0), S0.stride(1), S0.stride(2), o.stride(0), o.stride(1), Hk * Dk, DK=Dk, DV=Dv, BN=BN, BV=BV, SAVE=aux is not None, HAS_SLOT=state_indices is not None, num_warps=num_warps, num_stages=1)
    return o
