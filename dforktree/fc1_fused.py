# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import os
NN = int(os.environ.get('FC1_TREE_N', '64'))
BS = int(os.environ.get('FC1_DRAFT_BLOCK', '24'))
'Our-only GDN adapter. Native baselines never import this module.'
import torch
import triton
import triton.language as tl
import tree_kernels as TK

class Context:

    def __init__(self):
        self.anc = torch.eye(NN, device='cuda', dtype=torch.float32)
        self.keys = torch.empty(24, 32, NN, 128, device='cuda')
        self.values = torch.empty_like(self.keys)
        self.gains = torch.empty(24, 32, NN, device='cuda')
        self.layers = {}
        self.inter_indices = None
CTX = None
NATIVE_REFERENCE = False
COLLECT_LAYER = False
LAYER_STATS = []

@triton.jit
def snapshots(S, K, V, G, A, Path, Ends, Slot, InterSlot, Out, NN: tl.constexpr, BS: tl.constexpr, SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr, SS4: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr, OS2: tl.constexpr, OS3: tl.constexpr, OS4: tl.constexpr, OS5: tl.constexpr):
    lh = tl.program_id(0)
    l = lh // 32
    h = lh % 32
    ik = tl.program_id(1) * 32 + tl.arange(0, 32)
    q = tl.program_id(2)
    iv = q % 4 * 32 + tl.arange(0, 32)
    which = q // 4
    u = tl.load(Ends + which)
    slot = tl.load(Slot)
    dst = tl.load(InterSlot)
    t = tl.arange(0, 32)
    ids = tl.load(Path + t, mask=t < BS, other=-1)
    valid = (u >= 0) & (ids >= 0)
    on = tl.load(A + tl.maximum(u, 0) * NN + tl.maximum(ids, 0), mask=valid, other=0) > 0
    gn = tl.load(G + lh * NN + tl.maximum(u, 0))
    gt = tl.load(G + lh * NN + tl.maximum(ids, 0), mask=on, other=0.0)
    weights = tl.exp(tl.where(on, gn - gt, float('-inf')))
    kk = tl.load(K + (lh * NN + tl.maximum(ids[None, :], 0)) * 128 + ik[:, None], mask=on[None, :], other=0.0)
    vv = tl.load(V + (lh * NN + tl.maximum(ids[:, None], 0)) * 128 + iv[None, :], mask=on[:, None], other=0.0)
    old = tl.load(S + l * SS0 + slot * SS1 + h * SS2 + iv[None, :] * SS3 + ik[:, None] * SS4)
    new = old * tl.exp(gn) + tl.dot(kk * weights[None, :], vv, input_precision='tf32x3')
    tl.store(Out + l * OS0 + dst * OS1 + tl.maximum(u, 0) * OS2 + h * OS3 + iv[None, :] * OS4 + ik[:, None] * OS5, new, mask=u >= 0)

def install():
    global CTX
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.speculative import eagle_worker_common as common
    from sglang.srt.speculative.spec_utils import _verify_commit_step_indices
    original = TritonGDNKernel.target_verify
    commit = common.commit_mamba_states_after_verify

    def fused(self, A_log, dt_bias, q, k, v, a, b, **kw):
        global CTX
        if kw['cache_steps'] != NN:
            return original(self, A_log, dt_bias, q, k, v, a, b, **kw)
        if CTX is None:
            CTX = Context()
        key = A_log.data_ptr()
        if key not in CTX.layers:
            CTX.layers[key] = (len(CTX.layers), -A_log.float().exp())
        li, neg = CTX.layers[key]
        assert li < 24 and q.shape[0] == 1 and (q.shape[1] == NN)
        CTX.inter_indices = kw['intermediate_state_indices']
        pool = kw['ssm_states']
        state = pool[0].transpose(-1, -2)
        y = CTX.current_mixed
        out = TK.tree_gdn_split(y, a.reshape(NN, -1), b.reshape(NN, -1), A_log, dt_bias, state, CTX.anc, q.shape[2], v.shape[2], q.shape[3], v.shape[3], aux=(CTX.keys[li], CTX.values[li], CTX.gains[li]), neg_exp_A=neg, state_indices=kw['cache_indices'], state_slot_stride=pool.stride(0))
        return out.unsqueeze(0)

    def fused_commit(target_worker, batch, accept_lens, accept_index, draft_token_num, prepared_step_indices=None):
        if CTX is not None and draft_token_num == NN and accept_index.numel():
            last, track = _verify_commit_step_indices(batch=batch, accept_index=accept_index, accept_lens=accept_lens, draft_token_num=draft_token_num)
            if track is None:
                track = torch.full_like(last, -1)
            ends = torch.cat((last, track)).contiguous()
            pool = target_worker.model_runner.req_to_token_pool
            state = pool.get_speculative_mamba2_params_all_layers()
            slots = pool.get_mamba_indices(batch.req_pool_indices)
            snapshots[24 * 32, 4, 8](state.temporal, CTX.keys, CTX.values, CTX.gains, CTX.anc, accept_index, ends, slots, CTX.inter_indices, state.intermediate_ssm, NN, accept_index.shape[1], *state.temporal.stride(), *state.intermediate_ssm.stride(), num_warps=4)
        return commit(target_worker, batch, accept_lens, accept_index, draft_token_num, prepared_step_indices)
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    routing = GDNAttnBackend._target_verify_supports_strided_qkv

    def direct_routing(self, **kwargs):
        if not NATIVE_REFERENCE and kwargs['draft_token_num'] == NN:
            return True
        return routing(self, **kwargs)
    GDNAttnBackend._target_verify_supports_strided_qkv = direct_routing
    TritonGDNKernel.target_verify = fused
    common.commit_mamba_states_after_verify = fused_commit

