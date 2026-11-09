# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
"""Our-only, single-request tree plan using known host lengths, no GPU readback."""
import inspect
import torch
HOST_KV_LEN = None

def install():
    import flashinfer
    from sglang.srt.layers.attention.flashinfer_backend import fast_prefill_plan
    cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
    original = cls.plan
    names = list(inspect.signature(original).parameters)[1:]
    fast_names = set(inspect.signature(fast_prefill_plan).parameters) - {'self'}

    def plan(self, *args, **kwargs):
        values = dict(zip(names, args))
        values.update(kwargs)
        mask = values.get('custom_mask')
        if HOST_KV_LEN is None or mask is None or (not self.is_cuda_graph_enabled) or (getattr(self, '_backend', None) != 'fa2') or (getattr(self, '_cached_module', None) is None):
            return original(self, *args, **kwargs)
        qo = values['qo_indptr']
        assert qo.numel() == 2 and values['page_size'] == 1
        n = mask.numel() // HOST_KV_LEN
        assert n in (16, 32, 64) and n * HOST_KV_LEN == mask.numel()
        packed = flashinfer.packbits(mask.flatten(), bitorder='little')
        self._custom_mask_buf[:packed.numel()].copy_(packed, non_blocking=True)
        self._mask_indptr_buf.copy_(torch.tensor([0, packed.numel()], dtype=self._mask_indptr_buf.dtype, device=qo.device), non_blocking=True)
        host_q = torch.tensor([0, n], dtype=torch.int32)
        host_k = torch.tensor([0, HOST_KV_LEN], dtype=torch.int32)
        vals = {k: v for k, v in values.items() if k in fast_names}
        vals.update(qo_indptr_host=host_q, kv_indptr_host=host_k, kv_lens_host=host_k[1:], max_q_len=n, max_kv_len=HOST_KV_LEN)
        return fast_prefill_plan(self, **vals)
    cls.plan = plan
    cls.begin_forward = plan
