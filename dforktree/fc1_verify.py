# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import torch
from dataclasses import dataclass
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
_CHAIN_LINKS = {}

@dataclass
class ChainTreeVerifyInput(DFlashVerifyInput):
    topk: int = 2
    retrieve_next_token: torch.Tensor | None = None
    retrieve_next_sibling: torch.Tensor | None = None
    retrieve_parent_token: torch.Tensor | None = None

    def __post_init__(self):
        super().__post_init__()
        n = int(self.draft_token_num)
        bs = self.draft_token.numel() // n
        key = (str(self.draft_token.device), n, bs)
        if key not in _CHAIN_LINKS:
            ids = torch.arange(n, device=self.draft_token.device, dtype=torch.int32)
            nxt = torch.where(ids + 1 < n, ids + 1, torch.full_like(ids, -1))[None].expand(bs, n).contiguous()
            sib = torch.full((bs, n), -1, device=ids.device, dtype=torch.int32)
            parent = (ids - 1)[None].expand(bs, n).contiguous()
            _CHAIN_LINKS[key] = (nxt, sib, parent)
        self.retrieve_next_token, self.retrieve_next_sibling, self.retrieve_parent_token = _CHAIN_LINKS[key]
