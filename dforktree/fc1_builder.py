# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import os
import torch
import tree_ops as TO
from ngram_gpu import NGramGPU

class Builder:

    def __init__(self, logits, ngram=True, compile_build=False):
        self.n = int(os.environ.get('FC1_TREE_N', '32'))
        self.slots = logits.shape[0]
        self.chain_first = os.environ.get('FC1_CHAIN_FIRST', '0') == '1'
        self.logits = torch.zeros_like(logits)
        self.root = torch.zeros(1, device=logits.device, dtype=torch.long)
        self.pre1 = torch.zeros_like(self.root)
        self.ng = NGramGPU(os.environ.get('FC1_NGRAM_PATH', 'data/ngram_counts.npz'), logits.device, beta3=60.0) if ngram else None
        if compile_build:
            self.run = torch.compile(self.run, fullgraph=True, mode='default')
        self.run()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.run()

    def run(self):
        lp, ids = torch.log_softmax(self.logits.float(), -1).topk(16, -1)
        rest = torch.log((1 - lp.exp().sum(-1)).clamp(1e-06, 1.0))
        c0 = lp[0] - torch.logsumexp(torch.cat((lp[0], rest[:1])), 0)
        if self.ng is not None:
            fn = TO.ngram_path_fn(self.ng, 3, ids, lp, rest, self.pre1, self.root, 0.3, 0.6)
            logc = None
        else:
            fn = None
            base = lp[1:, None, :].expand(self.slots - 1, 16, 16)
            logc = base - torch.logsumexp(torch.cat((base, rest[1:, None, None].expand(self.slots - 1, 16, 1)), dim=-1), -1, keepdim=True)
        ss, cand, parent, _ = TO.beam_tree(c0, logc, self.n, 99, self.chain_first, fn, S=self.slots)
        self.parent = torch.cat((torch.full_like(self.root, -1), parent + 1))
        self.anc, self.depth, self.cidx = TO.tree_struct(self.parent, 4)
        self.tokens = torch.cat((self.root, ids[ss, cand]))
        u = torch.arange(self.n, device=ids.device)
        children = (self.parent[None, :] == u[:, None]) & (u[None, :] > 0)
        self.nxt = torch.where(children, u[None, :], self.n).min(-1).values
        sibling = (self.parent[:, None] == self.parent[None, :]) & (u[None, :] > u[:, None])
        self.sib = torch.where(sibling, u[None, :], self.n).min(-1).values
        self.nxt = torch.where(self.nxt < self.n, self.nxt, -1)
        self.sib = torch.where(self.sib < self.n, self.sib, -1)

    def __call__(self, logits, root, pre1):
        self.logits.copy_(logits)
        self.root.copy_(root)
        self.pre1.copy_(pre1)
        self.graph.replay()
        return (self.tokens[None], self.depth, self.parent, self.anc, self.nxt, self.sib)
