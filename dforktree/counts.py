# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import argparse, json, os, sys, time
import numpy as np

def merge(parts):
    if not parts:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64))
    k = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(k, return_inverse=True)
    return (u, np.bincount(inv, weights=c, minlength=len(u)).astype(np.int64))

class Counter3:

    def __init__(self, V, chunk_tokens):
        self.V, self.chunk = (V, chunk_tokens)
        self.buf2, self.buf3, self.nbuf = ([], [], 0)
        self.parts2, self.parts3 = ([], [])
        self.prev = np.zeros(V, np.int64)
        self.nxt = np.zeros(V, np.int64)
        self.N = 0
        self.n_tok = 0

    def add(self, s):
        s = np.asarray(s, np.int64)
        if len(s) < 2:
            return
        V = self.V
        self.n_tok += len(s)
        assert int(s.max()) < V and int(s.min()) >= 0, f'Invalid n-gram input or configuration{V}Invalid n-gram input or configuration'
        self.buf2.append(s[:-1] * V + s[1:])
        self.N += len(s) - 1
        if len(s) >= 3:
            self.buf3.append((s[:-2] * V + s[1:-1]) * V + s[2:])
        self.nbuf += len(s)
        if self.nbuf >= self.chunk:
            self.flush()

    def flush(self):
        if self.buf2:
            b2 = np.concatenate(self.buf2)
            self.prev += np.bincount(b2 // self.V, minlength=self.V)
            self.nxt += np.bincount(b2 % self.V, minlength=self.V)
            self.buf2 = [b2]
        for buf, parts in ((self.buf2, self.parts2), (self.buf3, self.parts3)):
            if buf:
                u, c = np.unique(np.concatenate(buf), return_counts=True)
                parts.append((u, c.astype(np.int64)))
                buf.clear()
        self.nbuf = 0
        if len(self.parts3) >= 8:
            self.parts2 = [merge(self.parts2)]
            self.parts3 = [merge(self.parts3)]

    def finish(self, min3):
        self.flush()
        k2, c2 = merge(self.parts2)
        k3, c3 = merge(self.parts3)
        keep = c3 >= min3
        return (k2, c2, k3[keep], c3[keep], int((~keep).sum()))

def context_sums(k3, c3, V):
    if len(k3) == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.float32))
    ctx = k3 // V
    u, inv = np.unique(ctx, return_inverse=True)
    return (u, np.bincount(inv, weights=c3, minlength=len(u)).astype(np.float32))
