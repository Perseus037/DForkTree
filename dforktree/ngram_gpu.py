# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import numpy as np
import torch

class NGramGPU:

    def __init__(self, path, dev, beta2=20.0, beta3=20.0, beta4=20.0, clip=10.0):
        z = np.load(path)
        f = set(z.files)
        T = lambda x, dt: torch.from_numpy(np.ascontiguousarray(np.asarray(x).astype(dt))).to(dev)
        self.V = int(z['V'])
        N = float(z['N'])
        self.k2, self.c2 = (T(z['keys'], np.int64), T(z['cab'], np.float32))
        self.prev = T(z['prev'], np.float32)
        self.pn = T((z['nxt'].astype(np.float64) + 1.0) / (N + self.V), np.float32)
        e = lambda dt: T(np.zeros(0), dt)
        self.k3 = T(z['keys3'], np.int64) if 'keys3' in f else e(np.int64)
        self.c3 = T(z['c3'], np.float32) if 'keys3' in f else e(np.float32)
        self.k3c = T(z['keys3c'], np.int64) if 'keys3c' in f else e(np.int64)
        self.n3c = T(z['n3c'], np.float32) if 'keys3c' in f else e(np.float32)
        self.has4 = 'keys4' in f
        self.k4 = T(z['keys4'], np.int64) if self.has4 else e(np.int64)
        self.c4 = T(z['c4'], np.float32) if self.has4 else e(np.float32)
        self.n4c = T(z['n4c'], np.float32) if self.has4 else T(np.zeros(max(len(self.k3), 1)), np.float32)
        self.beta2, self.beta3, self.beta4, self.clip = (beta2, beta3, beta4, clip)

    @staticmethod
    def _index(keys, q):
        if keys.shape[0] == 0:
            return torch.full_like(q, -1)
        idx = torch.searchsorted(keys, q.reshape(-1)).clamp(max=keys.shape[0] - 1).reshape(q.shape)
        return torch.where(keys[idx] == q, idx, torch.full_like(idx, -1))

    def _look(self, keys, vals, q):
        i = self._index(keys, q)
        return torch.where(i >= 0, vals[i.clamp(min=0)] if vals.shape[0] else torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(q, dtype=torch.float32))

    def p2(self, b, c):
        n = self._look(self.k2, self.c2, b * self.V + c)
        nb = torch.where(b >= 0, self.prev[b.clamp(min=0)], torch.zeros_like(n))
        return (n + self.beta2 * self.pn[c]) / (nb + self.beta2)

    def p3(self, a, b, c):
        V = self.V
        ok = (a >= 0) & (b >= 0)
        q = torch.where(ok, a * V + b, torch.full_like(a * V + b, -1))
        nab = self._look(self.k3c, self.n3c, q)
        q3 = torch.where(ok, (a * V + b) * V + c, torch.full_like((a * V + b) * V + c, -1))
        nabc = self._look(self.k3, self.c3, q3)
        return (nabc + self.beta3 * self.p2(b, c)) / (nab + self.beta3)

    def p4(self, z, a, b, c):
        V = self.V
        ok = (z >= 0) & (a >= 0) & (b >= 0)
        q = torch.where(ok, (z * V + a) * V + b, torch.full_like((z * V + a) * V + b, -1))
        i3 = self._index(self.k3, q)
        n4 = torch.where(i3 >= 0, self.n4c[i3.clamp(min=0)], torch.zeros_like(i3, dtype=torch.float32))
        q4 = torch.where(i3 >= 0, i3 * V + c, torch.full_like(i3 * V + c, -1))
        n4c_ = self._look(self.k4, self.c4, q4)
        return (n4c_ + self.beta4 * self.p3(a, b, c)) / (n4 + self.beta4)

    def pmi(self, order, z, a, b, c):
        if order == 2:
            p = self.p2(b, c)
        elif order == 3:
            p = self.p3(a, b, c)
        else:
            p = self.p4(z, a, b, c)
        return (torch.log(p) - torch.log(self.pn[c])).clamp(-self.clip, self.clip)
