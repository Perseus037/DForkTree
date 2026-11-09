# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import torch
import torch.nn.functional as F
NEG = float('-inf')

def beam_tree(c0, logc, N, lmax, chain_first=True, path_fn=None, S=None):
    S = logc.shape[0] + 1 if logc is not None else S
    M = c0.shape[0]
    W = max(N - 1, M)
    assert W >= M and lmax >= 1
    dev = c0.device
    BIG = 10000.0 if chain_first else 0.0
    arM = torch.arange(M, device=dev)
    top0 = arM == c0.argmax()
    pad = W - M
    sc = torch.cat([c0.float(), torch.full((pad,), NEG, device=dev)])
    fl = torch.cat([top0, torch.zeros(pad, dtype=torch.bool, device=dev)])
    lv = torch.cat([(~top0).long(), torch.full((pad,), lmax + 1, dtype=torch.long, device=dev)])
    cand = torch.cat([arM, torch.zeros(pad, dtype=torch.long, device=dev)])
    par = torch.full((W,), -1, dtype=torch.long, device=dev)
    al = torch.cat([arM == 0, torch.zeros(pad, dtype=torch.bool, device=dev)])
    pc = torch.zeros(W, dtype=torch.long, device=dev)
    ppc = torch.zeros(W, dtype=torch.long, device=dev)
    key = torch.where((lv <= lmax) & (sc > NEG), sc + BIG * fl.float(), torch.full_like(sc, NEG))
    keys, pars, cands, fls = ([key], [par], [cand], [fl])
    for s in range(1, S):
        lc = logc[s - 1].float()[cand] if path_fn is None else path_fn(s, cand, pc, ppc, al).float()
        is_top = arM[None, :] == lc.argmax(-1)[:, None]
        sc_c = sc[:, None] + lc
        fl_c = fl[:, None] & is_top
        lv_c = lv[:, None] + (~is_top).long()
        ok = (lv_c <= lmax) & (key[:, None] > NEG)
        key_c = torch.where(ok, sc_c + BIG * fl_c.float(), torch.full_like(sc_c, NEG))
        key, idx = key_c.reshape(-1).topk(W)
        p_, c_ = (idx // M, idx % M)
        sc, fl, lv = (sc_c.reshape(-1)[idx], fl_c.reshape(-1)[idx], lv_c.reshape(-1)[idx])
        pc, ppc, al = (cand[p_], pc[p_], al[p_] & (c_ == 0))
        par, cand = (p_, c_)
        keys.append(key)
        pars.append(par)
        cands.append(cand)
        fls.append(fl)
    K, Pb, Cb, Fb = (torch.stack(keys), torch.stack(pars), torch.stack(cands), torch.stack(fls))
    _, gi = K.reshape(-1).topk(N - 1)
    ss, bi = (gi // W, gi % W)
    order = torch.argsort(ss * W + torch.arange(N - 1, device=dev))
    ss, bi = (ss[order], bi[order])
    inv = torch.zeros(S, W, dtype=torch.long, device=dev)
    inv[ss, bi] = torch.arange(1, N, device=dev)
    pb = Pb[ss, bi]
    parent = torch.where(ss == 0, torch.zeros_like(ss), inv[(ss - 1).clamp(min=0), pb.clamp(min=0)]) - 1
    return (ss, Cb[ss, bi], parent, Fb[ss, bi])

def ngram_path_fn(ng, order, ids, lp, rest, pre1, root, lam_a, lam_p):
    S, M = ids.shape
    E = torch.cat([pre1.reshape(1, 1).expand(1, M), root.reshape(1, 1).expand(1, M), ids], 0)
    ch = E[3:S + 2]
    if order == 2:
        T = ng.pmi(2, None, None, E[2:S + 1][:, :, None], ch[:, None, :])
    elif order == 3:
        T = ng.pmi(3, None, E[1:S][:, :, None, None], E[2:S + 1][:, None, :, None], ch[:, None, None, :])
    else:
        T = ng.pmi(4, E[0:S - 1][:, :, None, None, None], E[1:S][:, None, :, None, None], E[2:S + 1][:, None, None, :, None], ch[:, None, None, None, :])
    T = T.float()
    return fn

def tree_struct(parent_full, Kc):
    N = parent_full.shape[0]
    dev = parent_full.device
    ar = torch.arange(N, device=dev)
    P = torch.zeros(N, N, device=dev)
    P[ar[1:], parent_full[1:].clamp(min=0)] = torch.ones_like(ar[1:], dtype=P.dtype)
    I = torch.eye(N, device=dev)
    Anc, Pk = (I + P, P @ P)
    for _ in range(4):
        Anc = Anc @ (I + Pk)
        Pk = Pk @ Pk
    Anc = (Anc > 0.5).float()
    depth = Anc.sum(1).long() - 1
    cols, Pr = ([], I)
    for r in range(Kc):
        if r > 0:
            Pr = Pr @ P
        has = Pr.sum(1) > 0.5
        a_r = Pr.argmax(1)
        cols.append(torch.where(r <= depth, Kc - 1 + a_r, Kc - 1 - (r - depth)))
    cidx = torch.stack(cols[::-1], 1)
    return (Anc, depth, cidx)
