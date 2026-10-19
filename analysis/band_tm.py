"""Row-to-row transfer matrix of percolation on a band of the square lattice (cylinder of
circumference W), in the connectivity representation; spectrum by momentum sector.

State of a row: bond — set partition of the W sites (connectivity through everything above);
site — occupancy plus a partition of the occupied sites.  One step = one row of squares:
vertical links to the new row, then horizontal links inside it.  The matrix is
column-stochastic (lambda = 1 at k = 0) and commutes with the shift along the row, so
a period-2 boundary perturbation (relative registry of two strips) decays through h rows as
lambda_pi^h, lambda_pi = largest eigenvalue in the k = pi sector.

usage: python analysis/band_tm.py site|bond W p [nev]
"""
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs


def canon(s):
    m, out = {}, []
    for x in s:
        if x < 0:
            out.append(-1)
        else:
            if x not in m:
                m[x] = len(m)
            out.append(m[x])
    return tuple(out)


def states(W, site):
    """all canonical states"""
    res = []

    def rec(prefix, nxt):
        if len(prefix) == W:
            res.append(tuple(prefix))
            return
        if site:
            rec(prefix + [-1], nxt)
        for lab in range(nxt + 1):
            rec(prefix + [lab], max(nxt, lab + 1))
    rec([], 0)
    return res


def build(W, site, p):
    S = states(W, site)
    idx = {s: i for i, s in enumerate(S)}
    n = len(S)

    def op(fn):
        rows, cols, vals = [], [], []
        for j, s in enumerate(S):
            for t, w in fn(list(s)):
                if w:
                    rows.append(idx[canon(t)]); cols.append(j); vals.append(w)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    ops = []
    for i in range(W):  # vertical links / occupation of the new row
        def V(s, i=i):
            big = max(s) + 1
            if site:
                occ = list(s)
                if occ[i] < 0:
                    occ[i] = big
                emp = list(s); emp[i] = -1
                return [(occ, p), (emp, 1 - p)]
            cut = list(s); cut[i] = big
            return [(s, p), (cut, 1 - p)]
        ops.append(op(V))
    for i in range(W):  # horizontal links in the new row
        j = (i + 1) % W

        def H(s, i=i, j=j):
            a, b = s[i], s[j]
            if site:
                if a >= 0 and b >= 0 and a != b:
                    return [([a if x == b else x for x in s], 1.0)]
                return [(s, 1.0)]
            if a == b:
                return [(s, 1.0)]
            return [([a if x == b else x for x in s], p), (s, 1 - p)]
        ops.append(op(H))
    shift = np.array([idx[canon([s[(k - 1) % W] for k in range(W)])] for s in S])
    return S, ops, shift


def sector_eigs(W, site, p, k, nev=6):
    S, ops, shift = build(W, site, p)
    n = len(S)

    def T(v):
        for M in ops:
            v = M @ v
        return v

    def P(v):  # projector onto momentum k
        acc = np.zeros(n, dtype=complex)
        cur = v.astype(complex)
        for j in range(W):
            acc += np.exp(-1j * k * j) * cur
            nxt = np.zeros(n, dtype=complex)
            nxt[shift] = cur  # apply shift once
            cur = nxt
        return acc / W

    A = LinearOperator((n, n), matvec=lambda v: T(P(v)), dtype=complex)
    vals = eigs(A, k=min(nev, n - 2), which='LM', return_eigenvectors=False, maxiter=5000, tol=1e-10)
    return n, sorted(vals, key=lambda z: -abs(z))


if __name__ == '__main__':
    mode, W, p = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    nev = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    site = mode == 'site'
    for k, name in ((0.0, 'k=0'), (np.pi, 'k=pi')):
        n, v = sector_eigs(W, site, p, k, nev)
        print(f'{mode} W={W} p={p} states={n} {name}: ' + '  '.join(f'{z.real:+.5f}{z.imag:+.5f}i' for z in v), flush=True)
