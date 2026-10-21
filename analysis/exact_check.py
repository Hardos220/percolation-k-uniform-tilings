"""Exact test of the simulation code on small tori (fourth review, item 4.3).

On a torus with K objects (sites or bonds), the first m objects of a uniformly random order form a uniform
m-subset, so the Monte Carlo distributions of n_e (first wrapping) and n_x (first cross configuration) must equal
    P(n_e <= m) = 1 - A_m / C(K, m),   P(n_x <= m) = C_m / C(K, m),
where A_m and C_m count the m-subsets without a wrapping cluster and with a cross cluster, by full enumeration
(nz/exact.c). For every case we compare the full distributions by a chi^2 test, and the root of
P_cross(p) = P_nowrap(p) (the exact finite-torus estimate) with the Monte Carlo root of R_e + R_x = 1.
For the smallest cases the enumeration itself is checked by an independent classifier written here (breadth-first
search and the rank of the winding vectors of every cluster, no union-find).

usage: python analysis/exact_check.py [SAMPLES_PER_CHUNK] [CHUNKS]   (workers $RMQ_CPU) -> results/exact_check.json
"""
import concurrent.futures as cf
import hashlib
import json
import math
import os
import subprocess
import sys
from math import comb

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import p_cross  # noqa: E402
from wrap import Hist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXACT = os.path.join(HERE, '..', 'nz', 'exact.exe')
NZ = os.path.join(HERE, '..', 'nz', 'nz.exe')
CASES = [('01__eu_4n_1', 'site', 4, 4), ('01__eu_4n_1', 'site', 5, 5), ('01__eu_4n_1', 'bond', 3, 3),
         ('01__eu_6a_1', 'site', 5, 5), ('01__eu_6a_1', 'bond', 3, 3), ('01__eu_3d_1', 'site', 3, 4),
         ('01__eu_3d_1', 'bond', 3, 3), ('01__eu_4f_1', 'site', 3, 3), ('01__eu_4f_1', 'bond', 2, 2),
         ('01__eu_5c_1', 'site', 4, 3), ('02__2__eu_4f_4k_1', 'site', 2, 2), ('03__3__eu_4f_4k_4r_1', 'site', 2, 2)]
OUT = 'results/exact_check'


def cell(tag):
    lines = open(f'data/cells_txt/{tag}.txt').read().split('\n')
    nv, ne = map(int, lines[0].split())
    return nv, [tuple(map(int, l.split())) for l in lines[1:1 + ne]]


def torus(tag, Lx, Ly):
    nv, edges = cell(tag)
    E = []
    for b in range(Ly):
        for a in range(Lx):
            for u, w, da, db in edges:
                E.append(((b * Lx + a) * nv + u, (((b + db) % Ly) * Lx + (a + da) % Lx) * nv + w, da, db))
    return Lx * Ly * nv, E


def rank2(vs):
    vs = [v for v in vs if v != (0, 0)]
    if not vs:
        return 0
    x0 = vs[0]
    return 2 if any(x0[0] * y[1] - x0[1] * y[0] != 0 for y in vs[1:]) else 1


def independent_counts(tag, mode, Lx, Ly):
    """A_m, C_m by breadth-first search: winding vectors of each cluster from its non-tree edges."""
    N, E = torus(tag, Lx, Ly)
    K = len(E) if mode == 'bond' else N
    A, C = [0] * (K + 1), [0] * (K + 1)
    for mask in range(1 << K):
        if mode == 'bond':
            act = [E[i] for i in range(K) if mask >> i & 1]
        else:
            act = [e for e in E if (mask >> e[0] & 1) and (mask >> e[1] & 1)]
        adj = {}
        for s, t, dx, dy in act:
            adj.setdefault(s, []).append((t, dx, dy))
            adj.setdefault(t, []).append((s, -dx, -dy))
        pos, wrap, cross = {}, False, False
        for s0 in adj:
            if s0 in pos:
                continue
            pos[s0] = (0, 0)
            stack, wind = [s0], []
            while stack:
                v = stack.pop()
                for t, dx, dy in adj[v]:
                    q = (pos[v][0] + dx, pos[v][1] + dy)
                    if t not in pos:
                        pos[t] = q
                        stack.append(t)
                    else:
                        wind.append((q[0] - pos[t][0], q[1] - pos[t][1]))
            r = rank2(wind)
            wrap |= r >= 1
            cross |= r == 2
        m = bin(mask).count('1')
        if not wrap:
            A[m] += 1
        if cross:
            C[m] += 1
    return A, C


def exact_counts(tag, mode, Lx, Ly):
    r = subprocess.run([EXACT, f'data/cells_txt/{tag}.txt', mode, str(Lx), str(Ly)], capture_output=True, text=True, check=True)
    A, C = [], []
    for line in r.stdout.splitlines():
        if line.startswith('#'):
            continue
        m, a, c, w = map(int, line.split())
        A.append(a)
        C.append(c)
    return A, C


def mc_chunk(args):
    tag, mode, Lx, Ly, M, c = args
    name = f'{tag}__{mode}__{Lx}x{Ly}__exact{c:02d}'
    out = f'{OUT}/{name}.txt'
    if not os.path.exists(out):
        seed = int(hashlib.sha1(name.encode()).hexdigest()[:12], 16)
        subprocess.run([NZ, f'data/cells_txt/{tag}.txt', mode, str(Lx), str(Ly), str(M), str(seed), out + '.part'], check=True)
        os.replace(out + '.part', out)
    return out


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
    chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    os.makedirs(OUT, exist_ok=True)
    nw = int(os.environ.get('RMQ_CPU', '1'))
    jobs = [(t, m, Lx, Ly, M, c) for (t, m, Lx, Ly) in CASES for c in range(chunks)]
    with cf.ThreadPoolExecutor(nw) as ex:
        exact = {cs: ex.submit(exact_counts, *cs) for cs in CASES}
        files = list(ex.map(mc_chunk, jobs))
        exact = {cs: f.result() for cs, f in exact.items()}
    res = []
    for cs in CASES:
        tag, mode, Lx, Ly = cs
        A, C = exact[cs]
        K = len(A) - 1
        rec = dict(tag=tag, mode=mode, torus=f'{Lx}x{Ly}', K=K)
        if K <= 20:
            Ai, Ci = independent_counts(*cs)
            rec['enumeration_agrees'] = (Ai == A and Ci == C)
        H = Hist([f for f in files if f'__{mode}__{Lx}x{Ly}__exact' in f and f'/{tag}__' in f.replace(os.sep, '/')])
        # exact distributions of n_e (column 'e' = first wrap in any direction) and n_x
        Fe = np.array([1 - A[m] / comb(K, m) for m in range(K + 1)])
        Fx = np.array([C[m] / comb(K, m) for m in range(K + 1)])
        for X, F, col in (('e', Fe, 2), ('x', Fx, 4)):
            pe = np.diff(np.concatenate([[0.0], F]))
            obs = np.zeros(K + 1)
            for n, cnt in zip(H.n, H.c[:, col]):
                obs[n] += cnt
            exp_ = pe * H.M
            ok = exp_ > 5
            c2 = float(np.sum((obs[ok] - exp_[ok]) ** 2 / exp_[ok]))
            dof = int(ok.sum()) - 1
            rec[f'chi2_{X}'] = c2
            rec[f'dof_{X}'] = dof
            rec[f'p_{X}'] = float(chi2dist.sf(c2, dof))
        # exact finite-torus root of P_cross - P_nowrap and the Monte Carlo root
        f = lambda p: sum((C[m] - A[m]) * p ** m * (1 - p) ** (K - m) for m in range(K + 1))
        rec['p_exact'] = brentq(f, 1e-6, 1 - 1e-6, xtol=1e-15)
        pmc, smc = p_cross(H)
        rec.update(samples=H.M, p_mc=pmc, s_mc=smc, z=(pmc - rec['p_exact']) / smc)
        res.append(rec)
        print(f"{tag:24s} {mode} {Lx}x{Ly} K={K:2d} enum_ok={rec.get('enumeration_agrees', '-')!s:5s} "
              f"chi2_e={rec['chi2_e']:.1f}/{rec['dof_e']} (p={rec['p_e']:.2f}) chi2_x={rec['chi2_x']:.1f}/{rec['dof_x']} (p={rec['p_x']:.2f}) "
              f"p_exact={rec['p_exact']:.8f} p_mc={pmc:.8f}({smc*1e8:.0f}) z={rec['z']:+.2f}", flush=True)
    json.dump(res, open('results/exact_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
