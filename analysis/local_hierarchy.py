"""Hierarchy of local descriptors for the thresholds (third review, item 4.6).

Colors of darts (directed edges) by refinement on the embedded periodic graph: for a dart d leaving vertex u,
c_0(d) = sequence of face sizes around u, starting at d (counterclockwise); c_r(d) = (c_{r-1}(d), sequence over the
darts d_m leaving u, starting at d, of c_{r-1}(reverse of d_m)), where the reverse dart points back to u, so the
neighbour's color keeps its orientation relative to the edge.  c_r(d) is determined by the tiling within graph
distance r of u.  The descriptor of a tiling at radius r is the vector of fractions of darts of each color, summed
over the tiling and its mirror image (so mirror images have equal descriptors); radius 0 is the vertex composition.

A. Extrapolation in k: linear rules p_c ~ sum_c x_c beta_c using all radii up to r (ridge regression; radius-0
   columns effectively unpenalized), fit on k <= 7 (2720 tilings), test on k = 8-10 (20676); the ridge parameter
   is chosen by fitting on k <= 6 and validating on k = 7.
B. Arrangement within a composition: the same, but with the descriptors and thresholds centered within each
   composition group (groups with at least two tilings), so that only the differences between tilings of equal
   composition are fitted; five-fold cross-validation over groups (one ridge parameter for all folds, chosen by
   fitting on folds 2-4 and validating on fold 1).
C. Slip pairs: smallest radius at which the descriptors of the two members differ.

usage: python analysis/local_hierarchy.py RMAX_FIT RMAX_SLIP MODE [RMAX_B]   -> results/local_hierarchy_MODE.json
(MODE = site or bond; part B uses radii 1..RMAX_B, default 2, with the ridge parameter chosen once on fold 0)
"""
import collections
import csv
import json
import math
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr


def darts_of(d, mirror=False):
    """For every vertex: list of (neighbour, index of the reverse dart at the neighbour), CCW, and face sizes."""
    s = -1.0 if mirror else 1.0
    P = np.array(d['pos'], float) * [s, 1.0]
    T1, T2 = np.array(d['T1']) * [s, 1.0], np.array(d['T2']) * [s, 1.0]
    nb = collections.defaultdict(list)
    for u, w, a, b in d['edges']:
        for x, y, aa, bb in ((u, w, a, b), (w, u, -a, -b)):
            vec = P[y] + aa * T1 + bb * T2 - P[x]
            nb[x].append((math.atan2(vec[1], vec[0]), y, aa, bb))
    for x in nb:
        nb[x].sort()
    rot, faces = [], []
    for v in range(d['nv']):
        lst = nb[v]
        rev = []
        for (_, y, aa, bb) in lst:
            j = [i for i, t in enumerate(nb[y]) if t[1] == v and t[2] == -aa and t[3] == -bb][0]
            rev.append((y, j))
        rot.append(rev)
        f = []
        for i in range(len(lst)):
            th = (lst[(i + 1) % len(lst)][0] - lst[i][0]) % (2 * math.pi)
            f.append(round(2 * math.pi / (math.pi - th)))
        faces.append(f)
    return rot, faces


def refine(rot, faces, rmax, intern):
    """Dart colors for r = 0..rmax; returns list over r of Counter(color id)."""
    nv = len(rot)
    col = []
    for v in range(nv):
        z = len(faces[v])
        col.append([intern[0].setdefault(tuple(faces[v][i:] + faces[v][:i]), len(intern[0])) for i in range(z)])
    H = [collections.Counter(c for cs in col for c in cs)]
    for r in range(1, rmax + 1):
        new = []
        for v in range(nv):
            z = len(rot[v])
            back = [col[y][j] for (y, j) in rot[v]]
            new.append([intern[r].setdefault((col[v][i], tuple(back[i:] + back[:i])), len(intern[r])) for i in range(z)])
        col = new
        H.append(collections.Counter(c for cs in col for c in cs))
    return H


def main():
    rfit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rslip = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    modes = (sys.argv[3],) if len(sys.argv) > 3 else ('site', 'bond')
    rB = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    rmax = max(rfit, rslip)
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    th = {(r['tag'], r['mode']): float(r['pc']) for r in rows}
    comp = {r['tag']: r['vertex_types'] for r in rows}
    kk = {r['tag']: int(r['k']) for r in rows}
    tags = sorted(comp)
    pairs = list(csv.DictReader(open('paper/supplemental/slip_pairs.csv')))
    fam = json.load(open('results/slip_family.json'))
    need_deep = {p['id_a'] for p in pairs} | {p['id_b'] for p in pairs} | {f[x] for f in fam for x in ('staggered', 'eclipsed')}
    intern = [dict() for _ in range(rmax + 1)]
    hist, nvs = {}, {}
    for t in tags:
        d = json.load(open(f'data/cells/{t}.json'))
        rm = rmax if t in need_deep else rfit
        Hs = [refine(*darts_of(d, m), rm, intern) for m in (False, True)]
        hist[t] = [Hs[0][r] + Hs[1][r] for r in range(rm + 1)]
        nvs[t] = d['nv']
    ncol = [len(x) for x in intern]
    print('colors per radius', ncol, flush=True)
    off = np.cumsum([0] + ncol[:rfit + 1])

    def design(ts, r, scale0=100.0):
        data, ii, jj = [], [], []
        for i, t in enumerate(ts):
            for q in range(r + 1):
                for c, m in hist[t][q].items():
                    ii.append(i)
                    jj.append(off[q] + c)
                    data.append(m / (2 * nvs[t]) * (scale0 if q == 0 else 1.0))
        return sp.csr_matrix((data, (ii, jj)), shape=(len(ts), off[r + 1]))

    def solve(X, y, lam):
        return lsqr(X, y, damp=math.sqrt(lam), atol=1e-10, btol=1e-10, iter_lim=10000)[0]

    lams = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    out = dict(colors=ncol, modes={})
    # A. extrapolation in k
    train = [t for t in tags if kk[t] <= 7]
    test = [t for t in tags if kk[t] >= 8]
    tr6 = [t for t in tags if kk[t] <= 6]
    va7 = [t for t in tags if kk[t] == 7]
    # B. groups
    groups = collections.defaultdict(list)
    for t in tags:
        groups[comp[t]].append(t)
    multi = sorted([g for g in groups.values() if len(g) >= 2], key=lambda g: g[0])
    rng = np.random.default_rng(12345)
    fold = rng.integers(0, 5, len(multi))
    gtags = [t for g in multi for t in g]
    gidx = np.concatenate([[i] * len(g) for i, g in enumerate(multi)])

    def center(X, y, gid):
        X = X.tocsr().astype(float)
        cnt = np.bincount(gid)
        S = sp.csr_matrix((np.ones(len(gid)), (gid, np.arange(len(gid)))), shape=(len(cnt), len(gid)))
        mx = sp.diags(1.0 / cnt) @ (S @ X)
        my = (S @ y) / cnt
        return (X - mx[gid]).tocsr(), y - my[gid]

    for mode in modes:
        Y = lambda ts: np.array([th[(t, mode)] for t in ts])
        resA, resB = [], []
        for r in range(rfit + 1):
            errs = [np.sqrt(np.mean((design(va7, r) @ solve(design(tr6, r), Y(tr6), lam) - Y(va7)) ** 2)) for lam in lams]
            lam = lams[int(np.argmin(errs))]
            e = design(test, r) @ solve(design(train, r), Y(train), lam) - Y(test)
            resA.append(dict(r=r, lam=lam, rms=float(np.sqrt(np.mean(e ** 2))),
                             by_k={k: float(np.sqrt(np.mean(e[[i for i, t in enumerate(test) if kk[t] == k]] ** 2))) for k in (8, 9, 10)}))
            print(mode, 'A', resA[-1], flush=True)
        Xall = {r: design(gtags, r, 1.0) for r in range(1, rB + 1)}
        yall = Y(gtags)
        _, yc = center(Xall[1], yall, gidx)
        base = float(np.sqrt(np.mean(yc ** 2)))
        for r in range(1, rB + 1):
            Xc, yc = center(Xall[r], yall, gidx)
            # ridge parameter: chosen once, fitting on folds 2-4 and validating on fold 1 (fold 0 not used)
            fit_in, inner = fold[gidx] >= 2, fold[gidx] == 1
            errs = [np.mean((Xc[inner] @ solve(Xc[fit_in], yc[fit_in], lam) - yc[inner]) ** 2) for lam in lams[2:6]]
            lam = lams[2:6][int(np.argmin(errs))]
            pred = np.zeros_like(yc)
            for f in range(5):
                te = fold[gidx] == f
                pred[te] = Xc[te] @ solve(Xc[~te], yc[~te], lam)
            res = pred - yc
            resB.append(dict(r=r, lam=lam, rms=float(np.sqrt(np.mean(res ** 2))), base_rms=base,
                             r2=float(1 - np.mean(res ** 2) / np.mean(yc ** 2))))
            print(mode, 'B', resB[-1], flush=True)
        out['modes'][mode] = dict(A=resA, B=resB)
    out['B_groups'] = len(multi)
    out['B_tilings'] = len(gtags)
    # C. slip pairs
    def first(a, b):
        return next((r for r in range(rmax + 1) if hist[a][r] != hist[b][r]), None)
    fr = [first(p['id_a'], p['id_b']) for p in pairs]
    out['slip_first_radius'] = sorted(collections.Counter(str(x) for x in fr).items())
    out['family_first_radius'] = {f['h']: first(f['staggered'], f['eclipsed']) for f in fam}
    byr = collections.defaultdict(list)
    for p, x in zip(pairs, fr):
        byr[str(x)].append((abs(float(p['d_bond'])) > 5 * float(p['err_bond']), abs(float(p['d_bond'])),
                            abs(float(p['d_site'])) > 5 * float(p['err_site']), abs(float(p['d_site']))))
    out['slip_by_radius'] = {r: dict(n=len(v), bond_resolved=sum(a for a, _, _, _ in v), bond_median_abs=float(np.median([b for _, b, _, _ in v])),
                                     site_resolved=sum(c for _, _, c, _ in v), site_median_abs=float(np.median([e for _, _, _, e in v])))
                             for r, v in sorted(byr.items())}
    print('slip first radius', out['slip_first_radius'], 'family', out['family_first_radius'])
    print('slip by radius', json.dumps(out['slip_by_radius'], indent=1))
    json.dump(out, open('results/local_hierarchy_%s.json' % '_'.join(modes), 'w'), indent=1)


if __name__ == '__main__':
    main()
