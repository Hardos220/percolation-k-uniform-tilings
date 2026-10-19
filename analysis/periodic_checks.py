"""Exact checks of the periodic graphs (data/cells/*.json), independent of the patch construction.

For every tiling:
  * every edge has unit length; the angles between consecutive edges at every vertex give the vertex type of
    the catalog; every face, traced through the rotation system, closes after n steps with zero net
    translation and is an n-gon at each of its corners (so the cell describes an edge-to-edge tiling by
    regular polygons);
  * the cell is primitive: no vector between two vertices of the same type, other than a lattice vector, maps
    the vertices and edges onto vertices and edges (modulo translations);
  * the full symmetry group: an isometry is determined by the image of one flag (vertex, edge, side), so all
    candidates map vertex 0 and its first edge onto a vertex of the same type and one of its edges, with or
    without reflection; a candidate is a symmetry if its linear part maps the lattice onto itself and it maps
    all vertices and edges of the cell onto vertices and edges modulo translations. The number of vertex
    orbits must equal k.
Across tilings: two tilings are the same tiling if and only if an isometry maps one periodic graph onto the
other (lattice onto lattice, vertices and edges onto vertices and edges). Tilings are grouped by composition,
number of vertices per cell, and cell area (invariants of an isometry); within each group every pair is tested.

usage: python analysis/periodic_checks.py [N_FILES]   -> results/periodic_checks.json
Workers: $RMQ_CPU (default 1).
"""
import collections
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

TOL = 1e-6
SIZES = {3: 3, 4: 4, 6: 6, 8: 8, 12: 12}


def canon(seq):
    """Canonical cyclic sequence under rotation and reflection."""
    n = len(seq)
    cands = []
    for s in (list(seq), list(seq)[::-1]):
        for i in range(n):
            cands.append(tuple(s[i:] + s[:i]))
    return min(cands)


class Cell:
    def __init__(self, d):
        self.d = d
        self.nv = d['nv']
        self.P = np.array(d['pos'], float)
        self.T1, self.T2 = np.array(d['T1']), np.array(d['T2'])
        self.M = np.column_stack([self.T1, self.T2])
        self.Minv = np.linalg.inv(self.M)
        self.cfg = [tuple(c) for c in d['vertex_configs']]
        self.edges = [tuple(e) for e in d['edges']]
        self.eset = set()
        for u, w, a, b in self.edges:
            self.eset.add((u, w, a, b))
            self.eset.add((w, u, -a, -b))
        # residues of the vertex positions modulo the lattice
        self.key = {}
        self.off = np.zeros((self.nv, 2), int)
        for v in range(self.nv):
            c, k = self.reduce(self.P[v])
            assert k not in self.key, 'two vertices at the same position'
            self.key[k] = v
            self.off[v] = c
        # darts: (neighbour, a, b, direction angle), sorted counterclockwise
        nb = collections.defaultdict(list)
        for u, w, a, b in self.edges:
            nb[u].append((w, a, b))
            nb[w].append((u, -a, -b))
        self.darts = {}
        for u in range(self.nv):
            lst = []
            for (w, a, b) in nb[u]:
                vec = self.P[w] + a * self.T1 + b * self.T2 - self.P[u]
                lst.append((math.atan2(vec[1], vec[0]), w, a, b, vec))
            lst.sort(key=lambda x: x[0])
            self.darts[u] = lst

    def reduce(self, x):
        f = self.Minv @ x
        c = np.floor(f + 1e-7)
        r = f - c
        return c.astype(int), (round(r[0] * 1e5) % 100000, round(r[1] * 1e5) % 100000)

    def validity(self):
        """Unit edges, vertex types from the angles, faces closing as regular polygons."""
        for u in range(self.nv):
            ds = self.darts[u]
            if len(ds) != len(self.cfg[u]):
                return 'degree'
            for t in ds:
                if abs(np.hypot(*t[4]) - 1) > TOL:
                    return 'edge length'
            sizes = []
            for i in range(len(ds)):
                th = (ds[(i + 1) % len(ds)][0] - ds[i][0]) % (2 * math.pi)
                n = 2 * math.pi / (math.pi - th)
                if abs(n - round(n)) > 1e-6 or round(n) not in SIZES:
                    return 'angle'
                sizes.append(round(n))
            if canon(sizes) != canon(self.cfg[u]):
                return 'vertex type'
        # faces: from dart (u -> w) turn to the next dart clockwise at w, i.e. the face to the right
        seen = set()
        nf = 0.0
        for u in range(self.nv):
            for (_, w, a, b, _) in self.darts[u]:
                if (u, w, a, b) in seen:
                    continue
                cu, cw, ca, cb = u, w, a, b
                tot = np.zeros(2, int)
                length = 0
                while True:
                    seen.add((cu, cw, ca, cb))
                    tot += (ca, cb)
                    length += 1
                    ds = self.darts[cw]
                    idx = [i for i, t in enumerate(ds) if t[1] == cu and t[2] == -ca and t[3] == -cb][0]
                    nxt = ds[(idx - 1) % len(ds)]
                    cu, cw, ca, cb = cw, nxt[1], nxt[2], nxt[3]
                    if (cu, cw, ca, cb) == (u, w, a, b):
                        break
                    if length > 12:
                        return 'face does not close'
                if tot.any():
                    return 'face with net translation'
                if length not in SIZES:
                    return 'face size'
                nf += 1
        if self.nv - len(self.edges) + nf != 0:
            return 'euler'
        return 'ok'


def isometries(G1, G2, linear_only=None, first=False):
    """All isometries g mapping periodic graph G1 onto G2 (as (A, b, vertex permutation))."""
    out = []
    u0 = G1.darts[0][0][4]
    phi0 = math.atan2(u0[1], u0[0])
    S0 = np.array([[math.cos(2 * phi0), math.sin(2 * phi0)], [math.sin(2 * phi0), -math.cos(2 * phi0)]])
    E1 = [(u, w, np.array([a, b])) for u, w, a, b in G1.edges]
    for wv in range(G2.nv):
        if G2.cfg[wv] != G1.cfg[0]:
            continue
        for t in G2.darts[wv]:
            u1 = t[4]
            ang = math.atan2(u1[1], u1[0]) - phi0
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            for refl in (False, True):
                A = R @ S0 if refl else R
                if linear_only is not None and not linear_only(A):
                    continue
                B = G2.Minv @ A @ G1.M
                Bi = np.round(B)
                if np.abs(B - Bi).max() > 1e-6 or abs(round(np.linalg.det(Bi))) != 1:
                    continue
                Bi = Bi.astype(int)
                b = G2.P[wv] - A @ G1.P[0]
                perm, offs, good = [], [], True
                for v in range(G1.nv):
                    c, k = G2.reduce(A @ G1.P[v] + b)
                    j = G2.key.get(k)
                    if j is None or G2.cfg[j] != G1.cfg[v]:
                        good = False
                        break
                    perm.append(j)
                    offs.append(c - G2.off[j])
                if not good:
                    continue
                for u, w, ab in E1:
                    du = offs[u]
                    dw = offs[w] + Bi @ ab
                    if (perm[u], perm[w], int(dw[0] - du[0]), int(dw[1] - du[1])) not in G2.eset:
                        good = False
                        break
                if good:
                    out.append((A, b, perm))
                    if first:
                        return out
    return out


def check_one(path):
    d = json.load(open(path))
    G = Cell(d)
    tag = os.path.splitext(os.path.basename(path))[0]
    res = dict(tag=tag, k_header=d['k_header'], nv=G.nv)
    res['valid'] = G.validity()
    # primitivity: pure translations other than lattice vectors
    extra = 0
    for A, b, perm in isometries(G, G, linear_only=lambda A: np.abs(A - np.eye(2)).max() < 1e-9):
        f = G.Minv @ b
        if np.abs(f - np.round(f)).max() > 1e-6:
            extra += 1
    res['extra_translations'] = extra
    syms = isometries(G, G)
    parent = list(range(G.nv))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    rots = set()
    for A, b, perm in syms:
        for v in range(G.nv):
            ra, rb = find(v), find(perm[v])
            if ra != rb:
                parent[ra] = rb
        if np.linalg.det(A) > 0:
            ang = math.degrees(math.atan2(A[1, 0], A[0, 0])) % 360
            for q in (1, 2, 3, 4, 6):
                if min(abs(ang * q % 360), abs(ang * q % 360 - 360)) < 1e-4:
                    rots.add(q)
                    break
    res['k'] = len({find(v) for v in range(G.nv)})
    res['point_group_order'] = len(syms) // 1  # symmetries modulo translations
    res['rot_orders'] = sorted(rots)
    res['key'] = [sorted(collections.Counter('.'.join(map(str, c)) for c in G.cfg).items()), G.nv,
                  round(abs(np.linalg.det(G.M)), 4)]
    return res


def pair_iso(args):
    p1, p2 = args
    G1, G2 = Cell(json.load(open(p1))), Cell(json.load(open(p2)))
    return os.path.basename(p1)[:-5], os.path.basename(p2)[:-5], bool(isometries(G1, G2, first=True))


def main():
    files = sorted(glob.glob('data/cells/*.json'))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    nw = int(os.environ.get('RMQ_CPU', '1'))
    prog = os.environ.get('RMQ_PROGRESS_FILE')
    t0 = time.time()
    res = []
    with mp.Pool(nw) as pool:
        for i, r in enumerate(pool.imap_unordered(check_one, files, chunksize=8)):
            res.append(r)
            if prog and i % 200 == 0:
                try:
                    open(prog, 'w').write(f'{i}/{len(files)} symmetry groups')
                except OSError:
                    pass
        res.sort(key=lambda r: r['tag'])
        groups = collections.defaultdict(list)
        for r in res:
            groups[json.dumps(r['key'])].append(r['tag'])
        pairs = [(f'data/cells/{a}.json', f'data/cells/{b}.json')
                 for g in groups.values() for i, a in enumerate(g) for b in g[i + 1:]]
        iso = []
        for i, (a, b, same) in enumerate(pool.imap_unordered(pair_iso, pairs, chunksize=16)):
            if same:
                iso.append((a, b))
            if prog and i % 500 == 0:
                try:
                    open(prog, 'w').write(f'{i}/{len(pairs)} pairs')
                except OSError:
                    pass
    bad_valid = [r['tag'] for r in res if r['valid'] != 'ok']
    bad_k = [(r['tag'], r['k'], r['k_header']) for r in res if r['k'] != r['k_header']]
    bad_prim = [r['tag'] for r in res if r['extra_translations']]
    big = collections.Counter(len(g) for g in groups.values())
    out = dict(n=len(res), invalid=bad_valid, k_mismatch=bad_k, nonprimitive=bad_prim,
               n_groups=len(groups), group_sizes=sorted(big.items()), n_pairs_tested=len(pairs), isometric_pairs=iso,
               per_tiling={r['tag']: dict(k=r['k'], point_group_order=r['point_group_order'], rot=r['rot_orders'])
                           for r in res})
    json.dump(out, open('results/periodic_checks.json' if len(sys.argv) == 1 else 'results/periodic_checks_test.json', 'w'))
    print(f"tilings {len(res)}  invalid {len(bad_valid)}  k mismatch {len(bad_k)}  nonprimitive {len(bad_prim)}")
    print(f"invariant groups {len(groups)}  pairs tested {len(pairs)}  isometric pairs {len(iso)}")
    print('examples', bad_valid[:5], bad_k[:5], bad_prim[:5], iso[:5])
    print('time %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
