"""Convert a HyperRogue .tes description of a Euclidean tiling by regular polygons
into a periodic graph: primitive translations T1, T2, vertex orbits of one cell
and the edges of one cell as (u, w, da, db) = edge from orbit u in cell (0, 0)
to orbit w in cell (da, db).

Checks performed for every tiling:
  * every edge class of the .tes gluing is used exactly once;
  * gluing is consistent (a tile met twice from different sides agrees);
  * Euler characteristic of one cell: V - E + F = 0;
  * number of vertex orbits under the full symmetry group (= k) and the
    multiset of vertex configurations agree with the .tes header.

Usage: python tes2cell.py OUT_DIR file1.tes [file2.tes ...]
"""
import cmath
import json
import math
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

TOL = 1e-6


# ---------------------------------------------------------------- parsing

def parse_tes(path):
    text = open(path, encoding='utf-8').read()
    header = ''
    tiles, rep, conway = [], {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('##'):
            header = line[2:].strip()
        elif line.startswith('unittile'):
            inner = line[line.index('(') + 1:line.rindex(')')]
            tiles.append([float(x) for x in inner.split(',')])
        elif line.startswith('repeat'):
            t, q = re.findall(r'\d+', line)
            rep[int(t)] = int(q)
        elif line.startswith('conway'):
            conway = line[line.index('"') + 1:line.rindex('"')]
        elif line in ('e2.', 'angleunit(deg)', ''):
            pass
        else:
            raise ValueError(f'{path}: unsupported line {line!r}')
    period = []
    for t, ang in enumerate(tiles):
        q = rep.get(t, 1)
        assert len(ang) % q == 0
        period.append(len(ang) // q)

    def edge(tok):
        m = re.fullmatch(r"(\d+)('*)(?:@(\d+))?", tok)
        i, primes, at = m.groups()
        t = int(at) if at is not None else len(primes)
        return (t, int(i))

    glue = {}
    for grp in re.finditer(r'([(\[])([^)\]]*)([)\]])', conway):
        mirror = grp.group(1) == '['
        toks = grp.group(2).split()
        a = edge(toks[0])
        b = edge(toks[1]) if len(toks) == 2 else a
        for x, y in ((a, b), (b, a)):
            assert x not in glue or glue[x] == (y, mirror), (path, x)
            glue[x] = (y, mirror)
    for t in range(len(tiles)):
        for i in range(period[t]):
            assert (t, i) in glue, f'{path}: edge {i} of tile {t} not glued'
    assert len(glue) == sum(period)
    return dict(header=header, angles=tiles, period=period, glue=glue)


def header_types(header):
    """'Euclidean, (3,12,12)A, (3,4,3,12)F' -> list of canonical configs."""
    out = []
    for m in re.finditer(r'\(([\d,]+)\)[A-Za-z0-9]*?(?:x(\d+))?(?=,|$)', header):
        mult = int(m.group(2)) if m.group(2) else 1
        out += [canon([int(x) for x in m.group(1).split(',')])] * mult
    return out


def canon(cfg):
    n = len(cfg)
    cands = []
    for seq in (cfg, cfg[::-1]):
        for s in range(n):
            cands.append(tuple(seq[s:] + seq[:s]))
    return min(cands)


# ---------------------------------------------------------------- geometry

def build_patch(tes, radius):
    """Place tiles by BFS from tile 0 until all tiles with centroid inside radius."""
    angles, period, glue = tes['angles'], tes['period'], tes['glue']

    def polygon(t, k, start, end, orient):
        """Vertices of tile t whose edge k runs start->end; orient=+1 clockwise."""
        n = len(angles[t])
        V = [None] * n
        V[k], V[(k + 1) % n] = start, end
        d = end - start
        j = (k + 1) % n
        for _ in range(n - 2):
            a = math.radians(angles[t][j - 1 if j > 0 else n - 1])  # angle at vertex j
            d = d * cmath.exp(-1j * orient * (math.pi - a))
            V[(j + 1) % n] = V[j] + d
            j = (j + 1) % n
        # closure check
        a = math.radians(angles[t][j - 1 if j > 0 else n - 1])
        d = d * cmath.exp(-1j * orient * (math.pi - a))
        assert abs(V[j] + d - V[(j + 1) % n]) < 1e-9
        return V

    n0 = len(angles[0])
    V0 = polygon(0, 0, 0j, 1 + 0j, +1)
    c0 = sum(V0) / n0
    V0 = [v - c0 for v in V0]
    tiles = [(0, V0, +1)]
    index = {}

    def key(z):
        return (round(z.real * 1e5), round(z.imag * 1e5))

    index[key(sum(V0) / n0)] = 0
    queue = [0]
    qi = 0
    while qi < len(queue):
        ti = queue[qi]
        qi += 1
        t, V, orient = tiles[ti]
        n = len(V)
        for k in range(n):
            (t2, i2), mirror = glue[(t, k % period[t])]
            o2 = -orient if mirror else orient
            A, B = V[k], V[(k + 1) % n]
            # non-mirrored: same orientation, shared edge reversed
            s, e = (B, A) if o2 == orient else (A, B)
            V2 = polygon(t2, i2, s, e, o2)
            c2 = sum(V2) / len(V2)
            kk = key(c2)
            if kk in index:
                tj = index[kk]
                assert tiles[tj][0] == t2, 'inconsistent tile type'
                Vo = tiles[tj][1]
                assert all(min(abs(z - w) for w in Vo) < 1e-6 for z in V2), 'inconsistent placement'
                continue
            if abs(c2) > radius:
                continue
            index[kk] = len(tiles)
            tiles.append((t2, V2, o2))
            queue.append(len(tiles) - 1)
    return tiles


def patch_graph(tiles):
    pts, pid = [], {}

    def vid(z):
        k = (round(z.real * 1e5), round(z.imag * 1e5))
        if k not in pid:
            # tolerate rounding at bin boundaries
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    k2 = (k[0] + dx, k[1] + dy)
                    if k2 in pid and abs(pts[pid[k2]] - z) < TOL:
                        pid[k] = pid[k2]
                        return pid[k]
            pid[k] = len(pts)
            pts.append(z)
        return pid[k]

    edges = set()
    faces = []
    inc = {}
    for t, V, _ in tiles:
        ids = [vid(z) for z in V]
        n = len(ids)
        c = sum(V) / n
        faces.append((n, c))
        for a in range(n):
            u, w = ids[a], ids[(a + 1) % n]
            edges.add((min(u, w), max(u, w)))
            inc.setdefault(u, []).append((n, c))
    P = np.array([[z.real, z.imag] for z in pts])
    E = np.array(sorted(edges))
    return P, E, faces, inc


def vertex_config(z, incident):
    """Cyclic list of polygon sizes around vertex z, or None if incomplete."""
    tot = sum((n - 2) * 180.0 / n for n, _ in incident)
    if abs(tot - 360) > 1e-6:
        return None
    srt = sorted(incident, key=lambda nc: cmath.phase(nc[1] - z))
    return canon([n for n, _ in srt])


# ---------------------------------------------------------------- lattice

class Checker:
    """Tests whether an isometry maps the (inner part of the) patch to itself."""

    def __init__(self, P, E, r_inner):
        self.P = P
        self.mid = 0.5 * (P[E[:, 0]] + P[E[:, 1]])
        self.tP = cKDTree(P)
        self.tM = cKDTree(self.mid)
        self.r = r_inner
        self.inP = P[np.hypot(P[:, 0], P[:, 1]) < r_inner]
        self.inM = self.mid[np.hypot(self.mid[:, 0], self.mid[:, 1]) < r_inner]

    def ok(self, A, b):
        """x -> A x + b maps inner vertices and edge midpoints into the patch sets."""
        for X, tree in ((self.inP, self.tP), (self.inM, self.tM)):
            d, _ = tree.query(X @ A.T + b)
            if d.max() > 1e-5:
                return False
        return True


def find_translations(P, E, chk, origin_idx, configs):
    x0 = P[origin_idx]
    c0 = configs[origin_idx]
    D = P - x0
    L = np.hypot(D[:, 0], D[:, 1])
    order = np.argsort(L)
    I2 = np.eye(2)
    T1 = None
    for j in order:
        if L[j] < 1e-6 or L[j] > chk.r / 2:
            continue
        if configs[j] != c0:
            continue
        v = D[j]
        if T1 is not None and abs(T1[0] * v[1] - T1[1] * v[0]) < 1e-6:
            continue
        if chk.ok(I2, v):
            if T1 is None:
                T1 = v
            else:
                return T1, v
    raise RuntimeError('translations not found')


def reduce_basis(a, b):
    # Lagrange-Gauss; tolerances stop the endless swap when |a.b| = |a|^2/2 exactly
    # (hexagonal lattices) and rounding decides the sign of the last bit
    while True:
        if a @ a > b @ b + 1e-9:
            a, b = b, a
        r = (a @ b) / (a @ a)
        if abs(r) <= 0.5 + 1e-9:
            break
        b = b - round(r) * a
    if a[0] * b[1] - a[1] * b[0] < 0:
        b = -b
    return a, b


def symmetries(P, chk, reps, T1, T2):
    """All isometries mapping vertex rep[0] to some rep (mod translations)."""
    tree = chk.tP
    x0 = P[reps[0]]
    # unit directions of edges at x0
    def dirs(x):
        idx = tree.query_ball_point(x, 1.0 + 1e-6)
        out = []
        for j in idx:
            d = P[j] - x
            r = math.hypot(*d)
            if abs(r - 1) < 1e-6:
                out.append(d)
        return out
    d0 = dirs(x0)[0]
    found = []
    for r in reps:
        y = P[r]
        for d in dirs(y):
            for refl in (False, True):
                # linear part: rotate d0 to d, optionally after reflection across d0
                ang = math.atan2(d[1], d[0]) - math.atan2(d0[1], d0[0])
                R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
                if refl:
                    phi = math.atan2(d0[1], d0[0])
                    S = np.array([[math.cos(2 * phi), math.sin(2 * phi)],
                                  [math.sin(2 * phi), -math.cos(2 * phi)]])
                    A = R @ S
                else:
                    A = R
                b = y - A @ x0
                if chk.ok(A, b):
                    found.append((A, b, r))
    return found


def process(path, radius=None, sym=True):
    """sym=False skips the symmetry group (k, rotation orders): checks are then
    chi = 0 and the set of vertex configurations against the header."""
    tes = parse_tes(path)
    want = header_types(tes['header'])
    k_want = len(want)
    radius = radius or 30.0
    for attempt in range(6):
        try:
            return build(path, tes, want, k_want, radius, sym)
        except NeedLarger as e:
            radius = max(radius * 1.6, e.radius)
    raise RuntimeError(f'{path}: patch too small')


class NeedLarger(Exception):
    def __init__(self, radius):
        self.radius = radius


def build(path, tes, want, k_want, radius, sym=True):
    tiles = build_patch(tes, radius)
    P, E, faces, inc = patch_graph(tiles)
    r_inner = radius - 4.0
    configs = [None] * len(P)
    for i in range(len(P)):
        z = complex(*P[i])
        if abs(z) < radius - 3.0:
            configs[i] = vertex_config(z, inc[i])
    chk = Checker(P, E, r_inner * 0.45)
    origin = int(np.argmin(np.hypot(P[:, 0], P[:, 1])))
    try:
        T1, T2 = find_translations(P, E, chk, origin, configs)
    except RuntimeError:
        raise NeedLarger(radius * 1.6)
    T1, T2 = reduce_basis(T1, T2)
    s = math.hypot(*T1) + math.hypot(*T2)
    need = 4 * s + 14 if sym else 2.5 * s + 16
    if radius < need:
        raise NeedLarger(need)
    if sym:
        chk = Checker(P, E, 2 * s + 4)
    M = np.column_stack([T1, T2])
    Minv = np.linalg.inv(M)
    x0 = P[origin]
    r_cell = r_inner * 0.5

    def cell_orbit(x):
        f = Minv @ (x - x0)
        c = np.floor(f + 1e-7)
        r = f - c
        return (int(c[0]), int(c[1])), (round(r[0] * 1e5) % 100000, round(r[1] * 1e5) % 100000)

    orbit_id, reps = {}, []
    vcell = {}
    for i in range(len(P)):
        if math.hypot(*P[i]) < r_cell:
            c, o = cell_orbit(P[i])
            if o not in orbit_id:
                orbit_id[o] = len(reps)
                reps.append(None)
            if c == (0, 0):
                reps[orbit_id[o]] = i
            vcell[i] = (c, orbit_id[o])
    nv = len(reps)
    assert all(r is not None for r in reps), 'missing representative in cell 0'
    ecls = set()
    for u, w in E:
        if u in vcell and w in vcell:
            (cu, ou), (cw, ow) = vcell[u], vcell[w]
            d = (cw[0] - cu[0], cw[1] - cu[1])
            a = (ou, ow, d[0], d[1])
            b = (ow, ou, -d[0], -d[1])
            ecls.add(min(a, b))
    ne = len(ecls)
    fcls = set()
    for n, c in faces:
        x = np.array([c.real, c.imag])
        if math.hypot(*x) < r_cell - 3:
            fcls.add(cell_orbit(x)[1])
    nf = len(fcls)
    deg = [0] * nv
    for u, w, _, _ in ecls:
        deg[u] += 1
        deg[w] += 1
    cfg = [configs[r] for r in reps]
    for o in range(nv):
        assert deg[o] == len(cfg[o]), 'degree mismatch'
    euler = nv - ne + nf
    # symmetry group -> vertex orbits (k) and rotation orders
    syms = symmetries(P, chk, reps, T1, T2) if sym else []
    parent = list(range(nv))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    rot_orders = set()
    for A, b, r in syms:
        if np.linalg.det(A) > 0:
            ang = math.degrees(math.atan2(A[1, 0], A[0, 0])) % 360
            for q in (1, 2, 3, 4, 6):
                if abs((ang * q) % 360) < 1e-4 or abs((ang * q) % 360 - 360) < 1e-4:
                    rot_orders.add(q)
                    break
        # map every orbit rep through the symmetry
        for o in range(nv):
            y = A @ P[reps[o]] + b
            _, j = chk.tP.query(y)
            _, oo = cell_orbit(P[j])
            parent[find(o)] = find(orbit_id[oo])
    if sym:
        k = len({find(o) for o in range(nv)})
        orbit_cfgs = sorted(cfg[o] for o in {find(o): o for o in range(nv)}.values())
        ok = (euler == 0 and k == k_want and sorted(set(orbit_cfgs)) == sorted(set(want))
              and orbit_cfgs == sorted(want))
        iso = 3 in rot_orders or 4 in rot_orders or 6 in rot_orders
    else:
        k, iso = None, None
        ok = euler == 0 and set(cfg) == set(want) and len(set(cfg)) <= k_want
    frac = {}
    for c in cfg:
        s = '.'.join(map(str, c))
        frac[s] = frac.get(s, 0) + 1
    return dict(
        name=os.path.splitext(os.path.basename(path))[0],
        path=path.replace('\\', '/'),
        header=tes['header'],
        k=k, k_header=k_want, ok=ok, euler=euler,
        nv=nv, ne=ne, nf=nf,
        mean_degree=2 * ne / nv,
        T1=list(map(float, T1)), T2=list(map(float, T2)),
        cell_area=float(abs(T1[0] * T2[1] - T1[1] * T2[0])),
        rot_orders=sorted(rot_orders) if sym else None, isotropic=iso,
        vertex_configs=[list(c) for c in cfg],
        config_counts=frac,
        pos=[list(map(float, P[r] - x0)) for r in reps],
        edges=sorted(ecls),
    )


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    bad = 0
    for path in sys.argv[2:]:
        d = process(path)
        if not d['ok']:
            bad += 1
        rel = path.replace('\\', '/').split('data/tes/')[-1]
        tag = rel[:-4].replace('/', '__').replace(' ', '_').replace('(', '').replace(')', '')
        json.dump(d, open(os.path.join(out, tag + '.json'), 'w'), indent=None)
        print(f"{'OK ' if d['ok'] else 'BAD'} k={d['k']}/{d['k_header']} nv={d['nv']:3d} ne={d['ne']:3d} "
              f"nf={d['nf']:3d} z={d['mean_degree']:.4f} rot={d['rot_orders']} {rel}")
    print('bad:', bad)


if __name__ == '__main__':
    main()
