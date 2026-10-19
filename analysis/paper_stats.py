"""Statistics of the threshold table for the paper: empirical formulas, composition, site-bond relation.

usage: python analysis/paper_stats.py            -> results/paper_stats.json (+ printed summary)
Input: results/tables/all_thresholds.csv
"""
import collections
import csv
import json
import math
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq

TAU = 187 / 91
C_STAR = (TAU - 2) / (TAU - 1)  # = 5/96, Neher-Mecke-Wagner estimator chi(p) = C (2p - 1)

# Galam-Mauger (PRE 53, 2177): p_c = p0 [(d-1)(q-1)]^-a d^b, d = 2; class I (square, triangular, honeycomb)
GM = {'site': (0.8889, 0.3601, 0.0), 'bond': (0.6558, 0.6897, 0.6897)}
LEBRECHT_A = 0.16428  # Physica A 569, 125802: p_s / p_b = a z^3 / (z - 1)^2


def composition(r):
    out = {}
    for part in r['vertex_types'].split(';'):
        t, f = part.rsplit(':', 1)
        out[t] = Fraction(f)
    return out


def local_counts(comp):
    """Per vertex: mean degree z, faces f_n of each size n, degree distribution {z_v: fraction}."""
    z = 0.0
    faces = collections.Counter()
    deg = collections.Counter()
    for t, x in comp.items():
        sides = list(map(int, t.split('.')))
        z += float(x) * len(sides)
        deg[len(sides)] += float(x)
        for n in sides:
            faces[n] += float(x) / n
    return z, faces, deg


def mec_site(p, z, faces, deg):
    """Mean Euler characteristic per site of the occupied complex (Neher et al. 2008, Eq. 7)."""
    return p - z / 2 * p * p + sum(f * p ** n for n, f in faces.items())


def mec_bond(p, z, faces, deg):
    """Per bond, covering-lattice convention (Neher et al. 2008, Eq. 8, generalized to several vertex types)."""
    e = z / 2  # bonds per vertex
    touched = sum(x * (1 - (1 - p) ** zv) for zv, x in deg.items())
    return (touched - e * p + sum(f * p ** n for n, f in faces.items())) / e


def root(f, lo=1e-6, hi=1 - 1e-9):
    grid = np.linspace(lo, hi, 4001)
    v = f(grid)
    rs = [brentq(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if v[i] * v[i + 1] < 0]
    return rs


def euler_estimates(r):
    z, faces, deg = local_counts(composition(r))
    g = mec_site if r['mode'] == 'site' else mec_bond
    p0 = [x for x in root(lambda p: g(p, z, faces, deg)) if 0.05 < x < 0.999]
    ps = [x for x in root(lambda p: g(p, z, faces, deg) - C_STAR * (2 * p - 1)) if 0.05 < x < 0.999]
    return (p0[0] if len(p0) == 1 else None), (ps[0] if len(ps) == 1 else None)


def gm(mode, z):
    p0, a, b = GM[mode]
    return p0 * ((2 - 1) * (z - 1)) ** (-a) * 2 ** b


def summary(x):
    x = np.asarray([v for v in x if v is not None], float)
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std()), rms=float(np.sqrt((x * x).mean())),
                maxabs=float(np.abs(x).max()), p05=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))


def main():
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    th = {}
    for r in rows:
        th[(r['tag'], r['mode'])] = r
    tags = sorted({r['tag'] for r in rows})
    out = {}

    # 1. Euler-characteristic estimates (Neher-Mecke-Wagner)
    eul = {'site': collections.defaultdict(list), 'bond': collections.defaultdict(list)}
    p0_above = {'site': [0, 0], 'bond': [0, 0]}
    for r in rows:
        p0, ps = euler_estimates(r)
        pc = float(r['pc'])
        grp = 'k<=4' if int(r['k']) <= 4 else 'k>=5'
        eul[r['mode']][grp].append(None if ps is None else ps - pc)
        eul[r['mode']]['p0_' + grp].append(None if p0 is None else p0 - pc)
        if p0 is not None and abs(pc - 0.5) > 1e-3:  # Neher et al.: p0 lies beyond p_c, away from 1/2
            p0_above[r['mode']][0] += 1
            p0_above[r['mode']][1] += (p0 - pc) * (pc - 0.5) > 0
    out['euler'] = {m: {g: summary(v) for g, v in d.items()} for m, d in eul.items()}
    out['euler_p0_bound'] = {m: dict(tested=v[0], p0_beyond_pc=int(v[1])) for m, v in p0_above.items()}

    # 2. Galam-Mauger with q -> mean degree; 3. Lebrecht et al. ratio formula
    gmres = {'site': [], 'bond': []}
    leb, sz, zb = [], [], []
    site_bond = []
    for t in tags:
        rs, rb = th[(t, 'site')], th[(t, 'bond')]
        z = float(rs['mean_degree'])
        ps, pb = float(rs['pc']), float(rb['pc'])
        gmres['site'].append(gm('site', z) - ps)
        gmres['bond'].append(gm('bond', z) - pb)
        leb.append(LEBRECHT_A * z ** 3 / (z - 1) ** 2 - ps / pb)
        sz.append(ps * float(rs['sites_per_area']) * math.pi / 4)  # Scher-Zallen filling fraction (unit edge)
        zb.append(z * pb)
        site_bond.append((ps, pb, z, int(rs['k'])))
    out['galam_mauger'] = {m: summary(v) for m, v in gmres.items()}
    out['lebrecht_ratio'] = summary(leb)
    out['scher_zallen_phi'] = summary(sz)
    out['z_pbond'] = summary(zb)
    a = np.array(site_bond)
    out['site_bond_corr'] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    out['site_z_corr'] = float(np.corrcoef(a[:, 0], a[:, 2])[0, 1])
    out['bond_z_corr'] = float(np.corrcoef(a[:, 1], a[:, 2])[0, 1])

    # 4. composition: identical vertex-type fractions -> spread of p_c
    groups = collections.defaultdict(list)
    for t in tags:
        key = tuple(sorted(composition(th[(t, 'site')]).items()))
        groups[key].append(t)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    comp_stats = {}
    for m in ('site', 'bond'):
        spreads, zmax = [], []
        for key, ts in multi.items():
            p = np.array([float(th[(t, m)]['pc']) for t in ts])
            s = np.array([float(th[(t, m)]['err']) for t in ts])
            i, j = int(p.argmax()), int(p.argmin())
            spreads.append(p[i] - p[j])
            zmax.append((p[i] - p[j]) / math.hypot(s[i], s[j]))
        spreads, zmax = np.array(spreads), np.array(zmax)
        comp_stats[m] = dict(groups=len(multi), tilings=sum(len(v) for v in multi.values()),
                             spread_median=float(np.median(spreads)), spread_p90=float(np.percentile(spreads, 90)),
                             spread_max=float(spreads.max()), frac_spread_gt_5sigma=float((zmax > 5).mean()))
    out['composition'] = comp_stats
    out['n_compositions'] = len(groups)

    # 5. linear mixing rule p_c ~ sum_t x_t beta_t fitted on k <= 4, tested on k >= 5
    types = sorted({ty for t in tags for ty in composition(th[(t, 'site')])})
    mix = {}
    for m in ('site', 'bond'):
        X, y, w, kk = [], [], [], []
        for t in tags:
            r = th[(t, m)]
            c = composition(r)
            X.append([float(c.get(ty, 0)) for ty in types])
            y.append(float(r['pc']))
            kk.append(int(r['k']))
        X, y, kk = np.array(X), np.array(y), np.array(kk)
        tr = kk <= 4
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        res = y - X @ beta
        beta_all, *_ = np.linalg.lstsq(X, y, rcond=None)
        res_all = y - X @ beta_all
        mix[m] = dict(train_rms=float(np.sqrt((res[tr] ** 2).mean())), test_rms=float(np.sqrt((res[~tr] ** 2).mean())),
                      test_maxabs=float(np.abs(res[~tr]).max()), allfit_rms=float(np.sqrt((res_all ** 2).mean())),
                      beta={ty: float(b) for ty, b in zip(types, beta_all)})
    out['mixing_rule'] = mix
    out['types'] = types
    json.dump(out, open('results/paper_stats.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
