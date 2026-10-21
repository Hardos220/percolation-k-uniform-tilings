"""Checks requested in the fourth review round.

(1) dedicated (3^2.4.3.4) site run: dependence of the extrapolation on the correction exponent;
(2) chi^2 of the Gu-Ziff comparison if our errors were smaller by a factor 0.88;
(3) linear rule: standard errors of the coefficients from a bootstrap over composition groups;
(4) composition groups: chi^2 homogeneity test per group, Benjamini-Hochberg at 1% false discovery rate;
(5) noise ceiling: statistical share of the within-composition variance;
(6) deviations of Neher et al. versus isotropy of the lattice;
(7) stackings of the h = 1 strips (composition 4/5 (3.4^2.6) + 1/5 (3.6.3.6)): fraction of eclipsed strip
    boundaries per tiling and linearity of p_c in it.
usage: python analysis/review4_checks.py   -> results/review4_checks.json
"""
import collections
import csv
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.stats import chi2 as chi2dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_stats import composition  # noqa: E402


def fit_exp(sizes, om):
    N = np.array([s['sites'] for s in sizes], float)
    p = np.array([s['p'] for s in sizes])
    s = np.array([s['s'] for s in sizes])
    X = np.column_stack([np.ones_like(N), N ** (-om)])
    W = np.diag(1 / s ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    b = cov @ X.T @ W @ p
    return float(b[0]), float(math.sqrt(cov[0, 0]))


def canon(c):
    c = list(c)
    return min(tuple(x[i:] + x[:i]) for x in (c, c[::-1]) for i in range(len(c)))


def main():
    out = {}
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    th = {(r['tag'], r['mode']): r for r in rows}
    tags = sorted({r['tag'] for r in rows})
    # (1)
    core = json.load(open('results/core/report_r0.json'))
    rec = next(r for r in core if r['tag'] == '01__eu_5e_1' and r['mode'] == 'site')
    out['snub_exponent'] = {f'{2 * om:.2f}': fit_exp(rec['sizes'], om) for om in (1.0, 1.375, 1.5, 2.0, 2.5)}
    # (2)
    gz = json.load(open('reference/two_uniform_bond_guziff.json'))['lattices']
    for f in (1.0, 0.88):
        z2 = [(float(th[(x['tag'], 'bond')]['pc']) - x['bond']) ** 2 /
              ((f * float(th[(x['tag'], 'bond')]['err'])) ** 2 + x['err'] ** 2) for x in gz]
        pcs = json.load(open('results/periodic_checks.json'))['per_tiling']
        an = [z for z, x in zip(z2, gz) if not any(q in pcs[x['tag']]['rot'] for q in (3, 4, 6))]
        out[f'guziff_chi2_scale_{f}'] = dict(all=float(sum(z2)), aniso=float(sum(an)), n_aniso=len(an))
    # (3) group bootstrap of the linear rule
    comps = [composition(th[(t, 'site')]) for t in tags]
    types = sorted({ty for c in comps for ty in c})
    X = np.array([[float(c.get(ty, 0)) for ty in types] for c in comps])
    k = np.array([int(th[(t, 'site')]['k']) for t in tags])
    gkey = [tuple(sorted(c.items())) for c in comps]
    rng = np.random.default_rng(2026)
    boot = {}
    for m in ('site', 'bond'):
        y = np.array([float(th[(t, m)]['pc']) for t in tags])
        for name, mask in (('train', k <= 4), ('all', np.ones(len(tags), bool))):
            idx = np.where(mask)[0]
            groups = collections.defaultdict(list)
            for i in idx:
                groups[gkey[i]].append(i)
            gl = list(groups.values())
            bs = []
            for _ in range(400):
                pick = rng.integers(0, len(gl), len(gl))
                ii = np.concatenate([gl[j] for j in pick])
                b = np.linalg.lstsq(X[ii], y[ii], rcond=None)[0]
                bs.append(b)
            bs = np.array(bs)
            # coefficients of types absent from a resample are 0 in lstsq: use only resamples containing the type
            se = {}
            for j, ty in enumerate(types):
                ok = [b[j] for b, pk in zip(bs, range(len(bs)))]
                se[ty] = float(np.std(bs[:, j]))
            boot[f'{m}:{name}'] = se
    out['beta_group_bootstrap_se'] = boot
    # (4) chi^2 homogeneity of composition groups with BH-FDR
    groups = collections.defaultdict(list)
    for t, c in zip(tags, comps):
        groups[tuple(sorted(c.items()))].append(t)
    multi = [g for g in groups.values() if len(g) >= 2]
    hom = {}
    for m in ('site', 'bond'):
        pv = []
        for g in multi:
            p = np.array([float(th[(t, m)]['pc']) for t in g])
            s = np.array([float(th[(t, m)]['err']) for t in g])
            w = 1 / s ** 2
            mean = np.sum(w * p) / w.sum()
            c2 = float(np.sum(w * (p - mean) ** 2))
            pv.append(chi2dist.sf(c2, len(g) - 1))
        pv = np.array(pv)
        order = np.argsort(pv)
        n = len(pv)
        thr = 0.01 * np.arange(1, n + 1) / n
        passed = pv[order] <= thr
        nsig = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
        hom[m] = dict(groups=n, significant=nsig, frac=nsig / n)
    out['group_homogeneity_bh01'] = hom
    # (5) noise ceiling
    ceil = {}
    for m in ('site', 'bond'):
        num, den = 0.0, 0.0
        for g in multi:
            p = np.array([float(th[(t, m)]['pc']) for t in g])
            s = np.array([float(th[(t, m)]['err']) for t in g])
            n = len(g)
            num += np.sum(s ** 2) * (1 - 1 / n)
            den += np.sum((p - p.mean()) ** 2)
        ceil[m] = dict(statistical_share=float(num / den))
    out['noise_ceiling'] = ceil
    # (6) Neher deviations vs isotropy
    nn = json.load(open('results/paper_numbers.json'))['neher_site']['entries']
    pcs = json.load(open('results/periodic_checks.json'))['per_tiling']
    iso = np.array([any(q in pcs[e['tag']]['rot'] for q in (3, 4, 6)) for e in nn])
    dv = np.array([e['neher'] - e['ours'] for e in nn])
    x = np.array([e['ours'] for e in nn])
    A = np.vstack([np.ones_like(x), x]).T
    res = dv - A @ np.linalg.lstsq(A, dv, rcond=None)[0]
    out['neher_isotropy'] = dict(n_iso=int(iso.sum()), n_aniso=int((~iso).sum()),
                                 mean_dev_iso=float(dv[iso].mean()), mean_dev_aniso=float(dv[~iso].mean()),
                                 mean_resid_iso=float(res[iso].mean()), mean_resid_aniso=float(res[~iso].mean()))
    # (7) stackings of the h = 1 strips
    target = {'3.4.4.6': Fraction(4, 5), '3.6.3.6': Fraction(1, 5)}
    stack = []
    for t, c in zip(tags, comps):
        if c != target:
            continue
        d = json.load(open(f'data/cells/{t}.json'))
        T1, T2 = np.array(d['T1']), np.array(d['T2'])
        P = np.array(d['pos'])
        centers = [i for i, cf in enumerate(d['vertex_configs']) if canon(cf) == canon([3, 6, 3, 6])]
        # strip direction: shortest lattice vector of length 2
        u = None
        for a in range(-3, 4):
            for b in range(-3, 4):
                v = a * T1 + b * T2
                if abs(np.hypot(*v) - 2) < 1e-6:
                    u = v / 2
                    break
            if u is not None:
                break
        nvec = np.array([-u[1], u[0]])
        pts = []
        for i in centers:
            for a in range(-8, 9):
                for b in range(-8, 9):
                    pts.append(P[i] + a * T1 + b * T2)
        pts = np.array(pts)
        across = np.round(pts @ nvec, 5)
        along = pts @ u
        levels = np.unique(across)
        mid = len(levels) // 2
        # strips around the middle; offsets of consecutive strips, mod 2
        offs = []
        for j in range(mid - 6, mid + 6):
            a0 = along[across == levels[j]]
            a1 = along[across == levels[j + 1]]
            o = (a1[0] - a0[0]) % 2.0
            offs.append(min(o, 2 - o))
        # one period of strips along the stacking direction
        cell_area = abs(T1[0] * T2[1] - T1[1] * T2[0])
        period = int(round(cell_area / 2 / (levels[1] - levels[0]))) if len(levels) > 1 else 1
        per = offs[:period]
        fe = sum(1 for o in per if o < 0.5) / len(per)
        stack.append(dict(tag=t, k=int(th[(t, 'site')]['k']), strips_per_period=period, eclipsed_fraction=fe,
                          sequence=''.join('E' if o < 0.5 else 'S' for o in per),
                          site=(float(th[(t, 'site')]['pc']), float(th[(t, 'site')]['err'])),
                          bond=(float(th[(t, 'bond')]['pc']), float(th[(t, 'bond')]['err']))))
    lin = {}
    for m in ('site', 'bond'):
        f = np.array([s['eclipsed_fraction'] for s in stack])
        p = np.array([s[m][0] for s in stack])
        e = np.array([s[m][1] for s in stack])
        Xl = np.column_stack([np.ones_like(f), f])
        W = np.diag(1 / e ** 2)
        cov = np.linalg.inv(Xl.T @ W @ Xl)
        b = cov @ Xl.T @ W @ p
        r = (p - Xl @ b) / e
        lin[m] = dict(a=float(b[0]), b=float(b[1]), chi2=float(np.sum(r * r)), dof=len(p) - 2,
                      max_abs_z=float(np.abs(r).max()))
    out['h1_stackings'] = dict(tilings=stack, linear_fit=lin)
    json.dump(out, open('results/review4_checks.json', 'w'), indent=1)
    print(json.dumps({kk: v for kk, v in out.items() if kk not in ('beta_group_bootstrap_se', 'h1_stackings')}, indent=1))
    for s in stack:
        print(f"{s['tag']:38s} k={s['k']:2d} per={s['strips_per_period']:2d} {s['sequence']:12s} fE={s['eclipsed_fraction']:.3f} "
              f"site {s['site'][0]:.6f}({s['site'][1]*1e6:.0f}) bond {s['bond'][0]:.6f}({s['bond'][1]*1e6:.0f})")
    print(json.dumps(lin, indent=1))


if __name__ == '__main__':
    main()
