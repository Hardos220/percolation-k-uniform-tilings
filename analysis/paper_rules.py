"""Like-for-like comparison of empirical threshold rules (Table IV, Fig. 4) and invariant statistics.

Training set T: the 243 tilings with k <= 4 (used only by the rules with fitted parameters).
Test sets: H = all 23 153 tilings with k >= 5; A = the subset of H whose vertex types all occur in
Archimedean lattices (where the Archimedean average is defined).
Rules for p_c (site and bond):
  GM        Galam-Mauger (1996), published class-I parameters, q = mean degree           0 fitted
  GM-fit    same form, p0 and a refitted on T (least squares in log p)                    2 fitted
  Euler     Neher-Mecke-Wagner p* (parameter free)                                         0 fitted
  Euler+c   p* plus a constant offset fitted on T                                          1 fitted
  ArchAvg   composition-weighted average of the 11 Archimedean thresholds, Eq. (6)         0 fitted
  Linear    one coefficient per vertex type, fitted on T, Eq. (7)                          15 fitted
Site from bond (p_site predicted from the measured p_bond):
  Lebrecht  p_s = 0.16428 z^3/(z-1)^2 p_b                                                  0 fitted
  GM98      p_s = (delta/d)^{a_s} p_b^{a_s/a_b}, class I: a_s = 0.3670, delta = 1.3638, a_b = 0.6897, d = 2
usage: python analysis/paper_rules.py   -> results/paper_rules.json, paper/tab_rules.tex
"""
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from paper_stats import composition, euler_estimates, gm  # noqa: E402

ARCH = {'3.12.12': (0.8079007636, 0.7404207989), '4.6.12': (0.7478008, 0.6937331), '4.8.8': (0.7297232, 0.6768031),
        '6.6.6': (0.69704023, 0.6527036447), '3.6.3.6': (0.6527036447, 0.5244049992), '3.4.6.4': (0.62181207, 0.5248314616),
        '4.4.4.4': (0.5927460508, 0.5), '3.3.3.3.6': (0.579498, 0.4343283172), '3.3.4.3.4': (0.550806, 0.4141378566),
        '3.3.3.4.4': (0.550213, 0.4196403589), '3.3.3.3.3.3': (0.5, 0.3472963553)}


def main():
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    th = {(r['tag'], r['mode']): r for r in rows}
    tags = sorted({r['tag'] for r in rows})
    k = np.array([int(th[(t, 'site')]['k']) for t in tags])
    z = np.array([float(th[(t, 'site')]['mean_degree']) for t in tags])
    comps = [composition(th[(t, 'site')]) for t in tags]
    types = sorted({ty for c in comps for ty in c})
    X = np.array([[float(c.get(ty, 0)) for ty in types] for c in comps])
    T = k <= 4
    H = k >= 5
    arch_ok = np.array([all(ty in ARCH for ty in c) and len(c) > 1 for c in comps])
    A = H & arch_ok
    pc = {m: np.array([float(th[(t, m)]['pc']) for t in tags]) for m in ('site', 'bond')}
    est = {}
    for mi, m in enumerate(('site', 'bond')):
        y = pc[m]
        e = {}
        e['GM'] = gm(m, z)
        # refit p0, a on T:  log p = log p0 - a log x,  x = (z-1) (site) or (z-1)/2 (bond)
        x = (z - 1) if m == 'site' else (z - 1) / 2
        B = np.column_stack([np.ones(T.sum()), -np.log(x[T])])
        (lp0, a), *_ = np.linalg.lstsq(B, np.log(y[T]), rcond=None)
        e['GM-fit'] = math.exp(lp0) * x ** (-a)
        ps = np.array([np.nan if v is None else v for v in (euler_estimates(th[(t, m)])[1] for t in tags)])
        e['Euler'] = ps
        e['Euler+c'] = ps + np.nanmean((y - ps)[T])
        e['ArchAvg'] = np.array([sum(float(xx) * ARCH[ty][mi] for ty, xx in c.items()) if ok else np.nan
                                 for c, ok in zip(comps, arch_ok)])
        beta, *_ = np.linalg.lstsq(X[T], y[T], rcond=None)
        e['Linear'] = X @ beta
        e['Linear-all'] = X @ np.linalg.lstsq(X, y, rcond=None)[0]  # in-sample variant (not used in Table IV)
        est[m] = e
    # site from bond
    pb, psite = pc['bond'], pc['site']
    est['site']['Lebrecht'] = 0.16428 * z ** 3 / (z - 1) ** 2 * pb
    a_s, delta, a_b = 0.3670, 1.3638, 0.6897
    est['site']['GM98'] = (delta / 2) ** a_s * pb ** (a_s / a_b)

    def stat(v, y, mask):
        r = (v - y)[mask]
        r = r[np.isfinite(r)]
        return dict(n=int(len(r)), mean=float(r.mean()), rms=float(np.sqrt((r * r).mean())))

    out = {'n_T': int(T.sum()), 'n_H': int(H.sum()), 'n_A': int(A.sum()), 'rules': {}}
    for m in ('site', 'bond'):
        for name, v in est[m].items():
            out['rules'][f'{m}:{name}'] = dict(H=stat(v, pc[m], H), A=stat(v, pc[m], A))
    # invariants
    phi = np.array([float(th[(t, 'site')]['pc']) * float(th[(t, 'site')]['sites_per_area']) * math.pi / 4 for t in tags])
    big = np.array([any(ty in ('3.12.12', '4.6.12', '4.8.8', '3.4.3.12', '3.3.4.12') for ty in c) for c in comps])
    zb = z * pb
    out['scher_zallen'] = dict(mean=float(phi.mean()), sd=float(phi.std()), min=float(phi.min()), max=float(phi.max()),
                               p05=float(np.percentile(phi, 5)), p95=float(np.percentile(phi, 95)),
                               mean_no_large=float(phi[~big].mean()), sd_no_large=float(phi[~big].std()),
                               n_large=int(big.sum()), frac_below_043=float((phi < 0.43).mean()),
                               frac_below_043_with_large=float((big & (phi < 0.43)).sum() / max((phi < 0.43).sum(), 1)),
                               min_tag=tags[int(phi.argmin())])
    out['vyssotsky'] = dict(mean=float(zb.mean()), sd=float(zb.std()), rel_sd=float(zb.std() / zb.mean()),
                            rel_sd_phi=float(phi.std() / phi.mean()))
    # composition groups: spreads where resolved
    groups = collections.defaultdict(list)
    for i, c in enumerate(comps):
        groups[tuple(sorted(c.items()))].append(i)
    multi = [g for g in groups.values() if len(g) >= 2]
    gs = {}
    for m in ('site', 'bond'):
        err = np.array([float(th[(t, m)]['err']) for t in tags])
        sp, zs = [], []
        for g in multi:
            p = pc[m][g]
            i, j = g[int(p.argmax())], g[int(p.argmin())]
            sp.append(pc[m][i] - pc[m][j])
            zs.append((pc[m][i] - pc[m][j]) / math.hypot(err[i], err[j]))
        sp, zs = np.array(sp), np.array(zs)
        res = zs > 5
        gs[m] = dict(groups=len(multi), median_spread_sigma=float(np.median(zs)), frac_resolved=float(res.mean()),
                     median_spread_resolved=float(np.median(sp[res])), max_spread=float(sp.max()))
    out['composition_groups'] = gs
    json.dump(out, open('results/paper_rules.json', 'w'), indent=1)
    np.save('results/paper_rules_resid.npy', {m: {n: (v - pc[m]) for n, v in est[m].items()} for m in est}, allow_pickle=True)

    # LaTeX rows for Table IV
    def f(x):
        return f'{x:.4f}' if abs(x) < 0.01 else f'{x:.3f}'
    lab = {'GM': ('Galam--Mauger~\\cite{GalamMauger96}, $q=\\zb$', '0'),
           'GM-fit': ('Galam--Mauger form, refitted', '2'),
           'Euler': ('Euler estimate $p^\\ast$~\\cite{Neher08}', '0'),
           'Euler+c': ('$p^\\ast$ + constant, refitted', '1'),
           'ArchAvg': ('Archimedean average, Eq.~\\eqref{eq:avg}', '0'),
           'Linear': ('linear rule, Eq.~\\eqref{eq:mix}', '15')}
    lines = []
    for name, (txt, npar) in lab.items():
        cells = [txt, npar]
        for m in ('site', 'bond'):
            r = out['rules'][f'{m}:{name}']
            cells.append(f(r['A']['rms']))
            cells.append('---' if name == 'ArchAvg' else f(r['H']['rms']))
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\hline')
    for name, txt in (('Lebrecht', 'Lebrecht \\textit{et al.}~\\cite{Lebrecht21}, $\\ps$ from $\\pb$'),
                      ('GM98', 'Galam--Mauger~\\cite{GalamMauger98}, $\\ps$ from $\\pb$')):
        r = out['rules'][f'site:{name}']
        lines.append(f"{txt} & 0 & {f(r['A']['rms'])} & {f(r['H']['rms'])} & & \\\\")
    open('paper/tab_rules.tex', 'w').write('\n'.join(lines) + '\n')
    print(json.dumps({kk: v for kk, v in out.items() if kk != 'rules'}, indent=1))
    for kk, v in out['rules'].items():
        print(f"{kk:18s} A rms {v['A']['rms']:.4f} (mean {v['A']['mean']:+.4f}, n={v['A']['n']})   "
              f"H rms {v['H']['rms']:.4f} (mean {v['H']['mean']:+.4f}, n={v['H']['n']})")


if __name__ == '__main__':
    main()
