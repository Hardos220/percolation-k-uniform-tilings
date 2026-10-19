"""Checks requested in the second review round.

(a) independent repeat of atlas thresholds for k = 8-10: the extension runs n1 of the atlas check alone
    versus the original atlas histograms (same torus, independent random streams);
(b) linear rule, Eq. (7): coefficients fitted on k <= 4 and on all tilings, with standard errors;
    occurrences of each vertex type in the training set; leave-one-out cross-validation on k <= 4;
    hold-out rms per k;
(c) bond slip family: weighted fit Delta(h) = A lambda^h, with a confidence interval for lambda;
(d) atlas check: summary of Delta/sigma by k group, aspect ratio and cell size.
usage: python analysis/review2_checks.py   -> results/review2_checks.json, paper/tab_beta.tex
"""
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tables import atlas_sources, read_member  # noqa: E402
from estimators import p_cross  # noqa: E402
from paper_stats import composition  # noqa: E402
from wrap import Hist  # noqa: E402

ARCH = {'3.12.12': (0.8079, 0.7404), '4.6.12': (0.7478, 0.6937), '4.8.8': (0.7297, 0.6768), '6.6.6': (0.6970, 0.6527),
        '3.6.3.6': (0.6527, 0.5244), '3.4.6.4': (0.6218, 0.5248), '4.4.4.4': (0.5927, 0.5000),
        '3.3.3.3.6': (0.5795, 0.4343), '3.3.4.3.4': (0.5508, 0.4141), '3.3.3.4.4': (0.5502, 0.4196),
        '3.3.3.3.3.3': (0.5000, 0.3473)}
TEX = {'3.3.3.3.3.3': '3^6', '3.3.3.3.6': '3^4.6', '3.3.3.4.4': '3^3.4^2', '3.3.4.3.4': '3^2.4.3.4',
       '3.3.4.12': '3^2.4.12', '3.3.6.6': '3^2.6^2', '3.4.3.12': '3.4.3.12', '3.4.4.6': '3.4^2.6',
       '3.4.6.4': '3.4.6.4', '3.6.3.6': '3.6.3.6', '3.12.12': '3.12^2', '4.4.4.4': '4^4', '4.6.12': '4.6.12',
       '4.8.8': '4.8^2', '6.6.6': '6^3'}


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = (r @ r) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), r


def main():
    out = {}
    # (a) independent repeat on the atlas torus
    sample = json.load(open('results/atlas_check_sample.json'))
    src = atlas_sources()
    zs = collections.defaultdict(list)
    for rec in sample:
        for mode in ('site', 'bond'):
            f1 = f"results/atlas_check/{rec['id']}__{mode}__{rec['torus1']}__n1.txt"
            texts = [t for t in (read_member(s, m) for s, m in src.get(f"{rec['id']}__{mode}__{rec['torus1']}.txt", [])) if t]
            pa, sa = p_cross(Hist(texts=texts))
            pn, sn = p_cross(Hist(files=[f1]))
            g = 'k5-7' if rec['k'] <= 7 else 'k8-10'
            zs[g].append((pn - pa) / math.hypot(sa, sn))
    out['repeat'] = {g: dict(n=len(v), mean=float(np.mean(v)), rms=float(np.sqrt(np.mean(np.square(v)))),
                             n_gt3=int(np.sum(np.abs(v) > 3))) for g, v in zs.items()}

    # (b) linear rule
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    th = {(r['tag'], r['mode']): r for r in rows}
    tags = sorted({r['tag'] for r in rows})
    k = np.array([int(th[(t, 'site')]['k']) for t in tags])
    comps = [composition(th[(t, 'site')]) for t in tags]
    types = sorted({ty for c in comps for ty in c}, key=lambda t: list(TEX).index(t))
    X = np.array([[float(c.get(ty, 0)) for ty in types] for c in comps])
    T = k <= 4
    occ_train = {ty: int(np.sum(X[T, i] > 0)) for i, ty in enumerate(types)}
    occ_all = {ty: int(np.sum(X[:, i] > 0)) for i, ty in enumerate(types)}
    lin = {}
    for m in ('site', 'bond'):
        y = np.array([float(th[(t, m)]['pc']) for t in tags])
        bT, sT, _ = ols(X[T], y[T])
        bA, sA, _ = ols(X, y)
        # leave-one-out on the training set
        Xt, yt = X[T], y[T]
        H = Xt @ np.linalg.pinv(Xt.T @ Xt) @ Xt.T
        rt = yt - Xt @ bT
        # leverage 1 (the single tiling with a (4.8^2) vertex): no leave-one-out prediction exists
        lev = np.diag(H)
        loo = rt[lev < 1 - 1e-9] / (1 - lev[lev < 1 - 1e-9])
        res = X @ bT - y
        per_k = {int(kk): float(np.sqrt(np.mean(res[k == kk] ** 2))) for kk in range(5, 11)}
        lin[m] = dict(beta_train=dict(zip(types, map(float, bT))), se_train=dict(zip(types, map(float, sT))),
                      beta_all=dict(zip(types, map(float, bA))), se_all=dict(zip(types, map(float, sA))),
                      train_rms=float(np.sqrt(np.mean(rt ** 2))), loo_rms=float(np.sqrt(np.mean(loo ** 2))),
                      holdout_rms=float(np.sqrt(np.mean(res[~T] ** 2))), holdout_rms_by_k=per_k)
    out['linear'] = lin
    out['occurrences_train'] = occ_train

    def cell(b, s):
        return f'{b:.4f}({round(s * 1e4):.0f})' if s < 9.5e-4 else f'{b:.3f}({round(s * 1e3):.0f})'
    lines = []
    for ty in types:
        a = ARCH.get(ty)
        c = [f'$({TEX[ty]})$', str(occ_train[ty]), str(occ_all[ty])]
        for mi, m in enumerate(('site', 'bond')):
            # a type that occurs in a single tiling has leverage 1: its coefficient is that threshold, without error
            c += [cell(lin[m]['beta_train'][ty], lin[m]['se_train'][ty]) if occ_train[ty] > 1
                  else f"{lin[m]['beta_train'][ty]:.4f}$^\\ast$",
                  cell(lin[m]['beta_all'][ty], lin[m]['se_all'][ty]) if occ_all[ty] > 1
                  else f"{lin[m]['beta_all'][ty]:.4f}$^\\ast$",
                  f'{a[mi]:.4f}' if a else '']
        lines.append(' & '.join(c) + r' \\')
    open('paper/tab_beta.tex', 'w').write('\n'.join(lines) + '\n')

    # (c) bond slip family: Delta(h) = A lambda^h, weighted least squares in (A, lambda)
    fam = json.load(open('results/slip_family.json'))
    h = np.array([f['h'] for f in fam], float)
    d = np.array([f['bond'][0] for f in fam])
    s = np.array([f['bond'][1] for f in fam])
    best = None
    lams = np.linspace(0.05, 0.3, 2501)
    chi = []
    for lam in lams:
        g = lam ** h
        A = np.sum(d * g / s ** 2) / np.sum(g * g / s ** 2)
        c2 = np.sum(((d - A * g) / s) ** 2)
        chi.append(c2)
        if best is None or c2 < best[0]:
            best = (c2, lam, A)
    chi = np.array(chi)
    ok = lams[chi <= best[0] + 1]
    out['slip_fit_all'] = dict(lam=float(best[1]), lo=float(ok.min()), hi=float(ok.max()), chi2=float(best[0]), dof=len(h) - 2)
    # fit without h = 1 (asymptotic regime)
    m2 = h >= 2
    best2 = None
    chi2 = []
    for lam in lams:
        g = lam ** h[m2]
        A = np.sum(d[m2] * g / s[m2] ** 2) / np.sum(g * g / s[m2] ** 2)
        c2 = np.sum(((d[m2] - A * g) / s[m2]) ** 2)
        chi2.append(c2)
        if best2 is None or c2 < best2[0]:
            best2 = (c2, lam)
    chi2 = np.array(chi2)
    ok2 = lams[chi2 <= best2[0] + 1]
    out['slip_fit_h2plus'] = dict(lam=float(best2[1]), lo=float(ok2.min()), hi=float(ok2.max()), chi2=float(best2[0]), dof=int(m2.sum()) - 2)

    # (d) atlas check summary by groups
    chk = json.load(open('results/atlas_check_summary.json'))['rows']
    groups = {'k5-7': lambda r: r['k'] <= 7, 'k8-10': lambda r: r['k'] >= 8,
              'aspect<0.5': lambda r: r['aspect1'] < 0.5, 'aspect>=0.5': lambda r: r['aspect1'] >= 0.5,
              'n_v>=100': lambda r: r['n_v'] >= 100, 'n_v<100': lambda r: r['n_v'] < 100}
    out['check_groups'] = {}
    for g, sel in groups.items():
        z = np.array([r['z'] for r in chk if sel(r)])
        out['check_groups'][g] = dict(n=len(z), mean=float(z.mean()), rms=float(np.sqrt(np.mean(z * z))),
                                      max_abs=float(np.abs(z).max()))
    json.dump(out, open('results/review2_checks.json', 'w'), indent=1)
    print(json.dumps({kk: v for kk, v in out.items() if kk != 'linear'}, indent=1))
    for m in ('site', 'bond'):
        L = lin[m]
        print(m, 'train rms %.4f LOO %.4f holdout %.4f' % (L['train_rms'], L['loo_rms'], L['holdout_rms']), L['holdout_rms_by_k'])


if __name__ == '__main__':
    main()
