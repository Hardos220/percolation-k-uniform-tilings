"""Checks requested in the third review round.

(1) Neher et al. 2-uniform site values: deviation trend with p_c; residual scatter; lattice 41 after the trend.
(2) Lattice 41 (1/2(3.12^2)+1/2(3.4.3.12), fourfold rotation, square tori): independent estimate from
    R_e(p) = R_e^inf(tau = i) = 0.690473725 on the three core tori, extrapolated with N^-11/8.
(3) Linear rule: leave-one-out on k <= 4 without the leverage-one point (4.8^2); group-equal-weight hold-out rms;
    low-rank baseline p_c = a + b zbar + sum_n c_n f_n fitted on k <= 4; logit-linear rule.
(4) Core extrapolation versus the weighted mean of the two larger tori.
(5) Square-lattice site finite-size correction exponent from the FSS study.
(6) Consistency: p_b(3^6) + p_b(6^3) = 1; Hammersley inequality p_bond < p_site and the Grimmett-Stacey bound
    p_site <= 1 - (1 - p_bond)^(Delta - 1) for all tilings.
(7) 2-uniform bond thresholds versus the four-decimal values listed by Lebrecht et al. (2021), Table 2.
usage: python analysis/review3_checks.py   -> results/review3_checks.json
"""
import collections
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_stats import composition, local_counts  # noqa: E402
from wrap import Hist  # noqa: E402

LEBRECHT = [  # Lebrecht, Centres, Ramirez-Pastor, Physica A 569, 125802 (2021), Table 2: (site, bond), 2-uniform rows
    (0.7680, 0.6749), (0.7157, 0.6454), (0.6808, 0.5578), (0.6499, 0.5363), (0.6329, 0.5171), (0.6286, 0.5189),
    (0.6279, 0.5177), (0.6221, 0.5197), (0.6171, 0.4892), (0.5885, 0.4723), (0.5883, 0.4657), (0.5720, 0.4584),
    (0.5648, 0.4453), (0.5607, 0.4111), (0.5505, 0.4163), (0.5504, 0.4155), (0.5440, 0.4038), (0.5407, 0.3891),
    (0.5342, 0.3949), (0.5258, 0.3829)]
PAIR = {0.6286: '02__2__eu_4f_4k_2', 0.6279: '02__2__eu_4f_4k_1'}  # the 16/17 pair, ordered as in Table II


def main():
    out = {}
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    th = {(r['tag'], r['mode']): r for r in rows}
    # (1) Neher trend
    nn = json.load(open('results/paper_numbers.json'))['neher_site']['entries']
    x = np.array([e['ours'] for e in nn])
    dv = np.array([e['neher'] - e['ours'] for e in nn])  # Neher minus ours
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, dv, rcond=None)
    res = dv - A @ coef
    i41 = [e['tag'] for e in nn].index('02__2__eu_3a_4b_1')
    # residual of lattice 41 from a fit without it
    m = np.arange(len(x)) != i41
    c2, *_ = np.linalg.lstsq(A[m], dv[m], rcond=None)
    r41 = dv[i41] - (c2[0] + c2[1] * x[i41])
    out['neher_trend'] = dict(corr=float(np.corrcoef(x, dv)[0, 1]), slope=float(coef[1]), intercept=float(coef[0]),
                              resid_rms=float(np.sqrt(np.mean(res ** 2))), resid_rms_without41=float(np.sqrt(np.mean((dv[m] - A[m] @ c2) ** 2))),
                              lattice41_dev=float(dv[i41]), lattice41_resid_vs_trend=float(r41),
                              lattice41_resid_in_quoted_sigma=float(r41 / 2e-4))
    # (2) lattice 41 with the R_e criterion on square tori
    rinf = 0.690473725
    pts = []
    for f in sorted(glob.glob('results/core/**/02__2__eu_3a_4b_1__site__*__r0.txt', recursive=True)):
        H = Hist([f])
        p, s = H.p_at(rinf, 'e')
        pts.append((H.N, p, s))
    pts.sort()
    Ns = np.array([q[0] for q in pts], float)
    ps = np.array([q[1] for q in pts])
    ss = np.array([q[2] for q in pts])
    X = np.column_stack([np.ones_like(Ns), Ns ** (-1.375)])
    W = np.diag(1 / ss ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    b = cov @ X.T @ W @ ps
    out['lattice41_Re'] = dict(points=[dict(N=int(n), p=float(p), s=float(s)) for n, p, s in pts], pc=float(b[0]),
                               err=float(math.sqrt(cov[0, 0])), chi2=float(np.sum(((ps - X @ b) / ss) ** 2)))
    # (3) linear rule variants
    tags = sorted({r['tag'] for r in rows})
    k = np.array([int(th[(t, 'site')]['k']) for t in tags])
    comps = [composition(th[(t, 'site')]) for t in tags]
    types = sorted({ty for c in comps for ty in c})
    Xc = np.array([[float(c.get(ty, 0)) for ty in types] for c in comps])
    T = k <= 4
    i488 = types.index('4.8.8')
    keep = T & (Xc[:, i488] == 0)
    gkey = [tuple(sorted(c.items())) for c in comps]
    lin = {}
    for mname in ('site', 'bond'):
        y = np.array([float(th[(t, mname)]['pc']) for t in tags])
        cols = [j for j in range(len(types)) if j != i488]
        Xt, yt = Xc[keep][:, cols], y[keep]
        H = Xt @ np.linalg.pinv(Xt.T @ Xt) @ Xt.T
        bt = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        loo = (yt - Xt @ bt) / (1 - np.diag(H))
        bT = np.linalg.lstsq(Xc[T], y[T], rcond=None)[0]
        res = Xc @ bT - y
        # group-equal-weight rms on the hold-out set
        g = collections.defaultdict(list)
        for gi, r_, kk in zip(gkey, res, k):
            if kk >= 5:
                g[gi].append(r_ * r_)
        gw = math.sqrt(np.mean([np.mean(v) for v in g.values()]))
        # low-rank baseline: a + b zbar + sum_n c_n f_n  (n = 3, 4, 6, 8, 12)
        feats = []
        for c in comps:
            z, faces, deg = local_counts(c)
            feats.append([1.0, z] + [faces.get(n, 0.0) for n in (3, 4, 6, 8, 12)])
        F = np.array(feats)
        bf = np.linalg.lstsq(F[T], y[T], rcond=None)[0]
        rf = F @ bf - y
        # logit-linear rule
        lg = np.log(y / (1 - y))
        bl = np.linalg.lstsq(Xc[T], lg[T], rcond=None)[0]
        pl = 1 / (1 + np.exp(-(Xc @ bl)))
        rl = pl - y
        lin[mname] = dict(loo_without_488=float(np.sqrt(np.mean(loo ** 2))), holdout_rms=float(np.sqrt(np.mean(res[~T] ** 2))),
                          holdout_group_equal_rms=float(gw), lowrank_params=int(F.shape[1]),
                          lowrank_holdout_rms=float(np.sqrt(np.mean(rf[~T] ** 2))),
                          logit_holdout_rms=float(np.sqrt(np.mean(rl[~T] ** 2))))
    out['linear'] = lin
    # (4) core extrapolation vs weighted mean of the two larger tori
    core = json.load(open('results/core/report_r0.json'))
    dz = []
    for r_ in core:
        s2 = r_['sizes'][1:]
        w = np.array([1 / q['s'] ** 2 for q in s2])
        pm = float(np.sum(w * np.array([q['p'] for q in s2])) / w.sum())
        dz.append((r_['pc'] - pm) / r_['spc'])
    dz = np.abs(np.array(dz))
    out['core_vs_two_largest'] = dict(median_abs_diff_sigma=float(np.median(dz)), p90=float(np.percentile(dz, 90)),
                                      max=float(dz.max()))
    # (5) square-lattice site correction exponent
    fss = json.load(open('results/fss_summary.json'))['01__eu_4n_1_site']
    N = np.array([q['N'] for q in fss[:3]], float)
    D = np.array([q['dev'] for q in fss[:3]])
    slope = np.polyfit(np.log(N), np.log(-D), 1)[0]
    out['square_site_fss'] = dict(exponent_in_N=float(-slope), dev_at_1000=float(-np.exp(np.polyval(np.polyfit(np.log(N), np.log(-D), 1), np.log(1000.0)))))
    # (6) consistency checks
    pb3, sb3 = float(th[('01__eu_6a_1', 'bond')]['pc']), float(th[('01__eu_6a_1', 'bond')]['err'])
    hc = [t for t in tags if th[(t, 'bond')]['vertex_types'] == '6.6.6:1'][0]
    pb6, sb6 = float(th[(hc, 'bond')]['pc']), float(th[(hc, 'bond')]['err'])
    viol = [t for t in tags if float(th[(t, 'bond')]['pc']) >= float(th[(t, 'site')]['pc'])]
    # Grimmett-Stacey: p_site <= 1 - (1 - p_bond)^(Delta - 1), Delta = maximal degree
    gs = []
    for t in tags:
        dmax = max(len(c) for c in json.load(open(f'data/cells/{t}.json'))['vertex_configs'])
        gs.append(1 - (1 - float(th[(t, 'bond')]['pc'])) ** (dmax - 1) - float(th[(t, 'site')]['pc']))
    out['consistency'] = dict(sum_tri_hex_bond=pb3 + pb6, err=math.hypot(sb3, sb6), hammersley_violations=len(viol),
                              min_site_minus_bond=float(min(float(th[(t, 'site')]['pc']) - float(th[(t, 'bond')]['pc']) for t in tags)),
                              grimmett_stacey_violations=int(sum(g < 0 for g in gs)), grimmett_stacey_min_margin=float(min(gs)))
    # (7) Lebrecht's four-decimal bond values; rows matched through their (Neher) site values
    by_site = {e['neher']: e['tag'] for e in nn}
    by_site.update(PAIR)
    dl = {}
    for ps_, pb_ in LEBRECHT:
        t = by_site[ps_]
        dl[t] = dict(site_ours=float(th[(t, 'site')]['pc']), bond_ours=float(th[(t, 'bond')]['pc']), bond_lebrecht=pb_,
                     diff=float(th[(t, 'bond')]['pc']) - pb_)
    d = np.array([v['diff'] for v in dl.values()])
    out['lebrecht_bond'] = dict(n=len(dl), max_abs_diff=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d * d))),
                                n_within_rounding=int(np.sum(np.abs(d) <= 5e-5 + 1e-9)), rows=dl)
    json.dump(out, open('results/review3_checks.json', 'w'), indent=1)
    print(json.dumps({kk: v for kk, v in out.items() if kk != 'lebrecht_bond'}, indent=1))
    lb = out['lebrecht_bond']
    print('lebrecht n %d  max|diff| %.2e  rms %.2e  within rounding %d' % (lb['n'], lb['max_abs_diff'], lb['rms'], lb['n_within_rounding']))
    for t, v in lb['rows'].items():
        print('  %-22s %.7f  %.4f  %+.1e' % (t, v['bond_ours'], v['bond_lebrecht'], v['diff']))


if __name__ == '__main__':
    main()
