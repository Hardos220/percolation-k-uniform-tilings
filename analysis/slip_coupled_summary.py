"""Slip differences from coupled runs (run/slip_coupled.py).

For every torus size: Delta = p(eclipsed) - p(staggered) from the merged histograms of all runs (root of
R_e + R_x = 1 for each member); its standard error from the scatter of the per-run differences,
sd(Delta_k)/sqrt(K).  Also the independent-sampling error sqrt(s_E^2 + s_S^2) of the merged estimate and the
variance-reduction factor (s_E^2 + s_S^2) / s_Delta^2.

usage: python analysis/slip_coupled_summary.py DIR [DIR ...]   -> printed table; with --json FILE also saved
"""
import collections
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import p_cross  # noqa: E402
from wrap import Hist  # noqa: E402


def summarize(d):
    groups = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(d, '*_E.txt')):
        m = re.match(r'(.+)__run(\d+)_E\.txt$', os.path.basename(f))
        groups[m.group(1)][int(m.group(2))] = (f, f[:-6] + '_S.txt')
    out = {}
    for key, runs in sorted(groups.items()):
        ks = sorted(runs)
        dk, ek, sk = [], [], []
        for k in ks:
            pe = p_cross(Hist([runs[k][0]]))[0]
            ps = p_cross(Hist([runs[k][1]]))[0]
            dk.append(pe - ps)
            ek.append(pe)
            sk.append(ps)
        # empirical reduction: per-run variances of the two members against that of their difference
        emp = (np.var(ek, ddof=1) + np.var(sk, ddof=1)) / np.var(dk, ddof=1) if len(ks) > 2 else None
        HE, HS = Hist([runs[k][0] for k in ks]), Hist([runs[k][1] for k in ks])
        (pe, se), (ps, ss) = p_cross(HE), p_cross(HS)
        dk = np.array(dk)
        sd = float(dk.std(ddof=1) / math.sqrt(len(dk))) if len(dk) > 1 else float('nan')
        out[key] = dict(runs=len(ks), samples=HE.M, N=HE.N, pE=pe, pS=ps, sE=se, sS=ss, delta=pe - ps, s_delta=sd,
                        s_indep=math.hypot(se, ss), reduction=(se * se + ss * ss) / sd ** 2 if sd > 0 else None,
                        mean_run_delta=float(dk.mean()), reduction_empirical=emp,
                        dir=d.replace(os.sep, '/'))
    return out


def main():
    args = sys.argv[1:]
    js = None
    if '--json' in args:
        i = args.index('--json')
        js = args[i + 1]
        args = args[:i] + args[i + 2:]
    res = {}
    for d in args:
        res.update(summarize(d))
    for k, v in res.items():
        print(f"{k:60s} K={v['runs']:3d} M={v['samples']:.2e}  Delta = {v['delta']*1e6:+9.3f}({v['s_delta']*1e6:.3f})e-6"
              f"   indep. {v['s_indep']*1e6:.3f}e-6  reduction x{(v['reduction'] or 0):.1f} (empirical x{(v['reduction_empirical'] or 0):.1f})   pE {v['pE']:.7f} pS {v['pS']:.7f}")
    if js:
        json.dump(res, open(js, 'w'), indent=1)


if __name__ == '__main__':
    main()
