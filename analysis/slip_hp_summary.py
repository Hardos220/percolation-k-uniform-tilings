"""Summarize the high-precision slip-pair runs (results/slip_hp) into results/slip_hp_summary.json.

Histograms of all jobs of one lattice (same torus) are added; p_L is the cross estimator.
usage: python analysis/slip_hp_summary.py
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from estimators import p_cross  # noqa: E402
from wrap import Hist  # noqa: E402

PAIRS = [  # (h, tag_a, tag_b, torus), as in run/make_slip_hp_jobs.py
    (2, '03__3__eu_4f_4k_4r_1', '03__3__eu_4f_4k_4s_1', '24x12'),
    (3, '03__3__eu_4f_4k_4u_1', '03__3__eu_4f_4k_4u_2', '25x10'),
]


def main():
    out = {}
    for h, a, b, torus in PAIRS:
        res = {'a': a, 'b': b, 'torus': torus}
        jobs, samples, ps, ss = [], [], [], []
        for tag in (a, b):
            files = sorted(glob.glob(f'results/slip_hp/{tag}__site__{torus}__hp*.txt'))
            H = Hist(files)
            p, s = p_cross(H)
            jobs.append(len(files))
            samples.append(H.M)
            ps.append(p)
            ss.append(s)
        res.update(pa=ps[0], sa=ss[0], pb=ps[1], sb=ss[1], d=ps[1] - ps[0], sd=math.hypot(*ss),
                   jobs=jobs, samples=samples)
        out[f'h={h}'] = res
        print(f"h={h}: d = {res['d']*1e6:+.2f} +- {res['sd']*1e6:.2f} (1e-6), jobs {jobs}")
    json.dump(out, open('results/slip_hp_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
