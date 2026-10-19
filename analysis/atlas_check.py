"""Analysis of the atlas finite-size check (run/make_atlas_check.py, results/atlas_check/).

For each sampled tiling and mode: p1 from the atlas torus (atlas histograms of both rounds plus the new
n1 run), p2 from the near-square torus with ~4x the sites (n2). Delta = p2 - p1 estimates minus the
finite-size bias of the atlas value (exactly so if the bias of p2 is negligible).
usage: python analysis/atlas_check.py   -> results/atlas_check_summary.json
"""
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tables import atlas_sources, read_member  # noqa: E402
from estimators import p_cross  # noqa: E402
from wrap import Hist  # noqa: E402


def main():
    sample = json.load(open('results/atlas_check_sample.json'))
    src = atlas_sources()
    rows = []
    for rec in sample:
        for mode in ('site', 'bond'):
            f1 = f"results/atlas_check/{rec['id']}__{mode}__{rec['torus1']}__n1.txt"
            f2 = f"results/atlas_check/{rec['id']}__{mode}__{rec['torus2']}__n2.txt"
            if not (os.path.exists(f1) and os.path.exists(f2)):
                continue
            name = f"{rec['id']}__{mode}__{rec['torus1']}.txt"
            texts = [t for t in (read_member(s, m) for s, m in src.get(name, [])) if t]
            Ha = Hist(texts=texts)
            pa, sa = p_cross(Ha)
            H1 = Hist(files=[f1], texts=texts)
            p1, s1 = p_cross(H1)
            H2 = Hist(files=[f2])
            p2, s2 = p_cross(H2)
            d = p2 - p1
            sd = math.hypot(s1, s2)
            rows.append(dict(id=rec['id'], stratum=rec['stratum'], k=rec['k'], mode=mode, n_v=rec['n_v'],
                             aspect1=rec['aspect1'], N1=H1.N, N2=H2.N, p_atlas=pa, s_atlas=sa, p1=p1, s1=s1,
                             p2=p2, s2=s2, d=d, sd=sd, z=d / sd, d_over_sa=d / sa))
    out = {'n': len(rows), 'rows': rows, 'strata': {}}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['stratum']].append(r)
        groups[f"{r['stratum']}:{r['mode']}"].append(r)
        groups['all'].append(r)
    for g, rs in sorted(groups.items()):
        z = np.array([r['z'] for r in rs])
        q = np.array([r['d_over_sa'] for r in rs])
        out['strata'][g] = dict(n=len(rs), mean_z=float(z.mean()), rms_z=float(np.sqrt((z * z).mean())),
                                n_absz_gt3=int((np.abs(z) > 3).sum()), mean_d_over_sigma_atlas=float(q.mean()),
                                max_abs_d_over_sigma_atlas=float(np.abs(q).max()),
                                frac_abs_d_gt_sigma_atlas=float((np.abs(q) > 1).mean()),
                                mean_d=float(np.mean([r['d'] for r in rs])),
                                max_abs_d=float(np.max([abs(r['d']) for r in rs])))
    json.dump(out, open('results/atlas_check_summary.json', 'w'), indent=1)
    for g, v in out['strata'].items():
        print(f"{g:8s} n={v['n']:3d}  mean z {v['mean_z']:+.2f}  rms z {v['rms_z']:.2f}  |z|>3: {v['n_absz_gt3']}  "
              f"mean d/sa {v['mean_d_over_sigma_atlas']:+.2f}  max |d|/sa {v['max_abs_d_over_sigma_atlas']:.2f}  "
              f"max |d| {v['max_abs_d']:.1e}")
    worst = sorted(rows, key=lambda r: -abs(r['z']))[:8]
    for r in worst:
        print(f"  {r['id']:40s} {r['mode']} {r['stratum']} N1={r['N1']} N2={r['N2']} d={r['d']:+.2e} "
              f"z={r['z']:+.1f} d/sa={r['d_over_sa']:+.1f}")


if __name__ == '__main__':
    main()
