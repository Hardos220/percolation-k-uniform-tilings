"""Consistency of the two independent atlas rounds (r0, r1) for the thresholds that have both.

For every threshold with histograms in both rounds, p_cross is computed separately per round and
z = (p_r0 - p_r1) / sqrt(s_r0^2 + s_r1^2) is collected. Workers: $RMQ_CPU.
usage: python analysis/atlas_rounds.py   -> results/atlas_rounds.json
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tables import atlas_sources, read_member  # noqa: E402
from estimators import p_cross  # noqa: E402
from wrap import Hist  # noqa: E402


def work(item):
    name, srcs = item
    out = []
    for s, m in srcs:
        t = read_member(s, m)
        if not t:
            return name, None
        H = Hist(texts=[t])
        out.append(p_cross(H) + (H.M,))
    return name, out


def main():
    items = [(n, s) for n, s in atlas_sources().items() if len(s) == 2]
    with mp.Pool(int(os.environ.get('RMQ_CPU', '1'))) as pool:
        res = [r for r in pool.imap_unordered(work, items, chunksize=20) if r[1]]
    z = np.array([(a[0] - b[0]) / np.hypot(a[1], b[1]) for _, (a, b) in res])
    by_k = {}
    for (name, _), zz in zip(res, z):
        by_k.setdefault(name[:2], []).append(zz)
    summary = dict(n=int(len(z)), mean=float(z.mean()), rms=float(np.sqrt((z * z).mean())),
                   frac_gt3=float((np.abs(z) > 3).mean()), maxabs=float(np.abs(z).max()),
                   by_k={k: dict(n=len(v), rms=float(np.sqrt(np.mean(np.square(v))))) for k, v in sorted(by_k.items())})
    json.dump(summary, open('results/atlas_rounds.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
