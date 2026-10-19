"""Finite-size study of the cross estimator on lattices with known thresholds (CPU, nz.exe).

Small tori, N ~ 64 ... 1024 sites, sigma ~ SIGMA per point; CPU seconds from the cost model
sigma*sqrt(T) = 5 x 1.65e-5 (N/256)^0.24 per CPU core (calibrated in a pilot run).
Longest jobs first (better packing on many workers).

usage: python run/make_fss_jobs.py  ->  run/jobs_fss.txt
"""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_jobs import shape  # noqa: E402

CASES = [('01__eu_4n_1', 'site'), ('01__eu_4n_1', 'bond'), ('01__eu_3d_1', 'site'), ('01__eu_4f_1', 'bond'),
         ('01__eu_3a_1', 'site'), ('01__eu_3a_1', 'bond'), ('01__eu_6a_1', 'bond'), ('01__eu_488', 'bond')]
TARGETS = [64, 128, 256, 512, 1024]
SIGMA = 2e-6


def main():
    jobs = []
    for tag, mode in CASES:
        d = json.load(open(f'data/cells/{tag}.json'))
        q, p = shape(d)
        used = set()
        for nt in TARGETS:
            L = max(3, round(math.sqrt(nt / (q * p * d['nv']))))
            while L in used:
                L += 1
            used.add(L)
            Lx, Ly = q * L, p * L
            n = Lx * Ly * d['nv']
            secs = round((5 * 1.65e-5 * (n / 256) ** 0.24 / SIGMA) ** 2)
            name = f'{tag}__{mode}__{Lx}x{Ly}'
            seed = int(hashlib.sha1(('fss' + name).encode()).hexdigest()[:12], 16)
            jobs.append((secs, f'data/cells_txt/{tag}.txt {mode} {Lx} {Ly} {secs} {seed} results/fss/{name}.txt'))
    jobs.sort(reverse=True)
    with open('run/jobs_fss.txt', 'w', newline='\n') as g:
        for _, line in jobs:
            g.write(line + '\n')
    tot = sum(s for s, _ in jobs)
    print(f'{len(jobs)} jobs, {tot/3600:.1f} CPU-h, longest {jobs[0][0]/60:.0f} min')


if __name__ == '__main__':
    main()
