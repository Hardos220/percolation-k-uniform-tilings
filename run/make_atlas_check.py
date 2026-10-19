"""Finite-size check of the single-torus atlas (k >= 5) on a stratified sample of tilings.

Strata (disjoint): E = 35 tilings with the most elongated atlas tori; L = 35 tilings with the largest unit
cells; R = 5 random tilings per k = 5..10. For each tiling and mode two runs:
  n1: the atlas torus again (added to the atlas histograms, combined error ~ sigma_atlas / 2);
  n2: a near-square torus with about four times as many sites (error ~ sigma_atlas / 2).
The difference p(n2) - p(n1) measures the finite-size bias of the atlas value.
CPU time per run from the atlas cost model sigma^2 T = 2.0e-8 N^0.176 (CPU-s), times SAFETY.

usage: python run/make_atlas_check.py OUT_JOBFILE   -> also results/atlas_check_sample.json
"""
import csv
import hashlib
import json
import math
import random
import sys

SAFETY = 1.3
TMAX = 2400


def cpu_seconds(N, sigma):
    return min(TMAX, max(5.0, SAFETY * 2.0e-8 * N ** 0.176 / sigma ** 2))


def near_square(d, nv, target):
    """Lx x Ly cells with Lx|T1| ~ Ly|T2| and about `target` sites, Lx, Ly >= 2."""
    l1, l2 = math.hypot(*d['T1']), math.hypot(*d['T2'])
    best = None
    for Ly in range(2, 200):
        for Lx in range(2, 400):
            n = Lx * Ly * nv
            if n > 3 * target:
                break
            asp = (Lx * l1) / (Ly * l2)
            cost = abs(math.log(asp)) + abs(math.log(n / target))
            if best is None or cost < best[0]:
                best = (cost, Lx, Ly)
    return best[1], best[2]


def main():
    out = sys.argv[1]
    rows = [r for r in csv.DictReader(open('paper/supplemental/thresholds.csv')) if int(r['k']) >= 5]
    for r in rows:
        t = r['tori_site'].split(':')
        r['torus'] = t[0]
        r['aspect'] = float(t[-1])
    rng = random.Random(20261019)
    by_aspect = sorted(rows, key=lambda r: (r['aspect'], r['id']))
    E = [r['id'] for r in by_aspect[:35]]
    rest = [r for r in rows if r['id'] not in E]
    L = [r['id'] for r in sorted(rest, key=lambda r: (-int(r['n_v']), r['id']))[:35]]
    used = set(E) | set(L)
    R = []
    for k in range(5, 11):
        pool = sorted(r['id'] for r in rows if int(r['k']) == k and r['id'] not in used)
        R += rng.sample(pool, 5)
    strata = {**{t: 'E' for t in E}, **{t: 'L' for t in L}, **{t: 'R' for t in R}}
    info = {r['id']: r for r in rows}
    lines, total, sample = [], 0.0, []
    for tag, st in strata.items():
        r = info[tag]
        d = json.load(open(f'data/cells/{tag}.json'))
        Lx1, Ly1 = map(int, r['torus'].split('x'))
        nv, ne = int(r['n_v']), int(r['n_e'])
        Lx2, Ly2 = near_square(d, nv, 4 * Lx1 * Ly1 * nv)
        rec = dict(id=tag, stratum=st, k=int(r['k']), n_v=nv, torus1=f'{Lx1}x{Ly1}', torus2=f'{Lx2}x{Ly2}',
                   aspect1=r['aspect'])
        for mode in ('site', 'bond'):
            sa = float(r[f'{mode}_err'])
            m = nv if mode == 'site' else ne
            for name, (Lx, Ly), sig in (('n1', (Lx1, Ly1), sa / math.sqrt(3)), ('n2', (Lx2, Ly2), sa / 2)):
                N = Lx * Ly * m
                secs = cpu_seconds(N, sig)
                res = f'results/atlas_check/{tag}__{mode}__{Lx}x{Ly}__{name}.txt'
                seed = int(hashlib.sha1(res.encode()).hexdigest()[:12], 16)
                lines.append((secs, f'data/cells_txt/{tag}.txt {mode} {Lx} {Ly} {secs:.0f} {seed} {res}'))
                total += secs
        sample.append(rec)
    lines.sort(key=lambda x: -x[0])  # longest first for load balance
    open(out, 'w', newline='\n').write('\n'.join(l for _, l in lines) + '\n')
    json.dump(sample, open('results/atlas_check_sample.json', 'w'), indent=1)
    print(len(sample), 'tilings', len(lines), 'jobs', f'{total / 3600:.1f} CPU-h', 'max job %.0f s' % lines[0][0])


if __name__ == '__main__':
    main()
