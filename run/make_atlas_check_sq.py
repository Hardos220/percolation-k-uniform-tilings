"""Atlas finite-size check, stratum S: the 30 atlas tilings (k >= 5) with the largest fraction of (4^4) vertices,
i.e., the largest square domains (fourth review). Same protocol as run/make_atlas_check.py: n1 = the atlas torus
again, n2 = a near-square torus with about four times as many sites.

usage: python run/make_atlas_check_sq.py OUT_JOBFILE_LOCAL OUT_JOBFILE_REMOTE   -> results/atlas_check_sample_sq.json
The jobs are split into two lists of about equal CPU time (this PC, and a second PC that is about twice slower).
"""
import csv
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_atlas_check import cpu_seconds, near_square  # noqa: E402


def frac44(vt):
    for part in vt.split(';'):
        t, f = part.rsplit(':', 1)
        if t == '4.4.4.4':
            a, b = (f.split('/') + ['1'])[:2]
            return int(a) / int(b)
    return 0.0


def main():
    out_local, out_remote = sys.argv[1:3]
    rows = [r for r in csv.DictReader(open('paper/supplemental/thresholds.csv')) if int(r['k']) >= 5]
    used = {r['id'] for r in json.load(open('results/atlas_check_sample.json'))}
    vt = {r['tag']: r['vertex_types'] for r in csv.DictReader(open('results/tables/all_thresholds.csv'))}
    cand = sorted((r for r in rows if r['id'] not in used), key=lambda r: (-frac44(vt[r['id']]), r['id']))[:30]
    jobs, sample = [], []
    for r in cand:
        tag = r['id']
        t = r['tori_site'].split(':')
        Lx1, Ly1 = map(int, t[0].split('x'))
        d = json.load(open(f'data/cells/{tag}.json'))
        nv, ne = int(r['n_v']), int(r['n_e'])
        Lx2, Ly2 = near_square(d, nv, 4 * Lx1 * Ly1 * nv)
        sample.append(dict(id=tag, stratum='S', k=int(r['k']), n_v=nv, f44=frac44(vt[tag]), torus1=f'{Lx1}x{Ly1}',
                           torus2=f'{Lx2}x{Ly2}', aspect1=float(t[-1])))
        for mode in ('site', 'bond'):
            sa = float(r[f'{mode}_err'])
            m = nv if mode == 'site' else ne
            for name, (Lx, Ly), sig in (('n1', (Lx1, Ly1), sa / math.sqrt(3)), ('n2', (Lx2, Ly2), sa / 2)):
                secs = cpu_seconds(Lx * Ly * m, sig)
                res = f'results/atlas_check/{tag}__{mode}__{Lx}x{Ly}__{name}.txt'
                seed = int(hashlib.sha1(res.encode()).hexdigest()[:12], 16)
                jobs.append((secs, f'data/cells_txt/{tag}.txt {mode} {Lx} {Ly} {secs:.0f} {seed} {res}'))
    jobs.sort(key=lambda x: -x[0])
    # greedy split: remote PC counts twice (it is about twice slower)
    loc, rem, tl, tr = [], [], 0.0, 0.0
    for secs, line in jobs:
        if tl <= 2 * tr:
            loc.append(line); tl += secs
        else:
            rem.append(line); tr += secs
    open(out_local, 'w', newline='\n').write('\n'.join(loc) + '\n')
    open(out_remote, 'w', newline='\n').write('\n'.join(rem) + '\n')
    json.dump(sample, open('results/atlas_check_sample_sq.json', 'w'), indent=1)
    print(len(sample), 'tilings; (4^4) fraction', round(sample[0]['f44'], 3), '..', round(sample[-1]['f44'], 3),
          '; k', sorted({s['k'] for s in sample}))
    print(f'local {len(loc)} jobs {tl/3600:.2f} CPU-h; remote {len(rem)} jobs {tr/3600:.2f} CPU-h (this-PC units)')
    print('tags:', ' '.join(s['id'] for s in sample))


if __name__ == '__main__':
    main()
