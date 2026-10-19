"""Job lists for the atlas (CPU, nz.exe): k = 5..10, site and bond, one torus size.

Torus (q L) x (p L) cells with q/p ~ |T2|/|T1|, L = max(3, round(sqrt(N_TARGET/(q p nv)))).
CPU seconds per job for sigma = SIGMA from an empirical cost model, sigma*sqrt(T) = c (N/256)^0.24 with c
calibrated in a pilot run (6.7 x 1.65e-5 per CPU core used).  Only tilings parsed OK are used.
Chunks: up to CHUNK_S CPU seconds, never mixing k (each chunk ships the cells of one k).

usage: python run/make_atlas_jobs.py FIRST_CHUNK_INDEX K [K ...]  ->  run/jobs_atlas_cNN.txt + run/atlas_chunks_*.json
"""
import glob
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_jobs import shape  # noqa: E402

N_TARGET = 1000
SIGMA = 5e-5
SAFETY = 1.1
CHUNK_S = 6 * 2.5 * 3600


def cpu_cost(n):
    return 6.7 * 1.65e-5 * (n / 256) ** 0.24


def main():
    first = int(sys.argv[1])
    ks = sys.argv[2:]
    queued = set()  # tilings already in earlier chunk lists are skipped
    for jf in glob.glob('run/jobs_atlas_c*.txt'):
        for line in open(jf):
            if line.strip():
                queued.add(os.path.basename(line.split()[0])[:-4])
    chunks, bad = [], []
    for k in ks:
        jobs = []
        for f in sorted(glob.glob(f'data/cells/{k}__*.json')):
            d = json.load(open(f))
            tag = os.path.basename(f)[:-5]
            if tag in queued:
                continue
            if not d['ok']:
                bad.append(tag)
                continue
            q, p = shape(d)
            L = max(3, round(math.sqrt(N_TARGET / (q * p * d['nv']))))
            Lx, Ly = q * L, p * L
            n = Lx * Ly * d['nv']
            secs = max(3, round(SAFETY * (cpu_cost(n) / SIGMA) ** 2, 1))
            for mode in ('site', 'bond'):
                name = f'{tag}__{mode}__{Lx}x{Ly}'
                seed = int(hashlib.sha1(('atlas' + name).encode()).hexdigest()[:12], 16)
                jobs.append((tag, mode, Lx, Ly, secs, seed, name))
        cur, t = [], 0
        for j in jobs:
            if cur and t + j[4] > CHUNK_S:
                chunks.append((k, cur))
                cur, t = [], 0
            cur.append(j)
            t += j[4]
        if cur:
            chunks.append((k, cur))
    meta = []
    for i, (k, js) in enumerate(chunks):
        c = f'c{first + i:02d}'
        with open(f'run/jobs_atlas_{c}.txt', 'w') as g:
            for tag, mode, Lx, Ly, secs, seed, name in js:
                g.write(f'data/cells_txt/{tag}.txt {mode} {Lx} {Ly} {secs} {seed} results/atlas/{c}/{name}.txt\n')
        s = sum(j[4] for j in js)
        meta.append(dict(chunk=c, k=k, jobs=len(js), cpu_s=s))
        print(f'{c}: k={k} {len(js)} jobs, {s/3600:.1f} CPU-h, ~{s/6/3600:.1f} h on 6 threads')
    json.dump(meta, open(f'run/atlas_chunks_{first:02d}.json', 'w'), indent=1)
    tot = sum(m['cpu_s'] for m in meta)
    print(f'{sum(m["jobs"] for m in meta)} jobs, {len(meta)} chunks, {tot/3600:.0f} CPU-h; not parsed OK: {len(bad)}')


if __name__ == '__main__':
    main()
