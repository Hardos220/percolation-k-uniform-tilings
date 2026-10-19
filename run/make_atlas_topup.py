"""Top-up round for the atlas: the same tori as round 0, FACTOR times its CPU seconds,
new seeds, outputs in results/atlas_r{ROUND}/cNN/ (added to round 0 in the analysis).
FACTOR = 8 turns sigma ~6e-5 into ~2e-5 after combining with round 0.

usage: python run/make_atlas_topup.py ROUND FACTOR FIRST_CHUNK K [K ...]
  -> run/jobs_atlasr{ROUND}_cNN.txt, run/atlas_r{ROUND}_chunks.json
"""
import glob
import hashlib
import json
import sys

CHUNK_S = 6 * 2.5 * 3600


def main():
    rnd, factor, first, ks = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    chunks = []
    for k in ks:
        jobs = []
        for jf in sorted(glob.glob('run/jobs_atlas_c[0-9][0-9].txt')):
            for line in open(jf):
                x = line.split()
                tag = x[0].split('/')[-1][:-4]
                if not tag.startswith(k + '__'):
                    continue
                name = x[6].split('/')[-1][:-4]
                secs = round(float(x[4]) * factor, 1)
                seed = int(hashlib.sha1((f'atlas_r{rnd}' + name).encode()).hexdigest()[:12], 16)
                jobs.append((k, [x[0], x[1], x[2], x[3], str(secs), str(seed), name]))
        cur, t = [], 0
        for k_, j in jobs:
            if cur and t + float(j[4]) > CHUNK_S:
                chunks.append((k, cur))
                cur, t = [], 0
            cur.append(j)
            t += float(j[4])
        if cur:
            chunks.append((k, cur))
    meta = []
    for i, (k, js) in enumerate(chunks):
        c = f'c{first + i:02d}'
        with open(f'run/jobs_atlasr{rnd}_{c}.txt', 'w', newline='\n') as g:
            for cell, mode, Lx, Ly, secs, seed, name in js:
                g.write(f'{cell} {mode} {Lx} {Ly} {secs} {seed} results/atlas_r{rnd}/{c}/{name}.txt\n')
        s = sum(float(j[4]) for j in js)
        meta.append(dict(chunk=c, k=k, jobs=len(js), cpu_s=s))
        print(f'r{rnd} {c}: k={k} {len(js)} jobs, {s/3600:.1f} CPU-h')
    json.dump(meta, open(f'run/atlas_r{rnd}_chunks.json', 'w'), indent=1)
    print(f'total {sum(m["cpu_s"] for m in meta)/3600:.0f} CPU-h in {len(meta)} chunks')


if __name__ == '__main__':
    main()
