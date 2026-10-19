"""Task lists for the GPU core run: all k <= 4 tilings, site and bond, three torus sizes.

Torus (q L) x (p L) cells, q/p ~ |T2|/|T1|, L chosen so that the torus has about
N_TARGET sites (L >= 3).  GPU seconds per task from the pilot cost model
sigma*sqrt(T) = 1.65e-5 (N/256)^0.24 (calibrated in a GPU pilot run), times a safety factor.
Tasks are packed in order (validation lattices first) into batches of about BATCH_H hours.

usage: python run/make_core_tasks.py [ROUND]
writes gpu/tasks/core_r{ROUND}_bNN.txt  lines: cell mode Lx Ly seconds seed name
"""
import glob
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from make_jobs import shape  # noqa: E402

SIZES = [(500, 3e-6), (2000, 2e-6), (8000, 3e-6)]  # (target sites, target sigma)
SAFETY = 1.2
BATCH_H = 3.5


def cost(n):
    return 1.65e-5 * (n / 256) ** 0.24


def main():
    rnd = sys.argv[1] if len(sys.argv) > 1 else '0'
    tasks = []
    for k in ('01', '02', '03', '04'):
        for f in sorted(glob.glob(f'data/cells/{k}__*.json')):
            d = json.load(open(f))
            tag = os.path.basename(f)[:-5]
            q, p = shape(d)
            for mode in ('site', 'bond'):
                Ls = []
                for n_t, sig in SIZES:
                    L = max(3, round(math.sqrt(n_t / (q * p * d['nv']))))
                    if L in [x[0] for x in Ls]:
                        L = Ls[-1][0] + 1
                    Ls.append((L, sig))
                for L, sig in Ls:
                    Lx, Ly = q * L, p * L
                    n = Lx * Ly * d['nv']
                    secs = max(10, round(SAFETY * (cost(n) / sig) ** 2))
                    name = f'{tag}__{mode}__{Lx}x{Ly}__r{rnd}'
                    seed = int(hashlib.sha1(name.encode()).hexdigest()[:12], 16)
                    tasks.append((tag, mode, Lx, Ly, secs, seed, name))
    os.makedirs('gpu/tasks', exist_ok=True)
    batches, cur, t = [], [], 0
    for x in tasks:
        if cur and t + x[4] > BATCH_H * 3600:
            batches.append(cur)
            cur, t = [], 0
        cur.append(x)
        t += x[4]
    if cur:
        batches.append(cur)
    tot = 0
    for i, b in enumerate(batches):
        fn = f'gpu/tasks/core_r{rnd}_b{i:02d}.txt'
        with open(fn, 'w', newline=chr(10)) as g:  # LF: the list is read by bash on Linux
            for x in b:
                g.write(' '.join(map(str, x)) + '\n')
        s = sum(x[4] for x in b)
        tot += s
        print(f'{fn}: {len(b)} tasks, {s/3600:.2f} h, first {b[0][6]}')
    print(f'{len(tasks)} tasks, {len(batches)} batches, {tot/3600:.1f} GPU-h')


if __name__ == '__main__':
    main()
