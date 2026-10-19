"""Exact small-torus critical polynomials for slip pairs (and a control pair)."""
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactpoly import counts  # noqa: E402

pairs = json.load(open('results/degeneracy_candidates.json'))


def jobs():
    for a, b in pairs:
        da, db = (json.load(open(f'data/cells/{t}.json')) for t in (a, b))
        if da['nv'] != db['nv']:
            continue
        nv, ne = da['nv'], da['ne']
        for mode, Lx, Ly, K in (('site', 1, 1, nv), ('site', 2, 1, 2 * nv), ('site', 1, 2, 2 * nv), ('bond', 1, 1, ne)):
            if K <= 26:
                yield (a, b, mode, Lx, Ly)


def work(j):
    a, b, mode, Lx, Ly = j
    return j, counts(a, mode, Lx, Ly), counts(b, mode, Lx, Ly)


if __name__ == '__main__':
    out = []
    with mp.Pool(int(os.environ.get('RMQ_CPU', '1'))) as pool:
        for (a, b, mode, Lx, Ly), ca, cb in pool.imap(work, list(jobs())):
            same = ca == cb
            nowrap_same = [r[1] for r in ca] == [r[1] for r in cb]
            cross_same = [r[2] for r in ca] == [r[2] for r in cb]
            out.append(dict(a=a, b=b, mode=mode, torus=f'{Lx}x{Ly}', identical=same, nowrap_same=nowrap_same,
                            cross_same=cross_same, ca=ca, cb=cb))
            print(f"{a[4:]:24s} vs {b[4:]:24s} {mode} {Lx}x{Ly}: {'IDENTICAL' if same else 'differ'}"
                  f" (nowrap {'=' if nowrap_same else '!='}, cross {'=' if cross_same else '!='})", flush=True)
    json.dump(out, open('results/slip_pairs_exact.json', 'w'))
