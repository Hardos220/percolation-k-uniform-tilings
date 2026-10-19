"""Core run report: per task p_cross and sigma; per threshold all sizes, FSS extrapolation
p_N = p_c + a N^(-w/2) (w = 2.75 fixed, N = sites), comparison with references.

usage: python analysis/core_report.py OUT_JSON  [result dirs...]
Result files are found as <dir>/**/<tag>__<mode>__<Lx>x<Ly>__r<k>.txt; several rounds of
the same task (r0, r1, ...) are added together.
"""
import collections
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from estimators import p_cross  # noqa: E402
from wrap import Hist  # noqa: E402

TARGET = {0: 3e-6, 1: 2e-6, 2: 3e-6}  # by size rank (small, main, large)


def references():
    ref = {}
    for tag, v in json.load(open('reference/archimedean.json')).items():
        if not tag.startswith('_'):
            ref[(tag, 'site')] = (v['site'], 0.0)
            ref[(tag, 'bond')] = (v['bond'], 0.0)
    for r in json.load(open('reference/two_uniform_reference.json')):
        tag = r['cell'][:-5]
        for mode in ('site', 'bond'):
            if len(r[mode]) == 1:
                m = re.match(r'([\d.]+)\((\d+)\)', r[mode][0])
                ref[(tag, mode)] = (float(m.group(1)), int(m.group(2)) * 10 ** -len(m.group(1).split('.')[1]))
    # #16/#17 resolved by the pilot: 4f_4k_1 = #17, 4f_4k_2 = #16
    ref[('02__2__eu_4f_4k_1', 'bond')] = (0.51769462, 3.5e-9)
    ref[('02__2__eu_4f_4k_2', 'bond')] = (0.51891529, 3.5e-9)
    return ref


def main():
    out = sys.argv[1]
    dirs = sys.argv[2:]
    files = collections.defaultdict(list)
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', '*__r*.txt'), recursive=True):
            m = re.match(r'(.+)__(site|bond)__(\d+)x(\d+)__r(\d+)\.txt$', os.path.basename(f))
            if m:
                files[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))].append(f)
    ref = references()
    by_thr = collections.defaultdict(list)
    for (tag, mode, Lx, Ly), fs in files.items():
        H = Hist(fs)
        p, s = p_cross(H)
        by_thr[(tag, mode)].append(dict(Lx=Lx, Ly=Ly, N=H.N, M=H.M, p=p, s=s, files=len(fs)))
    rows = []
    for (tag, mode), sizes in sorted(by_thr.items()):
        sizes.sort(key=lambda r: r['Lx'] * r['Ly'])
        nv = json.load(open(f'data/cells/{tag}.json'))['nv']
        for i, r in enumerate(sizes):
            r['sites'] = r['Lx'] * r['Ly'] * nv
            r['target'] = TARGET.get(i if len(sizes) == 3 else 1)
        rec = dict(tag=tag, mode=mode, sizes=sizes)
        if len(sizes) >= 2:
            # weighted LSQ with fixed exponent: p = pc + a x, x = sites^(-1.375)
            xs = [r['sites'] ** (-1.375) for r in sizes]
            w = [1 / r['s'] ** 2 for r in sizes]
            S, Sx, Sxx = sum(w), sum(wi * x for wi, x in zip(w, xs)), sum(wi * x * x for wi, x in zip(w, xs))
            Sy = sum(wi * r['p'] for wi, r in zip(w, sizes))
            Sxy = sum(wi * x * r['p'] for wi, x, r in zip(w, xs, sizes))
            det = S * Sxx - Sx * Sx
            pc = (Sxx * Sy - Sx * Sxy) / det
            a = (S * Sxy - Sx * Sy) / det
            spc = math.sqrt(Sxx / det)
            chi2 = sum(wi * (r['p'] - pc - a * x) ** 2 for wi, x, r in zip(w, xs, sizes))
            rec.update(pc=pc, spc=spc, a=a, chi2=chi2, dof=len(sizes) - 2)
        rec['ref'] = ref.get((tag, mode))
        rows.append(rec)
    json.dump(rows, open(out, 'w'), indent=1)
    n_task = sum(len(r['sizes']) for r in rows)
    short = sum(1 for r in rows for s in r['sizes'] if s['s'] > 1.05 * s['target'])
    print(f'{len(rows)} thresholds, {n_task} tasks; tasks above target sigma: {short}')
    print('reference comparisons:')
    for r in rows:
        if r['ref'] is None:
            continue
        rv, re_ = r['ref']
        line = f"  {r['tag']:26s} {r['mode']}: ref {rv:.8f}"
        for s in r['sizes']:
            line += f"  N={s['sites']}: {s['p']:.7f}({s['s']*1e7:.0f}) {((s['p']-rv)/math.hypot(s['s'], re_)):+.1f}s"
        if 'pc' in r:
            line += f"  | fit {r['pc']:.7f}({r['spc']*1e7:.0f}) {((r['pc']-rv)/math.hypot(r['spc'], re_)):+.1f}s"
        print(line)


if __name__ == '__main__':
    main()
