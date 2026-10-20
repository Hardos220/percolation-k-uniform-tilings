"""Add the extra runs in results/core_extra/ to the core report (results/core/report_r0.json).

For every threshold with extra histograms (<tag>__<mode>__<Lx>x<Ly>__<name>.txt), the histograms of each torus are
added to the original core histograms of the same torus (results/core/**/<tag>__<mode>__<Lx>x<Ly>__r*.txt; their
number must equal the one recorded in the report), and the finite-size fit of analysis/core_report.py,
p_N = p_c + a N_s^(-11/8), is repeated.  Only these thresholds change.

usage: python analysis/core_extra.py
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


def fit(sizes):
    xs = [r['sites'] ** (-1.375) for r in sizes]
    w = [1 / r['s'] ** 2 for r in sizes]
    S, Sx, Sxx = sum(w), sum(wi * x for wi, x in zip(w, xs)), sum(wi * x * x for wi, x in zip(w, xs))
    Sy = sum(wi * r['p'] for wi, r in zip(w, sizes))
    Sxy = sum(wi * x * r['p'] for wi, x, r in zip(w, xs, sizes))
    det = S * Sxx - Sx * Sx
    pc = (Sxx * Sy - Sx * Sxy) / det
    a = (S * Sxy - Sx * Sy) / det
    chi2 = sum(wi * (r['p'] - pc - a * x) ** 2 for wi, x, r in zip(w, xs, sizes))
    return pc, math.sqrt(Sxx / det), a, chi2


def main():
    rep = json.load(open('results/core/report_r0.json'))
    idx = {(r['tag'], r['mode']): r for r in rep}
    extra = collections.defaultdict(list)
    for f in glob.glob('results/core_extra/**/*.txt', recursive=True):
        m = re.match(r'(.+)__(site|bond)__(\d+)x(\d+)__\w+\.txt$', os.path.basename(f))
        if m and (m.group(1), m.group(2)) in idx:
            extra[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))].append(f)
    done = set()
    for (tag, mode, Lx, Ly), fs in sorted(extra.items()):
        rec = idx[(tag, mode)]
        sz = next(s for s in rec['sizes'] if s['Lx'] == Lx and s['Ly'] == Ly)
        orig = sorted(g for g in glob.glob(f'results/core/**/{tag}__{mode}__{Lx}x{Ly}__r*.txt', recursive=True))
        assert len(orig) == sz['files'], (tag, mode, Lx, Ly, orig)
        H = Hist(orig + fs)
        p, s = p_cross(H)
        print(f'{tag} {mode} {Lx}x{Ly}: samples {sz["M"]} -> {H.M}, p {sz["p"]:.7f}({sz["s"]*1e7:.0f}) -> {p:.7f}({s*1e7:.0f})')
        sz.update(M=H.M, p=p, s=s, files=len(orig) + len(fs), extra=len(fs))
        done.add((tag, mode))
    for key in sorted(done):
        rec = idx[key]
        old = (rec['pc'], rec['spc'])
        rec['pc'], rec['spc'], rec['a'], rec['chi2'] = fit(rec['sizes'])
        print(f'{key[0]} {key[1]}: p_c {old[0]:.7f}({old[1]*1e7:.0f}) -> {rec["pc"]:.7f}({rec["spc"]*1e7:.0f}), chi2 {rec["chi2"]:.2f}')
    json.dump(rep, open('results/core/report_r0.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
