"""Mean Euler characteristic per vertex of the occupied polygon complex and its
zero p* (topological estimate of p_c, Neher, Mecke, Wagner, J. Stat. Mech. 2008 P01011).

site:  chi(p) = p - (z/2) p^2 + sum_k f_k p^k      (face present if all k corners occupied)
bond:  chi(p) = 1 - (z/2) p   + sum_k f_k p^k      (face present if all k sides occupied)
with z the mean degree and f_k the number of k-gons per vertex.
"""
import json
import sys
from collections import Counter

import numpy as np
from scipy.optimize import brentq


def face_counts(d):
    f = Counter()
    for cfg in d['vertex_configs']:
        for k in cfg:
            f[k] += 1.0 / k
    nv = d['nv']
    return {k: v / nv for k, v in f.items()}


def chi(d, p, mode):
    z = 2 * d['ne'] / d['nv']
    fk = face_counts(d)
    s = sum(v * p ** k for k, v in fk.items())
    return (p - z / 2 * p * p + s) if mode == 'site' else (1 - z / 2 * p + s)


def p_star(d, mode):
    f = lambda p: chi(d, p, mode)
    grid = np.linspace(1e-6, 1 - 1e-9, 2001)
    vals = [f(p) for p in grid]
    roots = [brentq(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]
    return roots


if __name__ == '__main__':
    for path in sys.argv[1:]:
        d = json.load(open(path))
        print(path, 'site', p_star(d, 'site'), 'bond', p_star(d, 'bond'))
