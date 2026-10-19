"""Wrapping probabilities from nz histograms.

R_X(p) = (1/M) sum_i P[Bin(N, p) >= n_i^X],  X in {h, v, e(=min), b(=max)},
which is the canonical-ensemble average of the microcanonical first-wrap
indicator (Newman & Ziff 2001).  Variance of R_X(p) is computed from the same
per-sample terms, so errors need no batching.
"""
import glob
import re

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

COLS = {'h': 1, 'v': 2, 'e': 3, 'b': 4, 'x': 5}


class Hist:
    def __init__(self, files=None, texts=None):
        """files: paths (or a glob); texts: file contents as strings (e.g. read from a zip)."""
        if isinstance(files, str):
            files = sorted(glob.glob(files))
        sources = [open(f).read() for f in (files or [])] + list(texts or [])
        acc = {}
        self.N = None
        self.M = 0
        for txt in sources:
            head, _, body = txt.partition('\n')
            m = re.search(r'N (\d+) samples (\d+)', head)
            N, M = int(m.group(1)), int(m.group(2))
            if self.N is None:
                self.N = N
                self.meta = head
            assert N == self.N, 'N differs between merged histograms'
            self.M += M
            for line in body.splitlines():
                if not line or line[0] == '#':
                    continue
                row = list(map(int, line.split()))
                acc.setdefault(row[0], np.zeros(5, np.int64))
                acc[row[0]] += row[1:6]
        self.n = np.array(sorted(acc), dtype=np.int64)
        self.c = np.array([acc[k] for k in self.n], dtype=np.float64)
        self.files = files

    def R(self, p, X='e'):
        """R_X(p) and its standard error."""
        w = self.c[:, COLS[X] - 1]
        S = binom.sf(self.n - 1, self.N, p)
        r = (w * S).sum() / self.M
        r2 = (w * S * S).sum() / self.M
        return r, np.sqrt(max(r2 - r * r, 0) / self.M)

    def dR(self, p, X='e', h=None):
        h = h or 1e-3 / np.sqrt(self.N)
        return (self.R(p + h, X)[0] - self.R(p - h, X)[0]) / (2 * h)

    def p_at(self, target, X='e'):
        """p with R_X(p) = target, and its statistical error."""
        lo, hi = self.n.min() / self.N * 0.9, min(1.0, self.n.max() / self.N * 1.1)
        f = lambda p: self.R(p, X)[0] - target
        p = brentq(f, max(lo, 1e-9), min(hi, 1 - 1e-12), xtol=1e-13)
        s = self.R(p, X)[1] / self.dR(p, X)
        return p, s

    def mean_n(self, X='e'):
        w = self.c[:, COLS[X] - 1]
        return (w * self.n).sum() / self.M
