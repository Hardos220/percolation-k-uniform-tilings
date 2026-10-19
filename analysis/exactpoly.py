"""Exact critical polynomial on a small torus via nz/exact.exe (full enumeration)."""
import os
import subprocess
from fractions import Fraction

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nz', 'exact.exe')


def counts(cell, mode, Lx, Ly):
    """list of (m, nowrap, cross, other) for m occupied objects"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cells_txt', cell + '.txt')
    out = subprocess.run([os.path.abspath(EXE), path, mode, str(Lx), str(Ly)], capture_output=True, text=True, check=True).stdout
    return [tuple(map(int, l.split())) for l in out.splitlines() if l and l[0] != '#']


def value(rows, p):
    K = len(rows) - 1
    p = Fraction(p)
    return sum((c - a) * p ** m * (1 - p) ** (K - m) for m, a, c, w in rows)
