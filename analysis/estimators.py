"""Finite-size estimators of p_c from nz histograms.

cross:  p_L solves R_e(p) + R_x(p) = 1  (at p_c, P[no wrap] = P[cross] for any torus,
        Pinson/Arguin: pi^0 = pi^+ for Q = 1), valid for anisotropic lattices.
Re:     p_L solves R_e(p) = R_e^inf(tau)  (isotropic lattices only, tau from the cell).
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from pinson import wrapping
from wrap import COLS

_RINF = {}


def tau_of(d, Lx, Ly):
    """Modular parameter of the torus spanned by Lx T1, Ly T2 (orientation-free)."""
    a = complex(*d['T1']) * Lx
    b = complex(*d['T2']) * Ly
    t = b / a
    if t.imag < 0:
        t = t.conjugate()
    return t


def rinf(tau):
    key = (round(tau.real, 9), round(tau.imag, 9))
    if key not in _RINF:
        _RINF[key] = wrapping(tau)['Re']
    return _RINF[key]


def _terms(H, p, cols):
    S = binom.sf(H.n - 1, H.N, p)
    return [H.c[:, COLS[c] - 1] * S for c in cols], S


def p_cross(H):
    """Root of R_e + R_x - 1; error from var(R_e) + var(R_x) + 2 sd sd (upper bound)."""
    def f(p):
        (we, wx), _ = _terms(H, p, 'ex')
        return (we.sum() + wx.sum()) / H.M - 1

    lo, hi = max(H.n.min() / H.N * 0.8, 1e-9), min(H.n.max() / H.N * 1.2, 1 - 1e-12)
    p = brentq(f, lo, hi, xtol=1e-14)
    se = H.R(p, 'e')[1]
    sx = H.R(p, 'x')[1]
    h = 2e-3 / math.sqrt(H.N)
    slope = (f(p + h) - f(p - h)) / (2 * h)
    return p, (se + sx) / slope


def p_re(H, target):
    p, s = H.p_at(target, 'e')
    return p, s


def fit_fss(Ls, ps, ss, omega=None):
    """Weighted fit p_L = p_c + a L^-omega (omega fixed or free). Returns p_c, err, a, omega, chi2/dof."""
    Ls, ps, ss = map(np.asarray, (Ls, ps, ss))
    w = 1 / ss ** 2

    def lin(om):
        X = np.column_stack([np.ones_like(Ls, dtype=float), Ls ** (-om)])
        A = X.T @ (X * w[:, None])
        coef = np.linalg.solve(A, X.T @ (w * ps))
        cov = np.linalg.inv(A)
        r = ps - X @ coef
        return coef, cov, (w * r * r).sum()

    if omega is None:
        oms = np.linspace(0.5, 5, 451)
        om = oms[int(np.argmin([lin(o)[2] for o in oms]))]
    else:
        om = omega
    coef, cov, chi2 = lin(om)
    dof = max(len(Ls) - (2 if omega is not None else 3), 1)
    return coef[0], math.sqrt(cov[0, 0]), coef[1], om, chi2 / dof
