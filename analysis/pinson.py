"""Exact wrapping probabilities of critical percolation on a torus C/(Z + tau Z)
(Pinson, J. Stat. Phys. 75, 1167 (1994); notation of Arguin and of
Morin-Duchesne et al., arXiv 1402.0879, Sec. 2), Coulomb gas g = 8/3, e0 = 2/3.

pi0 = pi_plus = Z_c(g, 1)/2,   pi^{a,b} = sum_{k != 0} [Z_{bk,ak}(g/4) (cos(pi e0 k) - cos(pi k))].
R_e = 1 - pi0;  R_h = pi_plus + sum_{a != 0} pi^{a,b};  R_v likewise with b;
R_b = pi_plus + sum_{a, b != 0} pi^{a,b}.
"""
import math
from math import gcd

import numpy as np

G = 8.0 / 3.0
E0 = 2.0 / 3.0


def eta_abs(tau):
    q = np.exp(2j * np.pi * tau)
    prod = 1.0 + 0j
    qn = q
    for _ in range(400):
        prod *= (1 - qn)
        qn *= q
        if abs(qn) < 1e-18:
            break
    return abs(np.exp(2j * np.pi * tau / 24) * prod)


def Zmm(m, mp, g, tau):
    tr, ti = tau.real, tau.imag
    return math.sqrt(g / ti) / eta_abs(tau) ** 2 * math.exp(
        -math.pi * g * (m * m * ti * ti + (mp - m * tr) ** 2) / ti)


def wrapping(tau, K=40):
    tau = complex(tau)
    g4 = G / 4
    # Z_c(g, e0) = sum_{m,m'} Z_{m,m'}(g/4) cos(pi e0 gcd(m,m'))
    zc1 = 0.0
    zce = 0.0
    for m in range(-K, K + 1):
        for mp in range(-K, K + 1):
            z = Zmm(m, mp, g4, tau)
            d = gcd(abs(m), abs(mp))
            zc1 += z * math.cos(math.pi * d)
            zce += z * math.cos(math.pi * E0 * d)
    pi0 = 0.5 * zc1
    pip = pi0
    Rh = Rv = Rb = pip
    tot = pi0 + pip
    for a in range(0, K + 1):
        for b in range(-K, K + 1):
            if gcd(a, abs(b)) != 1 or (a == 0 and b != 1):
                continue
            s = 0.0
            for k in range(-K, K + 1):
                if k == 0:
                    continue
                s += Zmm(b * k, a * k, g4, tau) * (math.cos(math.pi * E0 * k) - math.cos(math.pi * k))
            tot += s
            if a != 0:
                Rh += s
            if b != 0:
                Rv += s
            if a != 0 and b != 0:
                Rb += s
    return dict(Re=1 - pi0, Rh=Rh, Rv=Rv, Rb=Rb, Z=zce, total=tot)


if __name__ == '__main__':
    for name, tau in (('square', 1j), ('hexagonal', complex(0.5, math.sqrt(3) / 2))):
        r = wrapping(tau)
        print(name, ' '.join(f'{k}={v:.9f}' for k, v in r.items()))
