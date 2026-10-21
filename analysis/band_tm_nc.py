"""Square-band transfer matrix restricted to cyclically non-crossing connectivity states.

On a cylinder the part of the band already traversed is an annulus bounded by the current row; a path in
it joining two sites of the row cuts off a disk together with one of the two arcs between them, so only
non-crossing partitions (of the occupied sites, for site percolation) are reachable. The full set-partition basis of band_tm.py is block-triangular, T = [[A, B], [0, D]],
and its spectrum contains spurious eigenvalues of the crossing block D. This script recomputes the
kappa = pi spectrum on the physical (non-crossing) subspace; closure of that subspace under every
step is checked by construction (an unreachable state raises KeyError).

usage: python analysis/band_tm_nc.py WMAX [bond] [site] [--p bond=0.51,site=0.62] [--out results/band_tm_nc.json]
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import band_tm  # noqa: E402

_all_states = band_tm.states


def noncrossing(s):
    """No a < b < c < d with s[a] == s[c] != s[b] == s[d] (labels of occupied sites, row positions cyclic)."""
    occ = [(i, x) for i, x in enumerate(s) if x >= 0]
    for (a, x1), (b, y1), (c, x2), (d, y2) in itertools.combinations(occ, 4):
        if x1 == x2 and y1 == y2 and x1 != y1:
            return False
    return True


def nc_states(W, site):
    return [s for s in _all_states(W, site) if noncrossing(s)]


def main():
    args = sys.argv[1:]
    pmap = {'bond': 0.51, 'site': 0.62}
    out_path = 'results/band_tm_nc.json'
    if '--p' in args:  # e.g. --p bond=0.5,site=0.5927 (limiting thresholds of the slip family)
        i = args.index('--p')
        for kv in args[i + 1].split(','):
            k, v = kv.split('=')
            pmap[k] = float(v)
        args = args[:i] + args[i + 2:]
    if '--out' in args:
        i = args.index('--out')
        out_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    wmax = int(args[0])
    modes = args[1:] or ['bond', 'site']
    band_tm.states = nc_states
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for mode, p in (('bond', pmap['bond']), ('site', pmap['site'])):
        if mode not in modes:
            continue
        for W in range(4, wmax + 1, 2):
            key = f'{mode}_W{W}'
            if key in out:
                continue
            n, v = band_tm.sector_eigs(W, mode == 'site', p, np.pi, 6)
            n0, v0 = band_tm.sector_eigs(W, mode == 'site', p, 0.0, 3)
            out[key] = dict(mode=mode, W=W, p=p, states=n, k_pi=[[z.real, z.imag] for z in v],
                            k_0=[[z.real, z.imag] for z in v0])
            print(key, n, 'pi:', ' '.join(f'{z.real:+.4f}{z.imag:+.4f}i' for z in v),
                  '| 0:', ' '.join(f'{z.real:+.4f}' for z in v0), flush=True)
            json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
