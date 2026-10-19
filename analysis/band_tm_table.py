"""Leading k = pi eigenvalues of the square-band transfer matrix for the paper (W = 4..Wmax).

usage: python analysis/band_tm_table.py WMAX  -> results/band_tm.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from band_tm import sector_eigs  # noqa: E402

POINTS = {'bond': 0.51, 'site': 0.62}  # near p_c of the h = 1 slip pair


def main():
    wmax = int(sys.argv[1])
    out = json.load(open('results/band_tm.json')) if os.path.exists('results/band_tm.json') else {}
    for mode, p in POINTS.items():
        for W in range(4, wmax + 1, 2):
            key = f'{mode}_W{W}'
            if key in out:
                continue
            n, v = sector_eigs(W, mode == 'site', p, np.pi, 6)
            out[key] = dict(mode=mode, W=W, p=p, states=n, k_pi=[[z.real, z.imag] for z in v])
            print(key, n, ' '.join(f'{z.real:+.4f}{z.imag:+.4f}i' for z in v), flush=True)
            json.dump(out, open('results/band_tm.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
