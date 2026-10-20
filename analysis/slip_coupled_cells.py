"""Cells with a common numbering of sites and bonds for the two members of a slip pair (coupled sampling).

The eclipsed member E (T2 orthogonal to T1) is cut along a line of constant T2 coordinate; its vertices are
re-assigned to the cells so that this line is the cell boundary.  The staggered partner S' is E with the rows of
cells shifted by T1/2: T2' = T2 + s T1/2, and every edge that crosses the cut is re-attached to the vertex that the
shifted row brings to its end point.  Vertex v and edge k of S' correspond to vertex v and edge k of E, so a common
random order of the elements couples the two simulations.  A cut is accepted if S' is a valid tiling (unit edges,
vertex types, closed faces) and is isometric to the staggered tiling of the catalog (analysis/periodic_checks.py).

usage: python analysis/slip_coupled_cells.py STAGGERED_TAG ECLIPSED_TAG OUTDIR
writes OUTDIR/<name>_E.txt and OUTDIR/<name>_S.txt (nz cell format) and OUTDIR/<name>.json (cut, s, check)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from periodic_checks import Cell, isometries  # noqa: E402


def recut(d, c):
    """Cell d with vertices re-assigned so that the cell rows are [c, c+1) in the T2 coordinate."""
    T1, T2 = np.array(d['T1']), np.array(d['T2'])
    M = np.column_stack([T1, T2])
    Mi = np.linalg.inv(M)
    P = np.array(d['pos'], float)
    f2 = (Mi @ P.T)[1]
    shift = np.where(f2 < c - 1e-9, 1, 0)
    P2 = P + shift[:, None] * T2
    edges = []
    for u, w, a, b in d['edges']:
        vec = P[w] + a * T1 + b * T2 - P[u]
        ab = Mi @ (P2[u] + vec - P2[w])
        edges.append((u, w, int(round(ab[0])), int(round(ab[1]))))
    return P2, edges


def slipped(d, P2, edges, s):
    """Partner with T2' = T2 + s T1/2; None if an end point is not a vertex."""
    T1, T2 = np.array(d['T1']), np.array(d['T2'])
    M = np.column_stack([T1, T2])
    Mi = np.linalg.inv(M)
    out = []
    for u, w, a, b in edges:
        if b == 0:
            out.append((u, w, a, 0))
            continue
        Y = P2[w] + a * T1 - b * s * T1 / 2
        hit = None
        for v in range(len(P2)):
            x, y = Mi @ (Y - P2[v])
            if abs(y) < 1e-6 and abs(x - round(x)) < 1e-6:
                hit = (v, int(round(x)))
                break
        if hit is None:
            return None
        out.append((u, hit[0], hit[1], b))
    return out


def as_cell(d, P2, edges, T2):
    e = dict(d)
    e['pos'] = [list(map(float, p)) for p in P2]
    e['T2'] = list(map(float, T2))
    e['edges'] = [list(x) for x in edges]
    return e


def main():
    tag_s, tag_e, outdir = sys.argv[1:4]
    dS = json.load(open(f'data/cells/{tag_s}.json'))
    dE = json.load(open(f'data/cells/{tag_e}.json'))
    T1, T2 = np.array(dE['T1']), np.array(dE['T2'])
    assert abs(T1 @ T2) < 1e-6, 'first tag must be staggered, second eclipsed'
    GS = Cell(dS)
    Mi = np.linalg.inv(np.column_stack([T1, T2]))
    f2 = sorted(set(round(float(x), 6) for x in (Mi @ np.array(dE['pos']).T)[1]))
    cuts = [0.0] + [(f2[i] + f2[i + 1]) / 2 for i in range(len(f2) - 1)]
    for c in cuts:
        P2, eE = recut(dE, c)
        for s in (1, -1):
            eS = slipped(dE, P2, eE, s)
            if eS is None:
                continue
            cE = as_cell(dE, P2, eE, T2)
            cS = as_cell(dE, P2, eS, T2 + s * T1 / 2)
            GE2, GS2 = Cell(cE), Cell(cS)
            if GE2.validity() != 'ok' or GS2.validity() != 'ok':
                continue
            if not isometries(GE2, Cell(dE), first=True) or not isometries(GS2, GS, first=True):
                continue
            os.makedirs(outdir, exist_ok=True)
            name = tag_e
            for suf, cc in (('E', cE), ('S', cS)):
                with open(os.path.join(outdir, f'{name}_{suf}.txt'), 'w', newline='\n') as g:
                    g.write(f"{cc['nv']} {len(cc['edges'])}\n")
                    for u, w, a, b in cc['edges']:
                        g.write(f'{u} {w} {a} {b}\n')
            n_cross = sum(1 for e in eE if e[3] != 0)
            json.dump(dict(staggered=tag_s, eclipsed=tag_e, cut=c, s=s, nv=dE['nv'], ne=len(eE), cross_edges=n_cross,
                           T1=list(T1), T2=list(T2), T2_staggered=list(T2 + s * T1 / 2), checked='valid; isometric to catalog'),
                      open(os.path.join(outdir, f'{name}.json'), 'w'), indent=1)
            print(f'{name}: cut {c:.4f} s {s:+d}, {n_cross} of {len(eE)} edges re-attached; both cells verified')
            return
    raise SystemExit(f'no valid cut for {tag_s} / {tag_e}')


if __name__ == '__main__':
    main()
