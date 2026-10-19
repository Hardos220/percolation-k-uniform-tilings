"""Are the matching graphs G1*, G2* of a slip pair isomorphic (while G1, G2 are not)?

Faces are recovered from the planar embedding of the cell graph (orbit positions and
translations); G* adds an edge between every two vertices of a common face.  Torus
graphs (L T1, M T2) for G1 and (L T1, M T2') for G2 are compared by nauty (labelg
canonical forms).  An isomorphism G1* ~ G2* gives p_c^site(G1) = p_c^site(G2) through the
matching-pair relation p_c(G) + p_c(G*) = 1.

usage: python analysis/matching_iso.py TAG1 TAG2 [L M ...]
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import networkx as nx

LABELG = os.environ.get('LABELG', 'labelg')  # nauty's labelg (McKay-Piperno); set LABELG if not on PATH


def load(tag):
    d = json.load(open(f'data/cells/{tag}.json'))
    T1, T2 = complex(*d['T1']), complex(*d['T2'])
    pos = [complex(*p) for p in d['pos']]
    nbr = {u: [] for u in range(d['nv'])}
    for u, w, a, b in d['edges']:
        nbr[u].append((w, a, b))
        nbr[w].append((u, -a, -b))
    return d, T1, T2, pos, nbr


def faces(d, T1, T2, pos, nbr):
    """faces as lists of (orbit, da, db) relative to the first vertex, by tracing darts."""
    def vec(u, w, a, b):
        return pos[w] + a * T1 + b * T2 - pos[u]
    rot = {u: sorted(nbr[u], key=lambda x: math.atan2(vec(u, *x).imag, vec(u, *x).real)) for u in nbr}
    seen, out = set(), []
    for u in nbr:
        for (w, a, b) in rot[u]:
            if (u, w, a, b) in seen:
                continue
            face, cu, ca, cb = [], u, 0, 0
            du = (w, a, b)
            while True:
                seen.add((cu, du[0], du[1], du[2]))
                face.append((cu, ca, cb))
                nu, na, nb_ = du[0], ca + du[1], cb + du[2]
                # next dart at nu: the neighbour just before the reversed dart in the rotation
                back = (cu, -du[1], -du[2])
                r = rot[nu]
                i = r.index(back)
                du = r[(i - 1) % len(r)]
                cu, ca, cb = nu, na, nb_
                if (cu, du[0], du[1], du[2]) in seen:
                    break
            if len(face) <= 12:  # skip the outer/unbounded artefacts (should not occur on a torus)
                out.append(face)
    return out


def torus_graph(tag, L, M, matching):
    d, T1, T2, pos, nbr = load(tag)
    nv = d['nv']
    vid = lambda u, a, b: ((b % M) * L + (a % L)) * nv + u
    G = nx.Graph()
    G.add_nodes_from(range(L * M * nv))
    for b0 in range(M):
        for a0 in range(L):
            for u, w, a, b in d['edges']:
                G.add_edge(vid(u, a0, b0), vid(w, a0 + a, b0 + b))
    if matching:
        fs = faces(d, T1, T2, pos, nbr)
        for b0 in range(M):
            for a0 in range(L):
                for f in fs:
                    vs = [vid(u, a0 + a, b0 + b) for u, a, b in f]
                    for i in range(len(vs)):
                        for j in range(i + 1, len(vs)):
                            if vs[i] != vs[j]:
                                G.add_edge(vs[i], vs[j])
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def canon(G):
    with tempfile.TemporaryDirectory() as td:
        fin = os.path.join(td, 'in.g6')
        open(fin, 'wb').write(nx.to_graph6_bytes(G, header=False))
        r = subprocess.run(['bash', '-c', f'{LABELG} -q "{fin}"'], capture_output=True, text=True)
        return r.stdout.strip()


def main():
    t1, t2 = sys.argv[1], sys.argv[2]
    sizes = list(map(int, sys.argv[3:])) or [2, 2, 3, 2, 4, 3]
    for L, M in zip(sizes[::2], sizes[1::2]):
        res = []
        for matching in (False, True):
            G1, G2 = torus_graph(t1, L, M, matching), torus_graph(t2, L, M, matching)
            same = canon(G1) == canon(G2)
            res.append(f"{'G*' if matching else 'G '}: {'ISOMORPHIC' if same else 'different'} "
                       f"(n={G1.number_of_nodes()}, m={G1.number_of_edges()}/{G2.number_of_edges()})")
        print(f'{L}x{M}: ' + ' | '.join(res), flush=True)


if __name__ == '__main__':
    main()


def torus_graph_sheared(tag, L, M, j, matching):
    """torus by the cell lattice {(L, 0), (j, M)}"""
    d, T1, T2, pos, nbr = load(tag)
    nv = d['nv']

    def vid(u, a, b):
        bb = b % M
        k = (b - bb) // M
        aa = (a - k * j) % L
        return (bb * L + aa) * nv + u
    G = nx.Graph()
    G.add_nodes_from(range(L * M * nv))
    fs = faces(d, T1, T2, pos, nbr) if matching else []
    for b0 in range(M):
        for a0 in range(L):
            for u, w, a, b in d['edges']:
                G.add_edge(vid(u, a0, b0), vid(w, a0 + a, b0 + b))
            for f in fs:
                vs = [vid(u, a0 + a, b0 + b) for u, a, b in f]
                for x in range(len(vs)):
                    for y in range(x + 1, len(vs)):
                        if vs[x] != vs[y]:
                            G.add_edge(vs[x], vs[y])
    G.remove_edges_from(nx.selfloop_edges(G))
    return G
