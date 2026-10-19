"""Draw a periodic tiling from its cell file (data/cells/<tag>.json): faces coloured by polygon size.

Faces are traced on a patch of cells from the planar rotation system (next edge clockwise).
usage (test): python analysis/draw_tiling.py TAG OUT.png
"""
import json
import math
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

COLORS = {3: '#f2c14e', 4: '#7fb7be', 6: '#d36b5f', 8: '#9a8fbf', 12: '#8cbf7f'}


def patch(tag, R=6):
    d = json.load(open(f'data/cells/{tag}.json'))
    T1, T2 = d['T1'], d['T2']
    pos = {}
    adj = {}
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for v, (x, y) in enumerate(d['pos']):
                pos[(v, i, j)] = (x + i * T1[0] + j * T2[0], y + i * T1[1] + j * T2[1])
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for u, w, da, db in d['edges']:
                a, b = (u, i, j), (w, i + da, j + db)
                if b in pos:
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)
    return d, pos, adj


def faces(pos, adj, max_sides=12):
    order = {}
    for v, nb in adj.items():
        x0, y0 = pos[v]
        order[v] = sorted(nb, key=lambda w: math.atan2(pos[w][1] - y0, pos[w][0] - x0))
    seen = set()
    out = []
    for u in adj:
        for w in adj[u]:
            if (u, w) in seen:
                continue
            cyc = [u]
            a, b = u, w
            ok = True
            while True:
                seen.add((a, b))
                nb = order[b]
                k = nb.index(a)
                c = nb[(k - 1) % len(nb)]  # next edge clockwise from the reverse direction
                a, b = b, c
                if a == u and b == w:
                    break
                cyc.append(a)
                if len(cyc) > max_sides:
                    ok = False
                    break
            if ok and len(cyc) >= 3:
                pts = [pos[v] for v in cyc]
                area = 0.5 * sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))
                if abs(area) > 1e-9:
                    out.append(pts)
    return out


def draw(ax, tag, xlim, ylim, R=6, cell=True, lw=0.5, rot=0.0):
    d, pos, adj = patch(tag, R)
    c, s = math.cos(rot), math.sin(rot)
    rotp = lambda p: (c * p[0] - s * p[1], s * p[0] + c * p[1])
    pos = {k: rotp(v) for k, v in pos.items()}
    for pts in faces(pos, adj):
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if not (xlim[0] - 3 < cx < xlim[1] + 3 and ylim[0] - 3 < cy < ylim[1] + 3):
            continue
        ax.add_patch(Polygon(pts, closed=True, fc=COLORS.get(len(pts), '#cccccc'), ec='k', lw=lw))
    if cell:
        T1, T2 = rotp(d['T1']), rotp(d['T2'])
        p0 = rotp(d['pos'][0])
        best = min(((p0[0] + i * T1[0] + j * T2[0], p0[1] + i * T1[1] + j * T2[1])
                    for i in range(-R, R + 1) for j in range(-R, R + 1)),
                   key=lambda o: (o[0] + (T1[0] + T2[0]) / 2 - (xlim[0] + xlim[1]) / 2) ** 2
                   + (o[1] + (T1[1] + T2[1]) / 2 - (ylim[0] + ylim[1]) / 2) ** 2)
        o = best  # cell corner on a vertex, cell centred in the window
        q = [o, (o[0] + T1[0], o[1] + T1[1]), (o[0] + T1[0] + T2[0], o[1] + T1[1] + T2[1]),
             (o[0] + T2[0], o[1] + T2[1])]
        ax.add_patch(Polygon(q, closed=True, fill=False, ec='#1f3b73', lw=1.4, ls='--'))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    return d


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(4, 4))
    draw(ax, sys.argv[1], (-6, 6), (-6, 6))
    fig.savefig(sys.argv[2], dpi=150, bbox_inches='tight')
