"""Supplemental Material of the paper -> paper/supplemental/

  thresholds.csv     one row per tiling: site and bond thresholds with uncertainties, composition, geometry, tori
  cells.jsonl        one JSON object per tiling: periodic graph (T1, T2, vertex positions, edges, vertex types)
  slip_pairs.csv     the 299 slip pairs with k <= 6 and their threshold differences
  composition_groups.csv  groups of tilings with identical vertex-type composition
  README.txt         description of the columns and fields

usage: python analysis/make_supplement.py
"""
import collections
import csv
import json
import math
import os
import zipfile

OUT = 'paper/supplemental'

README = """Supplemental Material: site and bond percolation thresholds of all k-uniform tilings with k <= 10

thresholds.csv - one row per tiling (23,396 rows)
  k              uniformity (number of vertex orbits)
  id             identifier: k, vertex-type orbit structure and catalog file name
  catalog_file   path of the .tes file in the catalog https://github.com/zenorogue/tes-catalog
                 (directory files/tessellations/k-uniform/, commit 6772c7c6)
  composition    vertex types (cyclic sequence of polygon sizes) with their fractions of vertices
  n_v, n_e       vertices and edges per unit cell
  mean_degree    mean vertex degree 2 n_e / n_v
  cell_area      area of the unit cell (unit edge length)
  site, site_err site percolation threshold and its standard error (statistical; for k <= 4 the error
                 of the finite-size extrapolation, not rescaled by chi^2)
  bond, bond_err bond percolation threshold and its standard error
  method         'core: 3 tori, FSS fit' (k <= 4) or 'atlas: 1 torus, n round(s)' (k >= 5; the atlas values
                 carry no finite-size correction and their errors are statistical only)
  tori_site, tori_bond
                 per torus: Lx x Ly unit cells : finite-size estimate (error) : number of samples :
                 aspect ratio of the torus (shorter side / longer side, from Lx|T1| and Ly|T2|)
  chi2_site, chi2_bond
                 chi^2 of the three-torus fit (one degree of freedom), k <= 4 only

cells.jsonl - one JSON object per line, same order as thresholds.csv
  id, k, T1, T2        translation vectors (unit edge length)
  pos                  positions of the n_v vertices of the unit cell
  edges                [u, w, a, b]: edge from vertex u of cell (0,0) to vertex w of cell (a,b),
                       i.e. to the point pos[w] + a*T1 + b*T2
  vertex_configs       vertex type of each vertex (sizes of the polygons around it, in cyclic order)

slip_pairs.csv - pairs with the same unit-cell content and T2' = T2 +- T1/2 (k <= 6)
  id_a, id_b, composition, n_v, d_site, err_site, d_bond, err_bond   (d = p_c(b) - p_c(a))

composition_groups.csv - tilings with identical vertex-type composition (groups of two or more)
  composition, n_tilings, ids (separated by '|'), spread_site, spread_bond (max - min threshold)

atlas_check.csv - finite-size check of the atlas on a stratified sample of 130 tilings (260 thresholds)
  stratum        E = most elongated atlas tori, L = largest unit cells, R = random (5 per k),
                 S = largest fraction of (4^4) vertices (largest square domains)
  N1, p1, err1   atlas torus: number of sites (bonds), estimate from the atlas runs plus an extension run
  N2, p2, err2   near-square torus with 3.4-4.6 times as many sites
  delta, err_delta   p2 - p1 and its error (measures the finite-size bias of the atlas value)

Errors are one standard deviation.
"""


def fmt_err(e):
    return f'{e:.3g}'


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    core = {(r['tag'], r['mode']): r for r in json.load(open('results/core/report_r0.json'))}
    by = collections.defaultdict(dict)
    for r in rows:
        by[r['tag']][r['mode']] = r
    tags = sorted(by, key=lambda t: (int(by[t]['site']['k']), t))

    def tori(t, d, mode):
        r = by[t][mode]
        L1, L2 = math.hypot(*d['T1']), math.hypot(*d['T2'])
        out = []
        if (t, mode) in core:
            for s in core[(t, mode)]['sizes']:
                a, b = s['Lx'] * L1, s['Ly'] * L2
                out.append(f"{s['Lx']}x{s['Ly']}:{s['p']:.8f}({fmt_err(s['s'])}):{s['M']}:{min(a, b) / max(a, b):.3f}")
        else:
            torus = r['tori'].split(':')[0]
            Lx, Ly = map(int, torus.split('x'))
            a, b = Lx * L1, Ly * L2
            out.append(f"{torus}:{float(r['pc']):.8f}({fmt_err(float(r['err']))}):{r['samples']}:{min(a, b) / max(a, b):.3f}")
        return ';'.join(out)

    with open(f'{OUT}/thresholds.csv', 'w', newline='') as f, open(f'{OUT}/cells.jsonl', 'w') as g:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['k', 'id', 'catalog_file', 'composition', 'n_v', 'n_e', 'mean_degree', 'cell_area',
                    'site', 'site_err', 'bond', 'bond_err', 'method', 'tori_site', 'tori_bond', 'chi2_site', 'chi2_bond'])
        for t in tags:
            s, b = by[t]['site'], by[t]['bond']
            d = json.load(open(f'data/cells/{t}.json'))
            path = d['path'].replace('\\', '/').split('data/tes/', 1)[-1]
            w.writerow([s['k'], t, path, s['vertex_types'], s['nv'], s['ne'], s['mean_degree'], s['cell_area'],
                        f"{float(s['pc']):.8f}", fmt_err(float(s['err'])), f"{float(b['pc']):.8f}",
                        fmt_err(float(b['err'])), s['method'], tori(t, d, 'site'), tori(t, d, 'bond'),
                        s['chi2'], b['chi2']])
            g.write(json.dumps(dict(id=t, k=int(s['k']), T1=d['T1'], T2=d['T2'], pos=d['pos'], edges=d['edges'],
                                    vertex_configs=d['vertex_configs']), separators=(',', ':')) + '\n')
    with open(f'{OUT}/slip_pairs.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id_a', 'id_b', 'composition', 'n_v', 'd_site', 'err_site', 'd_bond', 'err_bond'])
        for p in json.load(open('results/slip_pairs_k6.json')):
            w.writerow([p['a'], p['b'], p['comp'], p['nv'], f"{p['site']['d']:.3e}", fmt_err(p['site']['s']),
                        f"{p['bond']['d']:.3e}", fmt_err(p['bond']['s'])])
    groups = collections.defaultdict(list)
    for t in tags:
        groups[by[t]['site']['vertex_types']].append(t)
    with open(f'{OUT}/composition_groups.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['composition', 'n_tilings', 'ids', 'spread_site', 'spread_bond'])
        for comp, ts in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(ts) < 2:
                continue
            sp = {m: max(float(by[t][m]['pc']) for t in ts) - min(float(by[t][m]['pc']) for t in ts) for m in ('site', 'bond')}
            w.writerow([comp, len(ts), '|'.join(ts), f"{sp['site']:.3e}", f"{sp['bond']:.3e}"])
    chk = json.load(open('results/atlas_check_summary.json'))['rows']
    with open(f'{OUT}/atlas_check.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id', 'stratum', 'k', 'mode', 'n_v', 'N1', 'p1', 'err1', 'N2', 'p2', 'err2', 'delta', 'err_delta'])
        for r in chk:
            w.writerow([r['id'], r['stratum'], r['k'], r['mode'], r['n_v'], r['N1'], f"{r['p1']:.8f}", fmt_err(r['s1']),
                        r['N2'], f"{r['p2']:.8f}", fmt_err(r['s2']), f"{r['d']:.3e}", fmt_err(r['sd'])])
    open(f'{OUT}/README.txt', 'w').write(README)
    with zipfile.ZipFile(f'{OUT}/supplemental_material.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('README.txt', 'thresholds.csv', 'cells.jsonl', 'slip_pairs.csv', 'composition_groups.csv',
                     'atlas_check.csv'):
            z.write(f'{OUT}/{name}', name)
    print(len(tags), 'tilings ->', OUT)


if __name__ == '__main__':
    main()
