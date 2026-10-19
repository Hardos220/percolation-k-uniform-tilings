"""Final tables.

core  (k <= 4): from results/core/report_r0.json (three tori, FSS fit)
atlas (k = 5..10): one torus; rounds r0 (results/atlas/cNN.zip) and r1 (results/atlas_r1/cNN.zip
       or directory) merged per threshold.
Writes results/tables/{core_k4.csv, atlas_k5_10.csv, all_thresholds.csv}; lattice metadata from
data/cells/*.json (k, nv, mean degree, vertex-type fractions, face fractions).
Workers: $RMQ_CPU.
"""
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
import zipfile
from collections import Counter
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import p_cross  # noqa: E402
from wrap import Hist  # noqa: E402

_zip = {}


def read_member(src, name):
    """src: zip path or directory."""
    if src.endswith('.zip'):
        if src not in _zip:
            _zip[src] = zipfile.ZipFile(src)
        z = _zip[src]
        try:
            return z.read(name).decode()
        except KeyError:
            return None
    p = os.path.join(src, name)
    return open(p).read() if os.path.exists(p) else None


def atlas_sources():
    """name -> list of (source, member)."""
    out = {}
    for rnd, base in (('r0', 'results/atlas'), ('r1', 'results/atlas_r1')):
        pattern = 'run/jobs_atlas_c[0-9][0-9].txt' if rnd == 'r0' else 'run/jobs_atlasr1_c[0-9][0-9].txt'
        for jf in sorted(glob.glob(pattern)):
            c = jf.split('_')[-1][:-4]
            src = f'{base}/{c}.zip' if os.path.exists(f'{base}/{c}.zip') else f'{base}/{c}'
            for line in open(jf):
                x = line.split()
                if not x:
                    continue
                name = os.path.basename(x[6])
                out.setdefault(name, []).append((src, name))
    return out


def work(item):
    name, srcs = item
    texts = [t for t in (read_member(s, m) for s, m in srcs) if t]
    if not texts:
        return name, None
    H = Hist(texts=texts)
    p, s = p_cross(H)
    return name, (p, s, H.M, H.N, len(texts))


def meta(tag):
    d = json.load(open(f'data/cells/{tag}.json'))
    nv = d['nv']
    vc = Counter('.'.join(map(str, c)) for c in d['vertex_configs'])
    faces = Counter()
    for c in d['vertex_configs']:
        for kk in c:
            faces[kk] += Fraction(1, kk)
    return dict(k=int(tag[:2]), tag=tag, nv=nv, ne=d['ne'], mean_degree=round(2 * d['ne'] / nv, 6),
                vertex_types=';'.join(f'{t}:{Fraction(n, nv)}' for t, n in sorted(vc.items())),
                faces_per_vertex=';'.join(f'{kk}:{float(v / nv):.6f}' for kk, v in sorted(faces.items())),
                cell_area=round(d['cell_area'], 6), sites_per_area=round(nv / d['cell_area'], 6))


def main():
    os.makedirs('results/tables', exist_ok=True)
    rows = []
    for r in json.load(open('results/core/report_r0.json')):
        m = meta(r['tag'])
        m.update(mode=r['mode'], pc=r['pc'], err=r['spc'], method='core: 3 tori, FSS fit',
                 chi2=round(r['chi2'], 3), samples=sum(s['M'] for s in r['sizes']),
                 tori=';'.join(f"{s['Lx']}x{s['Ly']}:{s['p']:.8f}({s['s']:.1e})" for s in r['sizes']))
        rows.append(m)
    keys = ['k', 'tag', 'mode', 'pc', 'err', 'method', 'chi2', 'samples', 'tori', 'nv', 'ne', 'mean_degree',
            'vertex_types', 'faces_per_vertex', 'cell_area', 'sites_per_area']
    with open('results/tables/core_k4.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print('core rows', len(rows), flush=True)
    items = sorted(atlas_sources().items())
    nw = int(os.environ.get('RMQ_CPU', '1'))
    prog = os.environ.get('RMQ_PROGRESS_FILE')
    arows = []
    with mp.Pool(nw) as pool:
        for i, (name, res) in enumerate(pool.imap_unordered(work, items, chunksize=64)):
            if res is None:
                print('MISSING', name, flush=True)
                continue
            p, s, M, N, nf = res
            tag, mode, size = name[:-4].rsplit('__', 2)
            m = meta(tag)
            m.update(mode=mode, pc=p, err=s, method=f'atlas: 1 torus, {nf} round(s)', chi2='', samples=M,
                     tori=f'{size}:{p:.8f}({s:.1e})')
            arows.append(m)
            if prog and i % 500 == 0:
                try:
                    open(prog, 'w').write(f'{i}/{len(items)} thresholds')
                except OSError:
                    pass
    arows.sort(key=lambda r: (r['k'], r['tag'], r['mode']))
    with open('results/tables/atlas_k5_10.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(arows)
    with open('results/tables/all_thresholds.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows + arows)
    print('atlas rows', len(arows))


if __name__ == '__main__':
    main()
