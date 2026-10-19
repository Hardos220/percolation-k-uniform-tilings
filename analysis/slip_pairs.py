"""Find slip pairs: same vertex composition and nv, translation lattices <T1, T2> and
<T1, T2'> with T2' = +-T2 +- T1/2 (mod T1), for k <= KMAX; report site/bond differences."""
import csv
import itertools
import json
import math
import sys
from collections import defaultdict

KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rows = [r for r in csv.DictReader(open('results/tables/all_thresholds.csv')) if int(r['k']) <= KMAX]
val = {(r['tag'], r['mode']): (float(r['pc']), float(r['err'])) for r in rows}
tags = sorted({r['tag'] for r in rows})
info = {}
for t in tags:
    d = json.load(open(f'data/cells/{t}.json'))
    info[t] = (complex(*d['T1']), complex(*d['T2']), d['nv'], next(r['vertex_types'] for r in rows if r['tag'] == t))
groups = defaultdict(list)
for t, (T1, T2, nv, comp) in info.items():
    groups[(comp, nv)].append(t)


def is_slip(a, b):
    T1a, T2a = info[a][:2]
    T1b, T2b = info[b][:2]
    for s1 in (1, -1):
        if abs(T1b - s1 * T1a) > 1e-6:
            continue
        for s2 in (1, -1):
            for h in (0.5, -0.5):
                x = (s2 * T2b - T2a - h * T1a) / T1a  # must be a real integer
                if abs(x.imag) < 1e-6 and abs(x.real - round(x.real)) < 1e-6:
                    return True
    return False


out = []
for key, ts in groups.items():
    for a, b in itertools.combinations(sorted(ts), 2):
        if is_slip(a, b):
            rec = dict(a=a, b=b, comp=key[0], nv=key[1])
            for m in ('site', 'bond'):
                (pa, sa), (pb, sb) = val[(a, m)], val[(b, m)]
                rec[m] = dict(d=pb - pa, s=math.hypot(sa, sb))
            out.append(rec)
json.dump(out, open(f'results/slip_pairs_k{KMAX}.json', 'w'), indent=1)
print(len(out), 'slip pairs with k <=', KMAX)
for r in sorted(out, key=lambda r: -abs(r['bond']['d'])):
    print(f"{r['a']:34s} {r['b']:34s} site {r['site']['d']*1e6:+8.1f}({r['site']['s']*1e6:4.1f})  bond {r['bond']['d']*1e6:+8.1f}({r['bond']['s']*1e6:4.1f})  [{r['comp']}]")
