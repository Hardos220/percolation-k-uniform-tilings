"""LaTeX tables and validation numbers for the paper.

usage: python analysis/paper_tables.py   -> paper/tab_*.tex, results/paper_numbers.json
"""
import collections
import csv
import json
import math
import re

import numpy as np

S18 = math.sin(math.pi / 18)
# Archimedean references: value, uncertainty (0 = exact), source key
ARCH = [  # tag, name (exponent notation), site ref, bond ref
    ('01__eu_3a_1', '3.12^2', (math.sqrt(1 - 2 * S18), 0, 'SudingZiff99'), (0.740420798850811610, 2e-18, 'ScullardJacobsen20')),
    ('01__eu_3c_1', '4.6.12', (0.7478008, 2e-7, 'Jacobsen14'), (0.693733124922, 2e-12, 'ScullardJacobsen20')),
    ('01__eu_488', '4.8^2', (0.7297232, 5e-7, 'Jacobsen14'), (0.6768031243900113, 3e-16, 'ScullardJacobsen20')),
    ('01__eu_3d_1', '6^3', (0.697040230, 5e-9, 'Jacobsen14'), (1 - 2 * S18, 0, 'SykesEssam64')),
    ('01__eu_4f_1', '3.6.3.6', (1 - 2 * S18, 0, 'SykesEssam64'), (0.52440499916744820, 1e-17, 'ScullardJacobsen20')),
    ('01__eu_4l_1', '3.4.6.4', (0.62181207, 7e-8, 'Jacobsen14'), (0.524831461573, 1e-12, 'ScullardJacobsen20')),
    ('01__eu_4n_1', '4^4', (0.59274605079210, 2e-14, 'Jacobsen15'), (0.5, 0, 'Kesten80')),
    ('01__eu_5b_1', '3^4.6', (0.579498, 3e-6, 'SudingZiff99'), (0.4343283172240, 6e-13, 'ScullardJacobsen20')),
    ('01__eu_5e_1', '3^2.4.3.4', (0.550806, 3e-6, 'SudingZiff99'), (0.4141378565917, 1e-13, 'ScullardJacobsen20')),
    ('01__eu_5c_1', '3^3.4^2', (0.550213, 3e-6, 'SudingZiff99'), (0.41964035886369, 2e-14, 'ScullardJacobsen20')),
    ('01__eu_6a_1', '3^6', (0.5, 0, 'SykesEssam64'), (2 * S18, 0, 'SykesEssam64')),
]


def vt_tex(t):
    """3.3.6.6 -> $3^2.6^2$ (consecutive repeats as exponents, as in Grunbaum-Shephard)."""
    parts = t.split('.')
    out = []
    i = 0
    while i < len(parts):
        j = i
        while j + 1 < len(parts) and parts[j + 1] == parts[i]:
            j += 1
        n = j - i + 1
        out.append(parts[i] + (f'^{{{n}}}' if n > 1 else ''))
        i = j + 1
    return '.'.join(out)


def name_tex(s):
    return re.sub(r'\^(\d+)', r'^{\1}', s)


def comp_tex(vt, plain=False):
    items = []
    for part in vt.split(';'):
        t, f = part.rsplit(':', 1)
        items.append((t, f))
    items.sort(key=lambda x: (-eval(x[1]), x[0]))
    if len(items) == 1:
        return f'$({vt_tex(items[0][0])})$'
    if plain:  # compact form for the long table: 2/5\,(3^2.4.12)
        return '$' + '+'.join(f'{f}\\,({vt_tex(t)})' for t, f in items) + '$'
    return '$' + '+'.join(f'\\tfrac{{{f.split("/")[0]}}}{{{f.split("/")[1]}}}({vt_tex(t)})' for t, f in items) + '$'


def fmt(p, s, dec=7):
    """0.5927461(27): value with dec decimals, uncertainty in units of the last digit."""
    u = s * 10 ** dec
    if u < 9.5:
        return f'{p:.{dec}f}({u:.0f})'
    if u >= 99.5:
        return fmt(p, s, dec - 1)
    return f'{p:.{dec}f}({u:.0f})'


def fmt_ref(v, s):
    if s == 0 and v == 0.5:
        return '1/2'
    if s == 0:
        return f'{v:.9f}' + r'\ldots'
    dec = max(0, -math.floor(math.log10(s)))
    dec = min(dec, 10)
    if s < 10 ** -10:
        return f'{v:.10f}' + r'\ldots'
    return f'{v:.{dec}f}({round(s * 10 ** dec):.0f})'


def main():
    rows = list(csv.DictReader(open('results/tables/all_thresholds.csv')))
    th = {(r['tag'], r['mode']): r for r in rows}
    nums = {}

    # ---- Table: counts and precision per k
    by = collections.defaultdict(list)
    for r in rows:
        by[(int(r['k']), r['mode'])].append(float(r['err']))
    lines = []
    for k in range(1, 11):
        n = len(by[(k, 'site')])
        es, eb = np.median(by[(k, 'site')]), np.median(by[(k, 'bond')])
        meth = r'3 tori, extrap.' if k <= 4 else '1 torus'
        lines.append(f'{k} & {n} & {meth} & {es * 1e6:.1f} & {eb * 1e6:.1f} \\\\')
    open('paper/tab_counts.tex', 'w').write('\n'.join(lines) + '\n')
    nums['n_tilings'] = len({r['tag'] for r in rows})
    nums['n_thresholds'] = len(rows)

    # ---- validation against the 22 Archimedean references (exact values and primary sources)
    trivial = {('01__eu_6a_1', 'site'), ('01__eu_4n_1', 'bond')}  # self-matching / self-dual: exact on every torus
    zs = []
    lines = []
    for tag, name, sref, bref in ARCH:
        cells = [f'$({name_tex(name)})$']
        for mode, (v, s, src) in (('site', sref), ('bond', bref)):
            r = th[(tag, mode)]
            p, e = float(r['pc']), float(r['err'])
            z = (p - v) / math.hypot(e, s)
            zs.append(dict(tag=tag, mode=mode, z=z, src=src, ours=p, err=e, ref=v, refs=s, trivial=(tag, mode) in trivial))
            mark = r'$^{\dagger}$' if (tag, mode) in trivial else ''
            cells += [fmt(p, e), fmt_ref(v, s) + f'~\\cite{{{src}}}' + mark, f'{z:+.1f}']
        lines.append(' & '.join(cells) + r' \\')
    open('paper/tab_arch.tex', 'w').write('\n'.join(lines) + '\n')
    zz = np.array([d['z'] for d in zs])
    znt = np.array([d['z'] for d in zs if not d['trivial']])
    nums['validation'] = dict(n=len(zz), mean=float(zz.mean()), rms=float(np.sqrt((zz ** 2).mean())),
                              maxabs=float(np.abs(zz).max()), n_gt2=int((np.abs(zz) > 2).sum()),
                              n_nontrivial=len(znt), rms_nontrivial=float(np.sqrt((znt ** 2).mean())),
                              mean_nontrivial=float(znt.mean()), entries=zs)

    # ---- 2-uniform table: our site and bond values, site values of Neher et al. (supplement, Table 4)
    two = json.load(open('reference/two_uniform_reference.json'))
    geom = {'02__2__eu_4f_4k_1': 'staggered', '02__2__eu_4f_4k_2': 'eclipsed'}
    # Neher et al. list both values for this composition without distinguishing the lattices; Lebrecht et al.
    # (Table 2) pair 0.6286 with bond 0.5189 and 0.6279 with 0.5177, i.e. with our eclipsed and staggered tilings
    pair16 = {'02__2__eu_4f_4k_2': '0.6286(2)', '02__2__eu_4f_4k_1': '0.6279(2)'}
    # four-decimal bond values of Lebrecht et al. (2021), Table 2 (results/review3_checks.json)
    leb = {t: v['bond_lebrecht'] for t, v in json.load(open('results/review3_checks.json'))['lebrecht_bond']['rows'].items()}
    lines2 = []
    neher_dev = []
    for x in two:
        tag = x['cell'][:-5]
        ps, es = float(th[(tag, 'site')]['pc']), float(th[(tag, 'site')]['err'])
        pb, eb = float(th[(tag, 'bond')]['pc']), float(th[(tag, 'bond')]['err'])
        nm = tag.split('__eu_')[1].replace('_', ' ')
        if tag in geom:
            sstr = pair16[tag] + r'$^{\ast}$'
            nm += f' ({geom[tag]})'
        else:
            sstr = x['site'][0]
            msn = re.match(r'([\d.]+)\((\d+)\)', sstr)
            neher_dev.append(dict(tag=tag, ours=ps, neher=float(msn.group(1)), dev=ps - float(msn.group(1))))
        lines2.append((ps, f'\\texttt{{{nm}}} & {comp_tex(th[(tag, "site")]["vertex_types"])} & {fmt(ps, es)} & {sstr} & '
                           f'{fmt(pb, eb)} & {leb[tag]:.4f} \\\\'))
    lines2.sort(key=lambda t: -t[0])
    open('paper/tab_2uni.tex', 'w').write('\n'.join(l for _, l in lines2) + '\n')
    nd = np.array([d['dev'] for d in neher_dev])
    sig_n = 2e-4
    nums['neher_site'] = dict(n=len(nd), rms=float(np.sqrt((nd ** 2).mean())), maxabs=float(np.abs(nd).max()),
                              n_within_1sig=int((np.abs(nd) <= sig_n).sum()), n_within_2sig=int((np.abs(nd) <= 2 * sig_n).sum()),
                              chi2_per_dof_quoted=float(np.mean((nd / sig_n) ** 2)), entries=neher_dev)

    # ---- 3-uniform table (61)
    tri = sorted([r for r in rows if r['k'] == '3' and r['mode'] == 'site'], key=lambda r: -float(r['pc']))
    lines3 = []
    for i, r in enumerate(tri, 1):
        tag = r['tag']
        rb = th[(tag, 'bond')]
        nm = tag.split('__eu_')[1].replace('_', ' ')
        lines3.append(f'{i} & \\texttt{{{nm}}} & {comp_tex(r["vertex_types"], plain=True)} & {float(r["mean_degree"]):.3f} & '
                      f'{fmt(float(r["pc"]), float(r["err"]))} & {fmt(float(rb["pc"]), float(rb["err"]))} \\\\')
    open('paper/tab_3uni_a.tex', 'w').write('\n'.join(lines3[:31]) + '\n')  # two floats: one page is too short
    open('paper/tab_3uni_b.tex', 'w').write('\n'.join(lines3[31:]) + '\n')
    nums['n_3uniform'] = len(tri)
    json.dump(nums, open('results/paper_numbers.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in nums.items() if k not in ('validation', 'neher_site')}, indent=1))
    print('validation', {k: v for k, v in nums['validation'].items() if k != 'entries'})
    for d in zs:
        if abs(d['z']) > 1.8:
            print('  ', d)
    print('neher', {k: v for k, v in nums['neher_site'].items() if k != 'entries'})
    for d in neher_dev:
        print('   %s ours %.6f neher %.4f dev %+.1e' % (d['tag'], d['ours'], d['neher'], d['dev']))


if __name__ == '__main__':
    main()
