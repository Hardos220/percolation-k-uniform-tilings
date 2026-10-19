"""Figures for the paper -> paper/figs/*.pdf

usage: python analysis/paper_figs.py [names...]   (default: all)
Inputs: results/tables/all_thresholds.csv, results/paper_numbers.json, results/paper_stats.json,
results/fss_summary.json, results/slip_hp_summary.json, results/slip_pairs_k6.json, results/band_tm.json
"""
import collections
import csv
import json
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from draw_tiling import COLORS, draw  # noqa: E402
from paper_stats import C_STAR, composition, euler_estimates, gm, LEBRECHT_A  # noqa: E402

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                     'lines.linewidth': 0.9, 'savefig.dpi': 300})
COL = 3.375
FULL = 7.0
OUT = 'paper/figs'


def rows():
    return list(csv.DictReader(open('results/tables/all_thresholds.csv')))


def label(ax, s, x=0.02, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, ha='left', va='top', fontsize=9, fontweight='bold')


def rot_of(tag):
    d = json.load(open(f'data/cells/{tag}.json'))
    return -math.atan2(d['T1'][1], d['T1'][0])


def fig_examples():
    tags = [('03__3__eu_3c_4h_4k_1', '(a) $k=3$'), ('09__2_81__eu_3a5_3b3_4c_1', '(b) $k=9$'),
            ('10__2_91__eu_5c_6i9_1', '(c) $k=10$')]
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.45))
    for ax, (tag, lab) in zip(axs, tags):
        draw(ax, tag, (-7, 7), (-6, 6), R=8, lw=0.35, rot=rot_of(tag))
        ax.set_title(lab, fontsize=8, loc='left')
    handles = [matplotlib.patches.Patch(fc=COLORS[n], ec='k', lw=0.4, label=f'{n}-gon') for n in (3, 4, 6, 12)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08, wspace=0.04)
    fig.savefig(f'{OUT}/fig_examples.pdf')


def fig_validation():
    nums = json.load(open('results/paper_numbers.json'))
    fss = json.load(open('results/fss_summary.json'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    groups = [('site', lambda e: e['mode'] == 'site', 'o', '#1f77b4'),
              ('bond', lambda e: e['mode'] == 'bond', 's', '#d62728')]
    for name, sel, mk, c in groups:
        es = [e for e in nums['validation']['entries'] if sel(e) and not e['trivial']]
        a.plot([e['ref'] for e in es], [e['z'] for e in es], mk, ms=4, mfc='none', mec=c, mew=0.8, label=name)
        es = [e for e in nums['validation']['entries'] if sel(e) and e['trivial']]
        a.plot([e['ref'] for e in es], [e['z'] for e in es], mk, ms=4, mfc=c, mec=c, mew=0.8,
               label=f'{name}, exact on every torus')
    a.axhspan(-1, 1, color='0.92', zorder=0)
    a.axhline(0, color='0.4', lw=0.5)
    a.set_xlabel('reference $p_c$')
    a.set_ylabel(r'$(p_c - p_c^{\rm ref})/\sigma$')
    a.set_ylim(-3.2, 3.2)
    label(a, '(a)', x=0.93)
    names = {'01__eu_4n_1_site': '$(4^4)$ site', '01__eu_4n_1_bond': '$(4^4)$ bond', '01__eu_6a_1_bond': '$(3^6)$ bond',
             '01__eu_3a_1_site': '$(3.12^2)$ site', '01__eu_3a_1_bond': '$(3.12^2)$ bond',
             '01__eu_3d_1_site': '$(6^3)$ site', '01__eu_488_bond': '$(4.8^2)$ bond',
             '01__eu_4f_1_bond': '$(3.6.3.6)$ bond'}
    mk = 'osD^v<>p'
    for i, (k, v) in enumerate(fss.items()):
        N = np.array([x['N'] for x in v])
        dv = np.array([x['dev'] for x in v]) * 1e6
        s = np.array([x['s'] for x in v]) * 1e6
        sq = k == '01__eu_4n_1_site'
        b.errorbar(N * (1 + 0.025 * (i - 3.5)), dv, s, fmt=mk[i] + ('-' if sq else ''), ms=3, mfc='none', lw=0.6,
                   elinewidth=0.5, capsize=0, label=names.get(k, k), color='k' if sq else None)
    b.set_xscale('log')
    b.set_yscale('symlog', linthresh=5, linscale=1.2)  # linear for |y| < 5, logarithmic beyond
    b.axhspan(-5, 5, color='0.93', zorder=0)  # the linear part of the axis
    for yb in (-5, 5):
        b.axhline(yb, color='0.6', lw=0.4, ls=':')
    b.axhline(0, color='0.4', lw=0.5)
    b.set_xlabel('torus size $N_s$ (sites)')
    b.set_ylabel(r'$p_N - p_c$  $(10^{-6})$')
    b.set_ylim(-200, 12)
    yt = [-100, -20, -5, 0, 5]
    b.set_yticks(yt)
    b.set_yticklabels([f'{t:d}'.replace('-', '−') for t in yt])
    b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), frameon=False, ncol=4, fontsize=6.3,
             columnspacing=1.0, handletextpad=0.3)
    label(b, '(b)', x=0.02, y=0.97)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), frameon=False, ncol=2, fontsize=6.3)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f'{OUT}/fig_validation.pdf', bbox_inches='tight', pad_inches=0.02)


def fig_atlas():
    R = rows()
    th = collections.defaultdict(dict)
    for r in R:
        th[r['tag']][r['mode']] = r
    ks = np.array([int(v['site']['k']) for v in th.values()])
    ps = np.array([float(v['site']['pc']) for v in th.values()])
    pb = np.array([float(v['bond']['pc']) for v in th.values()])
    z = np.array([float(v['site']['mean_degree']) for v in th.values()])
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6))
    order = np.argsort(-ks)
    sc = a.scatter(pb[order], ps[order], c=z[order], s=1.2, lw=0, cmap='viridis', rasterized=True)
    arch = ks == 1
    a.plot(pb[arch], ps[arch], 'o', ms=4, mfc='none', mec='r', mew=0.8, label='Archimedean')
    a.set_xlabel(r'$p_c^{\rm bond}$')
    a.set_ylabel(r'$p_c^{\rm site}$')
    a.legend(loc='lower right', frameon=False)
    cb = fig.colorbar(sc, ax=a, pad=0.01)
    cb.set_label(r'mean degree $\bar z$')
    label(a, '(a)')
    zz = np.linspace(3, 6, 100)
    b.scatter(z, ps, s=1.0, lw=0, color='#1f77b4', rasterized=True, label='site')
    b.scatter(z, pb, s=1.0, lw=0, color='#d62728', rasterized=True, label='bond')
    b.plot(zz, gm('site', zz), 'k--', lw=0.7, label='Galam–Mauger, $q=\\bar z$')
    b.plot(zz, gm('bond', zz), 'k--', lw=0.7)
    b.plot(zz, 2 / zz, 'k:', lw=0.8, label=r'$p_c^{\rm bond}=2/\bar z$')
    b.set_xlabel(r'mean degree $\bar z$')
    b.set_ylabel(r'$p_c$')
    b.legend(loc='upper right', frameon=False, markerscale=4)
    label(b, '(b)', x=0.03, y=0.08)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f'{OUT}/fig_atlas.pdf')


def fig_formulas():
    """Errors on the hold-out set k >= 5 (same set as Table IV, column 'k >= 5'); from analysis/paper_rules.py."""
    resid = np.load('results/paper_rules_resid.npy', allow_pickle=True).item()
    kk = np.array([int(t[:2]) for t in sorted({r['tag'] for r in rows()})])
    hold = kk >= 5
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    bins = np.linspace(-0.08, 0.04, 241)
    for ax, m, lab in zip(axs, ('site', 'bond'), ('(a) site', '(b) bond')):
        for key, name, c in (('Linear', 'linear rule, Eq. (7)', 'k'), ('Euler', 'Euler estimate $p^*$', '#1f77b4'),
                             ('GM', 'Galam–Mauger, $q=\\bar z$', '#d62728')):
            x = resid[m][key][hold]
            x = x[np.isfinite(x)]
            ax.hist(x, bins=bins, histtype='step', color=c, lw=0.9,
                    label=f'{name}: rms {np.sqrt((x * x).mean()):.4f}')
        ax.set_yscale('log')
        ax.set_xlabel(r'$p_c^{\rm est} - p_c$')
        ax.legend(loc='upper left', frameon=False, fontsize=6.3)
        ax.set_ylim(0.8, 3e4)
        ax.text(0.97, 0.95, lab, transform=ax.transAxes, ha='right', va='top', fontsize=8.5, fontweight='bold')
    axs[0].set_ylabel('number of tilings')
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f'{OUT}/fig_formulas.pdf')


def slip_family():
    """h, staggered tag, eclipsed tag, site (d, s), bond (d, s) with d = p(eclipsed) - p(staggered)."""
    core = json.load(open('results/core/report_r0.json'))
    pc = {(r['tag'], r['mode']): (r['pc'], r['spc']) for r in core}
    hp = json.load(open('results/slip_hp_summary.json'))
    fam = [(1, '02__2__eu_4f_4k_1', '02__2__eu_4f_4k_2'), (2, '03__3__eu_4f_4k_4r_1', '03__3__eu_4f_4k_4s_1'),
           (3, '03__3__eu_4f_4k_4u_1', '03__3__eu_4f_4k_4u_2'), (4, '04__3_211__eu_4f_4k_4u2_1', '04__3_211__eu_4f_4k_4u2_2')]
    out = []
    for h, a, b in fam:
        for t in (a, b):  # orientation check: eclipsed has T2 orthogonal to T1
            d = json.load(open(f'data/cells/{t}.json'))
            T1, T2 = np.array(d['T1']), np.array(d['T2'])
            proj = abs(T1 @ T2) / np.linalg.norm(T1)
            assert (proj < 1e-6) == (t == b), (t, proj)
        rec = dict(h=h, staggered=a, eclipsed=b)
        for m in ('site', 'bond'):
            (pa, sa), (pb, sb) = pc[(a, m)], pc[(b, m)]
            rec[m] = (pb - pa, math.hypot(sa, sb), 'core')
        key = f'h={h}'
        if key in hp:
            rec['site'] = (hp[key]['d'], hp[key]['sd'], 'hp')
        out.append(rec)
    return out


def fig_slip():
    fam = slip_family()
    json.dump(fam, open('results/slip_family.json', 'w'), indent=1)
    fig = plt.figure(figsize=(FULL, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.08)
    for i, (tag, lab) in enumerate(((fam[1]['staggered'], '(a) staggered, $h=2$'), (fam[1]['eclipsed'], '(b) eclipsed, $h=2$'))):
        ax = fig.add_subplot(gs[0, i])
        draw(ax, tag, (-5.5, 5.5), (-5.2, 5.2), R=7, lw=0.35, cell=False, rot=rot_of(tag))
        ax.set_title(lab, fontsize=8, loc='left')
    ax = fig.add_subplot(gs[0, 2])
    hh = np.linspace(1, 4, 50)
    for m, c, mk in (('site', '#1f77b4', 'o'), ('bond', '#d62728', 's')):
        h = np.array([r['h'] for r in fam])
        d = np.array([r[m][0] for r in fam])
        s = np.array([r[m][1] for r in fam])
        res = np.abs(d) > 2 * s
        pos, neg = res & (d > 0), res & (d < 0)
        ax.errorbar(h[pos], d[pos], s[pos], fmt=mk, color=c, ms=4, capsize=1.5, elinewidth=0.6, label=f'{m}, $\\Delta>0$')
        if neg.any():
            ax.errorbar(h[neg], -d[neg], s[neg], fmt=mk, color=c, mfc='white', ms=4, capsize=1.5, elinewidth=0.6,
                        label=f'{m}, $\\Delta<0$')
        ul = ~res  # not resolved: 2-sigma upper limit on |Delta|
        dx = -0.06 if m == 'site' else 0.06
        ax.errorbar(h[ul] + dx, np.abs(d[ul]) + 2 * s[ul], yerr=0.35 * (np.abs(d[ul]) + 2 * s[ul]), uplims=True, fmt='none',
                    ecolor=c, elinewidth=0.7, capsize=2, label=f'{m}, 2$\\sigma$ upper limit')
        if m == 'bond':  # single dominant kappa = pi mode: Delta(h) = Delta(1) lambda^(h-1), lambda(W -> inf) = 0.14-0.16
            ax.fill_between(hh, d[0] * 0.14 ** (hh - 1), d[0] * 0.16 ** (hh - 1), color=c, alpha=0.15, lw=0)
            ax.plot(hh, d[0] * 0.15 ** (hh - 1), '--', color=c, lw=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('square-band height $h$')
    ax.set_ylabel(r'$|\Delta p_c|$')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(3e-7, 3e-2)
    ax.legend(loc='upper right', frameon=False, fontsize=6.3)
    ax.text(0.03, 0.06, '(c)', transform=ax.transAxes, fontsize=9, fontweight='bold')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    fig.subplots_adjust(left=0.01, right=0.93, top=0.92, bottom=0.17)
    fig.savefig(f'{OUT}/fig_slip.pdf')


FIGS = dict(examples=fig_examples, validation=fig_validation, atlas=fig_atlas, formulas=fig_formulas, slip=fig_slip)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
        print('ok', name, flush=True)
