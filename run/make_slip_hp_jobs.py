"""High-precision site runs for slip pairs on identical tori (registry decay test).

Both members of a pair use the same Lx x Ly torus; the total CPU time per lattice is split
into JOBS independent jobs (different seeds); histograms are added in the analysis.

usage: python run/make_slip_hp_jobs.py CPU_HOURS_PER_LATTICE JOBS OUT_JOBFILE
"""
import hashlib
import sys

PAIRS = [  # (tag_a, tag_b, Lx, Ly): h = 2 and h = 3 clean-family slip pairs
    ('03__3__eu_4f_4k_4r_1', '03__3__eu_4f_4k_4s_1', 24, 12),
    ('03__3__eu_4f_4k_4u_1', '03__3__eu_4f_4k_4u_2', 25, 10),
]


def main():
    hours, njobs, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    secs = round(hours * 3600 / njobs)
    lines = []
    for j in range(njobs):  # interleave lattices so that partial results stay balanced
        for a, b, Lx, Ly in PAIRS:
            for tag in (a, b):
                name = f'{tag}__site__{Lx}x{Ly}__hp{j:02d}'
                seed = int(hashlib.sha1(name.encode()).hexdigest()[:12], 16)
                lines.append(f'data/cells_txt/{tag}.txt site {Lx} {Ly} {secs} {seed} results/slip_hp/{name}.txt')
    open(out, 'w', newline='\n').write('\n'.join(lines) + '\n')
    print(len(lines), 'jobs of', secs, 's ->', out, f'({len(lines) * secs / 3600:.0f} CPU-h)')


if __name__ == '__main__':
    main()
