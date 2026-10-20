"""Pack the raw histograms (not tracked in git) into release assets.

usage: python tools/pack_raw.py OUTDIR
writes OUTDIR/raw_core.tar.gz, raw_atlas.tar, raw_checks.tar.gz and RAW_README.txt.
Every histogram file is the output of nz/nz.c, gpu/nz_gpu.cu, nz/nz_pair.c or gpu/nz_pair_gpu.cu:
a header line with the cell, mode, torus, numbers of elements and samples and the seed, then lines
"n count_h count_v count_min count_max count_cross" (sparse histograms of the occupation numbers at the first
wrapping along T1, along T2, in any direction, in both directions, and at the first cross configuration).
"""
import os
import sys
import tarfile

PARTS = {
    'raw_core.tar.gz': ['results/core'],
    'raw_atlas.tar': ['results/atlas', 'results/atlas_r1'],
    'raw_checks.tar.gz': ['results/atlas_check', 'results/fss', 'results/slip_hp', 'results/slip_coupled', 'results/core_extra'],
}
SKIP = ('.json', '.log', '.npy')

README = """Raw histograms for "Site and bond percolation thresholds of all k-uniform tilings with k <= 10"
(repository https://github.com/Hardos220/percolation-k-uniform-tilings).

raw_core.tar.gz    core, k <= 4: three tori per threshold (results/core/<batch or job>/...; file names
                   <tag>__<site|bond>__<Lx>x<Ly>__r<round>.txt; histograms of the same torus are added)
raw_atlas.tar      atlas, k = 5-10: one torus per threshold, zip archives per chunk; results/atlas = first round,
                   results/atlas_r1 = second round (k = 5-7)
raw_checks.tar.gz  results/atlas_check (finite-size check of 200 atlas thresholds), results/fss (finite-size study of
                   eight Archimedean thresholds), results/slip_hp (direct site runs of the slip family, h = 2, 3),
                   results/slip_coupled (coupled runs of slip pairs: <tag>__<mode>__<Lx>x<Ly>[__gpuX]__run<k>_{E,S}.txt,
                   E = eclipsed, S = staggered member on the same torus), results/core_extra (additional runs)

File format: one header line "# cell ... mode ... Lx ... Ly ... nv ... ne ... N ... samples ... seed ...", one comment
line, then "n count_h count_v count_min count_max count_cross": the number of samples in which the first cluster wrapping
along T1 (h), along T2 (v), in any direction (min), in both directions (max), and the first cross configuration (x)
appeared when n elements were occupied. N is the number of elements (sites or bonds) of the torus. The scripts
analysis/core_report.py, analysis/build_tables.py, analysis/atlas_rounds.py, analysis/atlas_check.py,
analysis/slip_hp_summary.py and analysis/slip_coupled_summary.py read these files (analysis/wrap.py: class Hist).
Code: MIT; data: CC BY 4.0.
"""


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, dirs in PARTS.items():
        mode = 'w:gz' if name.endswith('.gz') else 'w'
        n = 0
        with tarfile.open(os.path.join(out, name), mode) as tf:
            for d in dirs:
                if not os.path.isdir(d):
                    continue
                for root, _, files in os.walk(d):
                    for f in sorted(files):
                        if f.endswith(SKIP) or f.endswith('.part'):
                            continue
                        p = os.path.join(root, f)
                        tf.add(p, arcname=p.replace(os.sep, '/'))
                        n += 1
        print(name, n, 'files', round(os.path.getsize(os.path.join(out, name)) / 2**20, 1), 'MB')
    open(os.path.join(out, 'RAW_README.txt'), 'w', newline='\n').write(README)


if __name__ == '__main__':
    main()
