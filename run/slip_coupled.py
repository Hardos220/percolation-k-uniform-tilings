"""Coupled runs of a slip pair: K independent nz_pair runs in parallel (workers = $RMQ_CPU).

usage: python run/slip_coupled.py ECLIPSED_TAG site|bond Lx Ly K SAMPLES_PER_RUN OUTDIR [FIRST_RUN]
Cells: data/coupled/ECLIPSED_TAG_{E,S}.txt (analysis/slip_coupled_cells.py).  Ly must be even: both members are
simulated on the same torus <Lx T1, Ly T2> (twisted boundary for the staggered member, TWIST = s Ly/2).
Writes OUTDIR/<tag>__<mode>__<Lx>x<Ly>__run<k>_{E,S}.txt; runs whose outputs exist are skipped.
Seeds: first 12 hex digits of sha1 of the run name.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys

NZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nz', 'nz_pair.exe')


def one(args):
    tag, mode, Lx, Ly, k, M, out = args
    name = f'{tag}__{mode}__{Lx}x{Ly}__run{k:03d}'
    fe, fs = os.path.join(out, name + '_E.txt'), os.path.join(out, name + '_S.txt')
    if os.path.exists(fe) and os.path.exists(fs):
        return name, 'skip'
    seed = int(hashlib.sha1(name.encode()).hexdigest()[:12], 16)
    twist = json.load(open(f'data/coupled/{tag}.json'))['s'] * Ly // 2
    r = subprocess.run([NZ, f'data/coupled/{tag}_E.txt', f'data/coupled/{tag}_S.txt', mode, str(Lx), str(Ly), str(M),
                        str(seed), fe + '.part', fs + '.part', str(twist)], capture_output=True, text=True)
    if r.returncode != 0:
        return name, 'FAILED ' + r.stderr.strip()[-200:]
    os.replace(fe + '.part', fe)
    os.replace(fs + '.part', fs)
    return name, 'ok'


def main():
    tag, mode, Lx, Ly, K, M, out = sys.argv[1:8]
    first = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    Lx, Ly, K, M = int(Lx), int(Ly), int(K), int(M)
    assert Ly % 2 == 0, 'Ly must be even'
    os.makedirs(out, exist_ok=True)
    nw = int(os.environ.get('RMQ_CPU', '1'))
    prog = os.environ.get('RMQ_PROGRESS_FILE')
    jobs = [(tag, mode, Lx, Ly, k, M, out) for k in range(first, first + K)]
    done = 0
    with cf.ThreadPoolExecutor(nw) as ex:
        for name, st in ex.map(one, jobs):
            done += 1
            print(st, name, flush=True)
            if prog:
                try:
                    open(prog, 'w').write(f'{done}/{len(jobs)} runs')
                except OSError:
                    pass


if __name__ == '__main__':
    main()
