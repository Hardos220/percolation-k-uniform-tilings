"""Run a list of nz jobs in parallel (one process per worker).

Job file: one job per line, "cell mode Lx Ly cpu_seconds seed outfile".
Workers: $RMQ_CPU (or --workers).  Progress: $RMQ_PROGRESS_FILE.
Jobs whose outfile exists are skipped, so an interrupted batch can be resumed.

usage: python run/batch.py JOBFILE [--workers K] [--zip DIR]
With --zip DIR: when every job is done, DIR is packed into DIR.zip and removed
(one file travels back from a remote PC instead of thousands).
"""
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import threading
import zipfile

NZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nz', 'nz.exe')


def main():
    jobfile = sys.argv[1]
    workers = int(os.environ.get('RMQ_CPU', '1'))
    if '--workers' in sys.argv:
        workers = int(sys.argv[sys.argv.index('--workers') + 1])
    jobs = [l.split() for l in open(jobfile) if l.strip() and not l.startswith('#')]
    todo = [j for j in jobs if not os.path.exists(j[6])]
    total, done = len(jobs), len(jobs) - len(todo)
    lock = threading.Lock()
    env = {k: v for k, v in os.environ.items() if k != 'RMQ_PROGRESS_FILE'}
    prog = os.environ.get('RMQ_PROGRESS_FILE')

    def report():
        if prog:
            try:
                with open(prog, 'w', encoding='utf-8') as f:
                    f.write(f'{done}/{total} jobs')
            except OSError:
                pass

    def run(j):
        nonlocal done
        cell, mode, Lx, Ly, secs, seed, out = j
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + '.part'
        r = subprocess.run([NZ, cell, mode, Lx, Ly, '2000000000000', seed, tmp, secs],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print('FAILED', ' '.join(j), r.stderr, flush=True)
            return
        os.replace(tmp, out)
        with lock:
            done += 1
            report()
            print(f'{done}/{total}', out, flush=True)

    report()
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))
    if '--zip' in sys.argv:
        zdir = sys.argv[sys.argv.index('--zip') + 1].rstrip('/')
        if all(os.path.exists(j[6]) for j in jobs):
            with zipfile.ZipFile(zdir + '.zip.part', 'w', zipfile.ZIP_DEFLATED) as z:
                for name in sorted(os.listdir(zdir)):
                    z.write(os.path.join(zdir, name), name)
            os.replace(zdir + '.zip.part', zdir + '.zip')
            shutil.rmtree(zdir)
            print('packed', zdir + '.zip', flush=True)
        else:
            print('not all jobs finished; outputs left unpacked', flush=True)


if __name__ == '__main__':
    main()
