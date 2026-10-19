"""Parallel .tes -> cell conversion for many tilings (atlas).

usage: python tiling/parse_many.py OUT_JSON_DIR OUT_TXT_DIR [--sym] tes_dir_or_files...
Workers: $RMQ_CPU.  Existing outputs are skipped.  Log: OUT_JSON_DIR/parse.log
"""
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tes2cell  # noqa: E402


def tag_of(path):
    rel = path.replace(os.sep, '/').split('data/tes/')[-1]
    return rel[:-4].replace('/', '__').replace(' ', '_').replace('(', '').replace(')', '')


def work(args):
    path, jdir, tdir, sym = args
    tag = tag_of(path)
    jf = os.path.join(jdir, tag + '.json')
    if os.path.exists(jf):
        return tag, 'skip', 0.0
    t0 = time.time()
    try:
        d = tes2cell.process(path, sym=sym)
    except Exception:
        return tag, 'ERROR ' + traceback.format_exc().strip().splitlines()[-1], time.time() - t0
    with open(os.path.join(tdir, tag + '.txt'), 'w') as g:
        g.write(f"{d['nv']} {d['ne']}\n")
        for u, w, a, b in d['edges']:
            g.write(f'{u} {w} {a} {b}\n')
    json.dump(d, open(jf + '.part', 'w'))
    os.replace(jf + '.part', jf)
    return tag, ('OK' if d['ok'] else 'BAD') + f" nv={d['nv']} euler={d['euler']}", time.time() - t0


def main():
    jdir, tdir = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    sym = '--sym' in rest
    rest = [r for r in rest if r != '--sym']
    files = []
    for r in rest:
        files += sorted(glob.glob(os.path.join(r, '**', '*.tes'), recursive=True)) if os.path.isdir(r) else [r]
    os.makedirs(jdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    nw = int(os.environ.get('RMQ_CPU', '1'))
    prog = os.environ.get('RMQ_PROGRESS_FILE')
    done = 0
    with open(os.path.join(jdir, 'parse.log'), 'a') as log, mp.Pool(nw) as pool:
        for tag, status, dt in pool.imap_unordered(work, [(f, jdir, tdir, sym) for f in files], chunksize=4):
            done += 1
            if status != 'skip':
                log.write(f'{status} {dt:.1f}s {tag}\n')
                log.flush()
            if prog and done % 50 == 0:
                try:
                    open(prog, 'w').write(f'{done}/{len(files)} tilings')
                except OSError:
                    pass
    print('done', done)


if __name__ == '__main__':
    main()
