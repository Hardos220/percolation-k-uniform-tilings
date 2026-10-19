"""Write an nz job list.

Torus = (q L) x (p L) cells with q/p ~ |T2|/|T1| (small integers), so that the
torus is close to a square in the plane; L is the base size.

usage: python run/make_jobs.py OUT GLOB MODES LIST_OF_L CPU_SECONDS OUTDIR
  e.g. python run/make_jobs.py run/jobs.txt 'data/cells/01__*.json' site,bond 16,32 200 results/pilot/A
"""
import glob, hashlib, json, math, os, sys

SHAPES = [(q, p) for p in (1, 2, 3) for q in range(1, 8) if math.gcd(q, p) == 1]


def shape(d):
    r = math.hypot(*d['T2']) / math.hypot(*d['T1'])
    return min(SHAPES, key=lambda qp: (abs(math.log(qp[0] / qp[1] / r)), qp[1]))


def main():
    out, pat, modes, Ls, secs, outdir = sys.argv[1:7]
    lines = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        tag = os.path.basename(f)[:-5]
        q, p = shape(d)
        for mode in modes.split(','):
            for L in map(int, Ls.split(',')):
                Lx, Ly = q * L, p * L  # T1 is the short vector: more cells along it
                res = f'{outdir}/{tag}__{mode}__{Lx}x{Ly}.txt'
                seed = int(hashlib.sha1(res.encode()).hexdigest()[:12], 16)
                lines.append(f'data/cells_txt/{tag}.txt {mode} {Lx} {Ly} {secs} {seed} {res}')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print(len(lines), 'jobs ->', out)


if __name__ == '__main__':
    main()
