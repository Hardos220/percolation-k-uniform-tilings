"""data/cells/*.json -> data/cells_txt/*.txt (input format of nz)."""
import glob, json, os
os.makedirs('data/cells_txt', exist_ok=True)
for f in glob.glob('data/cells/*.json'):
    d = json.load(open(f))
    with open(os.path.join('data/cells_txt', os.path.basename(f)[:-5] + '.txt'), 'w') as g:
        g.write(f"{d['nv']} {d['ne']}\n")
        for u, w, a, b in d['edges']:
            g.write(f"{u} {w} {a} {b}\n")
