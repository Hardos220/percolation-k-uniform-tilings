"""Match the 20 2-uniform cells to the Wikipedia 2-uniform table (numbers #11..#42)
by vertex-type fractions, and write reference/two_uniform_reference.json."""
import glob, json, os, re, sys
from fractions import Fraction
sys.path.insert(0, 'tiling')
from tes2cell import canon

wiki = open('reference/wikipedia_percolation_threshold_2026-10-16.wiki', encoding='utf-8').read()
sec = wiki[wiki.index('=== 2-uniform lattices ==='):wiki.index('=== Inhomogeneous 2-uniform lattice ===')]
rows = sec.split('|-')[1:]

def parse_cfg(s):
    s = re.sub(r'<sup>(\d+)</sup>', r'^\1', s).replace(' ', '')
    out = []
    for tok in s.split(','):
        if '^' in tok:
            b, e = tok.split('^'); out += [int(b)] * int(e)
        else:
            out.append(int(tok))
    return canon(out)

ref = []
for r in rows:
    cells = [c.strip() for c in r.strip().split('\n|') if c.strip()]
    cells[0] = cells[0].lstrip('| ')
    if not cells or not cells[0].isdigit():
        continue
    num = int(cells[0])
    fr = {}
    for m in re.finditer(r'\(\{\{frac\|(\d+)\|(\d+)\}\}\)\(([^)]*)\)', cells[1]):
        fr[parse_cfg(m.group(3))] = Fraction(int(m.group(1)), int(m.group(2)))
    site = re.match(r'([\d.]+\(\d+\))', cells[4]).group(1)
    bond = re.match(r'([\d.]+\(\d+\))', cells[5]).group(1)
    star = '*' in cells[1] or "'" in cells[1]
    ref.append(dict(num=num, fractions={'.'.join(map(str, k)): str(v) for k, v in fr.items()},
                    site=site, bond=bond, raw=cells[1]))

out = []
for f in sorted(glob.glob('data/cells/02__*.json')):
    d = json.load(open(f))
    tot = sum(d['config_counts'].values())
    fr = {'.'.join(map(str, canon([int(x) for x in k.split('.')]))): str(Fraction(v, tot))
          for k, v in d['config_counts'].items()}
    cand = [r for r in ref if r['fractions'] == fr]
    out.append(dict(cell=os.path.basename(f), fractions=fr, wiki=[c['num'] for c in cand],
                    site=[c['site'] for c in cand], bond=[c['bond'] for c in cand]))
    print(f"{out[-1]['cell']:28s} {fr}  -> {out[-1]['wiki']}")
print(len(ref), 'wiki rows')
json.dump(out, open('reference/two_uniform_reference.json', 'w'), indent=1)
