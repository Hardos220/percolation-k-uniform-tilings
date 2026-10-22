"""Build the arXiv submission bundle of the paper.

usage: python tools/make_arxiv.py [OUTDIR]   (default arxiv/; needs tectonic, or pdflatex and bibtex, on PATH)
writes OUTDIR/src/ (main.tex, main.bbl, refs.bib, tables, figures), OUTDIR/arxiv_submission.tar.gz, OUTDIR/main.pdf and
OUTDIR/arxiv_metadata.txt (title, abstract and comments for the submission form).
The only difference from the journal version is the Supplemental Material note: it points to the repository and the
Zenodo archive instead of the publisher's placeholder.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tarfile

PAPER = 'paper'
SM_ARXIV = ('See Supplemental Material, archived at Zenodo, '
            '\\href{https://doi.org/10.5281/zenodo.23533708}{doi:10.5281/zenodo.23533708} (directory paper/supplemental of '
            'the repository), for the site and bond thresholds of all 23\\,396 tilings with their uncertainties and '
            'compositions, and for the periodic graphs of the tilings')
REPO = 'https://github.com/Hardos220/percolation-k-uniform-tilings'
DOI = 'https://doi.org/10.5281/zenodo.23533708'


def compile_tex(work):
    if shutil.which('tectonic'):
        subprocess.run(['tectonic', '-X', 'compile', '--keep-intermediates', '--keep-logs', 'main.tex'], cwd=work,
                       check=True, stdout=subprocess.DEVNULL)
        return
    for cmd in (['pdflatex', '-interaction=nonstopmode', 'main.tex'], ['bibtex', 'main'],
                ['pdflatex', '-interaction=nonstopmode', 'main.tex'], ['pdflatex', '-interaction=nonstopmode', 'main.tex']):
        subprocess.run(cmd, cwd=work, check=True, stdout=subprocess.DEVNULL)


def tex_to_text(s):
    """Abstract and title for the arXiv form: inline math stays in $...$ (rendered by arXiv), text macros are removed."""
    s = re.sub(r'(\d)\\,(\d{3})', r'\1,\2', s).replace('\\,', ' ')
    s = re.sub(r'\\textit\{([^}]*)\}', r'\1', s)
    s = s.replace('--', '-').replace('~', ' ')
    return ' '.join(s.split())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'arxiv'
    work, src = os.path.join(out, 'build'), os.path.join(out, 'src')
    for d in (work, src):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(os.path.join(d, 'figs'))
    tex = open(os.path.join(PAPER, 'main.tex'), encoding='utf-8').read()
    inputs = re.findall(r'\\input\{([^}]+)\}', tex)
    graphics = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex)
    files = ['main.tex'] + inputs + graphics
    for f in files:
        shutil.copy(os.path.join(PAPER, f), os.path.join(work, f))
    bib = open(os.path.join(PAPER, 'refs.bib'), encoding='utf-8').read()
    bib, n = re.subn(r'(@misc\{SM,\s*note\s*=\s*)\{[^\n]*\}', lambda m: m.group(1) + '{' + SM_ARXIV + '}', bib)
    assert n == 1, 'SM entry not found'
    open(os.path.join(work, 'refs.bib'), 'w', encoding='utf-8', newline='\n').write(bib)
    compile_tex(work)
    log = open(os.path.join(work, 'main.log'), encoding='utf-8', errors='replace').read()
    assert not re.search(r'undefined (references|citations)|Citation .* undefined|Reference .* undefined', log), 'undefined'
    bbl = open(os.path.join(work, 'main.bbl'), encoding='utf-8').read()
    assert 'zenodo.23533708' in bbl and 'inserted by publisher' not in bbl
    assert all(ord(c) < 128 for c in bbl + tex), 'non-ASCII character in the source'
    files += ['main.bbl', 'refs.bib']
    for f in files:
        shutil.copy(os.path.join(work, f), os.path.join(src, f))
    tar = os.path.join(out, 'arxiv_submission.tar.gz')
    with tarfile.open(tar, 'w:gz') as tf:
        for f in sorted(files):
            tf.add(os.path.join(src, f), arcname=f)
    shutil.copy(os.path.join(work, 'main.pdf'), os.path.join(out, 'main.pdf'))

    title = tex_to_text(re.search(r'\\title\{(.*)\}', tex).group(1))
    abstract = tex_to_text(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1))
    nfig = len(re.findall(r'\\begin\{figure\*?\}', tex))
    ntab = len(re.findall(r'\\begin\{table\*?\}', tex))
    pages = None
    try:
        import fitz
        pages = fitz.open(os.path.join(out, 'main.pdf')).page_count
    except ImportError:
        pass
    comments = (f'{pages} pages, ' if pages else '') + (f'{nfig} figures, {ntab} tables. Code, data and raw histograms: '
                                                        f'{REPO} and {DOI}')
    meta = (f'Title:\n{title}\n\nAuthors:\nMaksim Elisov\n\nAbstract ({len(abstract)} characters, limit 1920):\n{abstract}\n\n'
            f'Comments:\n{comments}\n\nPrimary category: cond-mat.stat-mech\nCross-list: physics.comp-ph\n'
            f'License: arXiv.org perpetual, non-exclusive license 1.0\n')
    open(os.path.join(out, 'arxiv_metadata.txt'), 'w', encoding='utf-8', newline='\n').write(meta)
    print(meta)
    print(tar, os.path.getsize(tar), 'bytes;', len(files), 'files:', ' '.join(sorted(files)))


if __name__ == '__main__':
    main()
