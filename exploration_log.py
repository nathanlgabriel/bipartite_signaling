"""User-readable progress reports for parameter sweeps (PSA_poster convention).

While a sweep runs, <project>/CURRENT_exploration_<desc>.md is rewritten after every finished point. It says what is
being explored and why, the settings, a summary table, one entry per finished point, progress and ETA. When the sweep
ends, the file moves to <project>/archives/PAST_NN_exploration_<desc>.md (NN = next free number). If a sweep stops
early (error / kill), the CURRENT file stays in place and says so; re-running the same command resumes it.
Reports are Markdown since 2026-11-14 (user request); convert_txt_report() converts the older plain-text ones.
"""
import datetime as dt
import os
import re
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_TYPE = {0: 'Umma', 1: 'Kish', 2: 'Akkad'}

LEGEND = """HOW TO READ A POINT BLOCK
  BIPARTITE   runs where one preference type ends stably split between two strategy profiles that earn it different
              payoffs: the smaller half holds >= 10% of the type, the split held for the last 10,000 steps, and no
              agent can gain more than 0.001 per interaction by switching (epsilon-Nash). Listed by split label:
              <type> [<half> | <half>] says how the two halves behave. 'X-in' = the half coordinates on action X
              within itself; 'blind' = it sends the null signal (treats everyone alike); 'Kish-copy'/'Umma-copy' =
              it behaves exactly like the other Sumerian type. The halves coordinate on the split type's favourite
              (Umma B, Kish S) with each other. Kish [B-in | B-in] = Kish 50/50 cross-matching; [E-in | copy] =
              Appendix G (xv)/(xvi).
  RECURRENT   runs not counted as BIPARTITE (the split did not hold for the whole last 10,000 steps) in which one
              type was nevertheless split in >= 50% of the snapshots of the second half of the run: the split keeps
              collapsing for a moment (one half takes over) and re-forming (the missing half re-invades, because it does
              better when rare). '>=' = lower bound (older files kept time series only for runs split at the end).
  balanced-split eq.  noise-robust alternative to the epsilon-Nash test: split stable, its share within 0.05 of a
              stable balance point, and no deviation other than switching halves gains > 0.001 per interaction
              (matters at small N / high mutation, where the state jitters around the balance point)
  raw splits  final-step splits before the stability + Nash filters (mostly runs caught mid-cycle)
  neutral     splits that only matter to partners, not to the splitting type's own payoff (selection-neutral drift)
  unclass.    runs whose modal outcome table matches none of the 17 known tables
  outcomes    the most common modal outcome tables (Appendix G labels) and their share of runs"""


def now():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M')


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial rate."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def split_labels(d):
    """Structural split label per run (psa_bipartite_tools.split_label), read from the file or, for files written
    before the field existed, recomputed from the stored final counts of the flagged runs."""
    if 'split_label' in d:
        return d['split_label'].astype(str)
    import psa_bipartite_tools as T
    M = T.point_matrix(d)
    out = np.array([''] * int(d['runs']), dtype=object)
    for r in np.flatnonzero(d['flagged']):
        out[r] = T.split_label(d['final_counts'][r], T.detect_bipartite(d['final_counts'][r], M, float(d['split_threshold'])))
    return out


def point_report(path):
    """Summary of one screen_point .npz (see psa_bipartite_tools.screen_point)."""
    d = np.load(path, allow_pickle=True)
    labels = d['labels'].astype(str)
    b = np.flatnonzero(d['bipartite'])
    runs = int(d['runs'])
    m0 = float(d['mutation_rate'][0]) if 'mutation_rate' in d else 0.01
    c = round(-float(d['signal_cost']), 6) if 'signal_cost' in d else 0.01
    e = round(float(d['edm_payoff']), 6) if 'edm_payoff' in d else 0.5
    nonull = 'null_signal' in d and not bool(d['null_signal'])
    bal = int(d['bipartite_bal'].sum()) if 'bipartite_bal' in d else None
    rec = rec_fam = rec_lb = None
    if 'recurrent' in d:
        rr = np.flatnonzero(d['recurrent'])
        rec, rec_fam, rec_lb = len(rr), Counter(d['rec_label'].astype(str)[rr]), bool(d['recurrent_lower_bound'])
    try:  # persistent criterion (Q7): split in >= 23 of the last 25 snapshots, no two misses in a row, final split eps-Nash
        import psa_bipartite_tools as T
        pers = int(T.persistent_runs(d).sum())
    except Exception:
        pers = None
    return dict(rec=rec, rec_families=rec_fam, rec_lb=rec_lb, persistent=pers,N=int(d['N']), alpha=float(d['alpha']), beta=float(d['beta']), runs=runs, bipartite=len(b), m0=m0, c=c, e=e, bal=bal, nonull=nonull,
                families=Counter(split_labels(d)[r] for r in b),
                flagged=int(d['flagged'].sum()), neutral=int(d['neutral'].sum()), other=int((labels == 'other').sum()),
                top=[(k, v / runs) for k, v in Counter(labels).most_common(3)], file=os.path.relpath(path, ROOT))


def format_point(p, minutes=None, note=''):
    fam = ', '.join(f'{k} x{v}' for k, v in p['families'].most_common()) or '-'
    top = ', '.join(f'{k} {v:.0%}' for k, v in p['top'])
    t = f'   [{minutes:.1f} min]' if minutes is not None else ''
    m0 = f"  m0={p['m0']:g}" if p.get('m0', 0.01) != 0.01 else ''
    m0 += f"  c={p['c']:g}" if p.get('c', 0.01) != 0.01 else ''
    m0 += f"  e={p['e']:g}" if p.get('e', 0.5) != 0.5 else ''
    m0 += '  NO-NULL model' if p.get('nonull') else ''
    bal = f"   balanced-split eq. {p['bal']}" if p.get('bal') is not None else ''
    bal += f"   persistent {p['persistent']}" if p.get('persistent') is not None else ''
    rec = ''
    if p.get('rec') is not None:
        rf = ', '.join(f'{k} x{v}' for k, v in p['rec_families'].most_common(3))
        rec = (f"\n      RECURRENT {'>=' if p['rec_lb'] else ''}{p['rec']} ({p['rec'] / p['runs']:.1%})"
               + (f'   {rf}' if rf else ''))
    return (f"N={p['N']:,}  α={p['alpha']:.2f}  β={p['beta']:.2f}{m0}  runs={p['runs']}   "
            f"BIPARTITE {p['bipartite']} ({p['bipartite'] / p['runs']:.2%})   {fam}{bal}{t}{note}{rec}\n"
            f"      raw splits {p['flagged']}, neutral {p['neutral']}, unclass. {p['other']}  |  outcomes: {top}")


def summary_lines(points):
    out = []
    tot_r = sum(p['runs'] for p in points)
    tot_b = sum(p['bipartite'] for p in points)
    fam = sum((p['families'] for p in points), Counter())
    out.append(f"total: {tot_b} bipartite in {tot_r:,} runs over {len(points)} points"
               + (f"; families: {', '.join(f'{k} x{v}' for k, v in fam.most_common())}" if fam else ''))
    # one α × β grid per (N, m0, c): mixing signal costs or mutation rates in one grid would overwrite cells
    groups = sorted({(p['N'], p.get('m0', 0.01), p.get('c', 0.01), p.get('e', 0.5), p.get('nonull', False)) for p in points})
    for N, m0, c, e, nn in groups:
        P = {(p['alpha'], p['beta']): p for p in points
             if (p['N'], p.get('m0', 0.01), p.get('c', 0.01), p.get('e', 0.5), p.get('nonull', False)) == (N, m0, c, e, nn)}
        alphas = sorted({a for a, _ in P})
        betas = sorted({b for _, b in P})
        tag = (f'N={N:,}' + (f', m0={m0:g}' if m0 != 0.01 else '') + (f', c={c:g}' if len(groups) > 1 or c != 0.01 else '')
               + (f', e={e:g}' if e != 0.5 else '') + (', NO-NULL model' if nn else ''))
        if len(alphas) > 1 and len(betas) > 1:
            out.append(f'bipartite runs per point at {tag} (rows α, columns β; "." = not finished yet)')
            out.append('    α \\ β ' + ''.join(f'{b:>7.2f}' for b in betas))
            for a in alphas:
                out.append(f'    {a:5.2f}   ' + ''.join(f"{P[(a, b)]['bipartite']:>7d}" if (a, b) in P else f"{'.':>7}"
                                                     for b in betas))
            if any(p.get('persistent') is not None for p in P.values()):
                out.append(f'persistent bipartite runs per point at {tag} (split in >= 23 of the last 25 snapshots; "." = not finished)')
                out.append('    α \\ β ' + ''.join(f'{b:>7.2f}' for b in betas))
                for a in alphas:
                    out.append(f'    {a:5.2f}   ' + ''.join(
                        f"{P[(a, b)]['persistent']:>7d}" if (a, b) in P and P[(a, b)].get('persistent') is not None
                        else f"{'.':>7}" for b in betas))
            if any(p.get('rec') is not None for p in P.values()):
                out.append(f'recurrent-split runs per point at {tag} ("." = not finished / not computed)')
                out.append('    α \\ β ' + ''.join(f'{b:>7.2f}' for b in betas))
                for a in alphas:
                    out.append(f'    {a:5.2f}   ' + ''.join(
                        f"{P[(a, b)]['rec']:>7d}" if (a, b) in P and P[(a, b)].get('rec') is not None else f"{'.':>7}"
                        for b in betas))
    if (len({p['N'] for p in points}) > 1 or len({p.get('m0', 0.01) for p in points}) > 1
            or len({p.get('c', 0.01) for p in points}) > 1 or len({p.get('e', 0.5) for p in points}) > 1 or len(points) <= 6):
        m0col = len({q.get('m0', 0.01) for q in points}) > 1 or any(q.get('m0', 0.01) != 0.01 for q in points)
        ccol = any(q.get('c', 0.01) != 0.01 for q in points)
        ecol = any(q.get('e', 0.5) != 0.5 for q in points)
        out.append('stable bipartite rate per point (95% Wilson interval):')
        out.append('    ' + '  '.join(['N', 'α', 'β'] + (['m0'] if m0col else []) + (['c'] if ccol else []) + (['e'] if ecol else [])
                                      + ['bipartite', 'runs', 'rate', '95% CI']))
        for p in sorted(points, key=lambda p: (p['alpha'], p['beta'], p['N'], p.get('e', 0.5))):
            lo, hi = wilson(p['bipartite'], p['runs'])
            out.append('    ' + '  '.join([f"{p['N']:,}", f"{p['alpha']:.2f}", f"{p['beta']:.2f}"]
                                          + ([f"{p['m0']:g}"] if m0col else []) + ([f"{p.get('c', 0.01):g}"] if ccol else [])
                                          + ([f"{p.get('e', 0.5):g}"] if ecol else [])
                                          + [str(p['bipartite']), str(p['runs']), f"{p['bipartite'] / p['runs']:.2%}",
                                             f'{lo:.2%} – {hi:.2%}']))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Markdown rendering. Formatters and summarizers return plain-text lines (table columns separated by >= 2 spaces); the
# md_* helpers turn those into Markdown, so custom formatters keep working and old .txt reports convert to exactly the
# same layout (convert_txt_report).

def _cells(line):
    return [c.strip() for c in re.split(r'\s{2,}', line.strip())]


def _esc(s):
    return s.replace('|', r'\|')


def _md_table(rows):
    """Text rows -> Markdown table, if the first row is a header (first cell without digits, no '=' or '/' in any cell)
    and every row has the same number (>= 2) of columns; otherwise None."""
    cells = [_cells(r) for r in rows]
    n = len(cells[0])
    if (len(rows) < 2 or n < 2 or any(len(c) != n for c in cells) or re.search(r'\d', cells[0][0])
            or any(re.search(r'[=/]', c) for c in cells[0])):
        return None
    numeric = lambda s: bool(re.fullmatch(r'[\d.,%≥>=+\-·/]+', s))
    align = [':---' if j == 0 or not all(numeric(c[j]) for c in cells[1:]) else '---:' for j in range(n)]
    return (['| ' + ' | '.join(_esc(c) for c in cells[0]) + ' |', '| ' + ' | '.join(align) + ' |']
            + ['| ' + ' | '.join(_esc(c) for c in r) + ' |' for r in cells[1:]])


def md_block(lines):
    """Summary-style text lines -> Markdown. A line followed by more-indented lines is a bold caption plus a table (or a
    fixed-width code block when the rows are not a clean table); any other line is a paragraph."""
    ind = lambda s: len(s) - len(s.lstrip(' '))
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j].strip() and ind(lines[j]) > ind(line):
            j += 1
        body = lines[i + 1:j]
        if body:
            t = _md_table(body)
            if t is None:
                d = min(ind(b) for b in body)
                t = ['```text'] + [b[d:] for b in body] + ['```']
            out += [f"**{line.strip().rstrip(':')}**", ''] + t + ['']
        else:
            out += [line.strip(), '']
        i = j
    return out


_KEY = re.compile(r'((?:BIPARTITE|RECURRENT|RETAINED|HELD) (?:>=)?\d[\d,]*(?:/\d+)?(?: \([^)]*\))?)')


def _point_line(s):
    return _KEY.sub(r'**\1**', re.sub(r'\s{2}', ' ', re.sub(r'\s{3,}', ' · ', s.strip())))


def md_points(blocks):
    """Point blocks (first line + indented continuation lines) -> nested Markdown list, key counts in bold."""
    out = []
    for b in blocks:
        ls = [l for l in b.split('\n') if l.strip()]
        out.append('- ' + _point_line(ls[0]))
        out += ['  - ' + _point_line(l) for l in ls[1:]]
    return out


def md_legend(text):
    """'TITLE' + '  TERM   description' lines (deeper-indented lines continue the description) -> heading + list."""
    ls = text.split('\n')
    items = []
    for l in ls[1:]:
        if not l.strip():
            continue
        if len(l) - len(l.lstrip()) <= 2 or not items:
            m = re.match(r'\s*(\S(?:.*?\S)?)\s{2,}(\S.*)$', l)
            items.append([m[1], m[2]] if m else [l.strip(), ''])
        else:
            items[-1][1] += ' ' + l.strip()
    return [f'## {ls[0].strip().capitalize()}', ''] + [f'- **{t}**' + (f' — {d}' if d else '') for t, d in items] + ['']


def md_footer(lines):
    """['EXACT COMMAND (note)', '  python ...'] -> heading + bash code block."""
    if not lines:
        return []
    m = re.match(r'([A-Z][A-Z ]*?)\s*(\(.*\))?$', lines[0].strip())
    head = (m[1].capitalize() + (f' {m[2]}' if m[2] else '')) if m else lines[0].strip()
    return [f'## {head}', '', '```bash'] + [l.strip() for l in lines[1:] if l.strip()] + ['```', '']


def render_markdown(title, status, started, updated, progress, purpose, settings, summary_title, summary, points,
                    todo, legend, footer):
    L = [f'# Exploration: {title}', '',
         f'- **Status:** {status}', f'- **Started:** {started} · **last update:** {updated}',
         f'- **Progress:** {progress}', '', '## Why', '', purpose, '', '## Settings', '']
    L += [f'- {s}' for s in settings] + ['', f'## {summary_title}', '']
    L += md_block(summary) if summary else ['(no point finished yet)', '']
    L += ['## Points (in completion order)', '']
    L += md_points(points) if points else ['(none yet)']
    if todo:
        L += ['', f'*{todo}*']
    L += [''] + (md_legend(legend) if legend else []) + md_footer(footer)
    return '\n'.join(L).rstrip() + '\n'


class Exploration:
    def __init__(self, desc, title, purpose, settings, planned, root=ROOT, started=None, footer=(), formatter=None,
                 summarizer=None, legend=None):
        """desc: file-name part after 'exploration_'. planned: [(key, cost)] for every point (cost ~ N*runs, used for
        the ETA). settings / footer: lists of lines (footer goes at the very end, e.g. the exact command).
        formatter(report, minutes, note) -> str and summarizer(reports) -> [lines] override the screen_point defaults."""
        self.desc, self.title, self.purpose, self.settings, self.footer = desc, title, purpose, list(settings), list(footer)
        self.root = root
        self.formatter = formatter or format_point
        self.summarizer = summarizer or summary_lines
        self.legend = LEGEND if legend is None else legend
        self.path = os.path.join(root, f'CURRENT_exploration_{desc}.md')
        self.planned = dict(planned)
        self.done = {}  # key -> (report, minutes, note), in completion order
        self.started = started or now()
        self.status = 'RUNNING'
        self.write()

    def add(self, key, report, minutes=None, note=''):
        self.done[key] = (report, minutes, note)
        self.write()

    def eta(self):
        timed = [(self.planned[k], m) for k, (_, m, _) in self.done.items() if m is not None and k in self.planned]
        remaining = sum(c for k, c in self.planned.items() if k not in self.done)
        if not timed or remaining == 0:
            return None
        rate = sum(m for _, m in timed) / sum(c for c, _ in timed)
        return (dt.datetime.now() + dt.timedelta(minutes=rate * remaining)).strftime('%H:%M')

    def text(self):
        """The report as Markdown."""
        eta = self.eta()
        pts = [r for r, _, _ in self.done.values()]
        todo = [k for k in self.planned if k not in self.done]
        running = self.status == 'RUNNING'
        return render_markdown(
            self.title, self.status, self.started, now(),
            f'{len(self.done)} / {len(self.planned)} points' + (f' · ETA ~{eta}' if eta and running else ''),
            self.purpose, self.settings, 'Summary so far' if running else 'Summary',
            self.summarizer(pts) if pts else [], [self.formatter(r, m, note) for r, m, note in self.done.values()],
            f'... {len(todo)} point(s) still to run, next: {todo[0]}' if todo and running else '',
            self.legend, self.footer)

    def write(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            f.write(self.text())
        os.replace(tmp, self.path)

    def stop(self, reason):
        """Sweep ended early: leave the CURRENT file in place, saying why (re-running the command resumes)."""
        self.status = f'STOPPED {now()} ({reason}); re-run the same command to resume'
        self.write()

    def finish(self):
        """Mark finished and move to archives/PAST_NN_exploration_<desc>.md. Returns the archive path."""
        self.status = f'FINISHED {now()}'
        self.write()
        arch = os.path.join(self.root, 'archives')
        os.makedirs(arch, exist_ok=True)
        nums = [int(m.group(1)) for f in os.listdir(arch) if (m := re.match(r'PAST_(\d+)_exploration_', f))]
        dest = os.path.join(arch, f'PAST_{max(nums, default=0) + 1:02d}_exploration_{self.desc}.md')
        os.replace(self.path, dest)
        return dest


_SECTION = re.compile(r'^(WHY|SETTINGS|SUMMARY(?: SO FAR)?|POINTS \(in completion order\)|HOW TO READ.*|EXACT COMMAND.*)$')


def convert_txt_report(path):
    """Convert an old plain-text report (CURRENT_/PAST_..._exploration_*.txt, layout of this module before 2026-11-14)
    to the Markdown layout. Writes the same name with .md and returns that path; the .txt is left for the caller."""
    ls = open(path).read().split('\n')
    title = ls[1].split('EXPLORATION:', 1)[1].strip()
    status, started, updated = re.match(r'status: (.*?)\s{3,}started (.*?)\s{3,}last update (.*)$', ls[2]).groups()
    progress = re.sub(r'\s{3,}', ' · ', ls[3].split('progress:', 1)[1].strip())
    sec, cur = {}, None
    for l in ls[5:]:
        if _SECTION.match(l):
            cur = l
            sec[cur] = []
        elif cur is not None:
            sec[cur].append(l)
    get = lambda pre: next(((k, v) for k, v in sec.items() if k.startswith(pre)), (None, []))
    purpose = ' '.join(l.strip() for l in get('WHY')[1] if l.strip())
    settings = []
    for l in get('SETTINGS')[1]:
        if l.strip():
            if len(l) - len(l.lstrip()) > 2 and settings:
                settings[-1] += ' ' + l.strip()
            else:
                settings.append(l.strip())
    sk, sl = get('SUMMARY')
    summary = [l[2:] if l.startswith('  ') else l for l in sl]
    blocks, todo = [], ''
    for l in get('POINTS')[1]:
        if not l.strip():
            continue
        if l.startswith('...'):
            todo = l.strip()
        elif not l.startswith(' ') or not blocks:
            blocks.append(l)
        else:
            blocks[-1] += '\n' + l
    lk, ll = get('HOW TO READ')
    fk, fl = get('EXACT COMMAND')
    md = render_markdown(title, status, started, updated, progress, purpose, settings,
                         sk.capitalize() if sk else 'Summary', summary, blocks, todo,
                         '\n'.join([lk] + ll).rstrip() if lk else None, [fk] + [l for l in fl if l.strip()] if fk else [])
    out = re.sub(r'\.txt$', '.md', path)
    with open(out, 'w') as f:
        f.write(md)
    return out
