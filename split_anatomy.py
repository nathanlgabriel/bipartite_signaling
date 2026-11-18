"""Anatomy of every stable split at c = .001 (user request 2026-11-17: what do the various stable splits have in common, and which
arrangement of payoffs / preferences, together with a majority that ignores signals, stabilises them?).

For each strict-stable split (field `bipartite`) in the c = .001, m0 = .01 files of the null model (3 signals), from the final state:
  - the two halves of the split type (largest payoff-equivalence classes; a silent half is listed first, a copy half second),
  - who is silent (share of each type on the null signal),
  - whose signal each half sends ('lender' = another type whose modal signal it is),
  - the greeting (coordinated action) of every kind of meeting: half-half, and each half against each other type,
  - the split type's payoff from each kind of meeting, and the decomposition of D(x) = EU_1 - EU_2 (split_conditions):
      D_out = sum over other types o of share_o * (payoff_1 vs o - payoff_2 vs o)   (+ the remainder of the split type),
      G1 = pi12 + pi21 - pi11 - pi22.
Output: results/split_anatomy.csv (one row per run) and results/split_anatomy.txt (per-family summary + common features).
Usage: python split_anatomy.py [dirs...]   (default: results/c001 results/c001map results/c001e results/c001e2)
"""
import collections
import glob
import io
import sys

import numpy as np
import pandas as pd

import exploration_log as X
import psa_bipartite_tools as T
import split_conditions as SC

FAV = {0: 'B', 1: 'S', 2: 'E'}
NAMES = ['Umma', 'Kish', 'Akkad']
ACT = 'BSE'


def outcome_mix(p, C_o):
    """Distribution of the coordinated action when profile p meets type o (counts C_o): shares of B, S, E, failure."""
    w = np.zeros(4)
    for q in np.flatnonzero(C_o):
        o = T.interaction_outcome(p, int(q))
        w[o if o >= 0 else 3] += C_o[q]
    return w / w.sum()


def mix_str(w, tol=.9):
    k = int(np.argmax(w))
    s = (ACT + '-')[k]
    return s if w[k] >= tol else f"{s}{w[k]:.0%}/" + (ACT + '-')[int(np.argsort(w)[-2])]


def anatomy(C, t, M, label):
    m1, m2, p1, p2 = SC.halves(C, t, M)
    sig = lambda p: int(T.PROFILES[p][0])
    # order: silent half first; otherwise the copy half second (split_label convention); otherwise the larger half first
    if sig(p2) == 0 and sig(p1) != 0:
        m1, m2, p1, p2 = m2, m1, p2, p1
    elif label.split('[')[1].split('|')[0].strip().endswith('-copy'):
        m1, m2, p1, p2 = m2, m1, p2, p1
    cnd = SC.conditions(C, t, M, m1, m2, p1, p2)
    n = C.sum(0).astype(float)
    N = n.sum()
    ps = T.present_signals_of(C)
    modal = [T.canonical_signal(int(np.argmax([C[u, T.PROFILES[:, 0] == q].sum() for q in range(T.NUMSIGNALS)])), ps) for u in range(3)]
    row = dict(type=NAMES[t], label=label, x_obs=cnd['obs'], x_pred=cnd['xstar'], G1=cnd['G1'], D_out=cnd['D_out'],
               split_share=cnd['s'])
    for u in range(3):
        row[f'silent_{NAMES[u]}'] = C[u, T.PROFILES[:, 0] == 0].sum() / C[u].sum()
    for i, p in enumerate((p1, p2), 1):
        s = T.canonical_signal(sig(p), ps)
        row[f'h{i}_profile'] = T.profile_str(p)
        row[f'h{i}_silent'] = s == 0
        row[f'h{i}_lender'] = '+'.join(NAMES[u] for u in range(3) if u != t and modal[u] == s) or ('silence' if s == 0 else 'own')
    row['in1'] = ACT[T.interaction_outcome(p1, p1)] if T.interaction_outcome(p1, p1) >= 0 else '-'
    row['in2'] = ACT[T.interaction_outcome(p2, p2)] if T.interaction_outcome(p2, p2) >= 0 else '-'
    o12 = T.interaction_outcome(p1, p2)
    row['across'] = ACT[o12] if o12 >= 0 else '-'
    contrib = {}
    for u in range(3):
        if u == t:
            continue
        for i, p in enumerate((p1, p2), 1):
            row[f'h{i}_vs_{NAMES[u]}'] = mix_str(outcome_mix(p, C[u]))
            row[f'pay{i}_vs_{NAMES[u]}'] = (M[t, p] @ C[u]) / C[u].sum()
        contrib[NAMES[u]] = (C[u].sum() / N) * (row[f'pay1_vs_{NAMES[u]}'] - row[f'pay2_vs_{NAMES[u]}'])
        row[f'D_{NAMES[u]}'] = contrib[NAMES[u]]
    rest = C[t].astype(float) * ~(m1 | m2)
    row['D_rest'] = (M[t, p1] - M[t, p2]) @ rest / N
    # within-type terms of D at the observed split (what the two halves earn from meeting the split type)
    x = cnd['obs']
    pi = cnd['pi']
    row['D_within'] = cnd['s'] * ((1 - x) * (pi[0, 0] - pi[1, 0]) + x * (pi[0, 1] - pi[1, 1]))
    row['pi11'], row['pi12'], row['pi21'], row['pi22'] = pi[0, 0], pi[0, 1], pi[1, 0], pi[1, 1]
    return row


def main(dirs):
    rows = []
    for dname in dirs:
        for f in sorted(glob.glob(f'{dname}/N10000_*.npz')):
            d = np.load(f, allow_pickle=True)
            if '_m' in f.split('/')[-1] or abs(float(d['signal_cost']) + 0.001) > 1e-9 or not bool(d.get('null_signal', True)):
                continue
            M = T.point_matrix(d)
            lab = X.split_labels(d)
            for r in np.flatnonzero(d['bipartite']):
                t = NAMES.index(lab[r].split()[0])
                row = anatomy(d['final_counts'][r], t, M, lab[r])
                row.update(file=f, run=int(r), alpha=float(d['alpha']), beta=float(d['beta']), e=float(d['edm_payoff']))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('results/split_anatomy.csv', index=False)
    return df


def summary(df, out):
    print(__doc__.split('Usage')[0].strip() + '\n', file=out)
    print(f'{len(df)} stable splits in {df.file.nunique()} files\n', file=out)
    keys = ['label', 'h1_lender', 'h2_lender', 'in1', 'in2', 'across', 'h1_vs_Kish', 'h2_vs_Kish', 'h1_vs_Umma', 'h2_vs_Umma',
            'h1_vs_Akkad', 'h2_vs_Akkad']
    df = df.copy()
    df[keys] = df[keys].fillna('').astype(str)
    df['h1_lender'] = df['h1_lender'].replace({'': 'silence'})
    for k in keys:
        if k.startswith('h') and '_vs_' in k:
            df[k] = df[k].str[0]   # modal greeting only (mixed cases keep their majority letter)
    g = df.groupby(keys, dropna=False)
    tab = g.agg(n=('run', 'size'), G1=('G1', 'mean'), sil_U=('silent_Umma', 'mean'), sil_K=('silent_Kish', 'mean'),
                sil_A=('silent_Akkad', 'mean'), D_U=('D_Umma', 'mean'), D_K=('D_Kish', 'mean'), D_A=('D_Akkad', 'mean'),
                D_in=('D_within', 'mean'), x=('x_obs', 'mean'), xerr=('x_pred', lambda s: 0)).reset_index()
    tab = tab.sort_values('n', ascending=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.max_colwidth', 30):
        print('Configurations (halves: h1 = silent / non-copy half, h2 = copy / other half; lender = whose signal the half sends;\n'
              'in1/in2 = greeting inside each half, across = between halves; hX_vs_T = greeting of half X with type T\n'
              '(a% = mixed); D_T = share_T * (payoff_1 - payoff_2 against T); D_in = within-type part of D at the observed x):\n', file=out)
        print(tab.drop(columns='xerr').round(3).to_string(index=False), file=out)


if __name__ == '__main__':
    dirs = [a for a in sys.argv[1:] if a != '--cached'] or ['results/c001', 'results/c001map', 'results/c001e', 'results/c001e2']
    df = pd.read_csv('results/split_anatomy.csv') if '--cached' in sys.argv else main([d for d in dirs if d != '--cached'])
    buf = io.StringIO()
    summary(df, buf)
    open('results/split_anatomy.txt', 'w').write(buf.getvalue())
    sys.stdout.write(buf.getvalue())
