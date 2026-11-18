"""One condition for every bipartite split family (E4 in RESEARCH_FRAMING.md, 2026-11-15).

For a type T split between strategies 1 and 2 (x = share of T in strategy 2; everyone else held at the final state), the per-interaction
payoff difference is linear in x:
    D(x) = EU_1 - EU_2 = D_out + s * [(1 - x)(pi11 - pi21) + x (pi12 - pi22)]
with pi_ij = payoff to a T-agent playing i against a T-agent playing j, s = T's population share, and D_out = the difference earned against
all other types. An interior balance x* in (0, 1) that is *stable* requires
    (G1) mixing pays inside the type:   pi12 + pi21 > pi11 + pi22    (then D rises with x: negative frequency dependence), and
    (G2) the outside difference is intermediate:   D(0) < 0 < D(1),  i.e.  -s(pi12 - pi22) < D_out < -s(pi11 - pi21).
G1 is what signals make possible: the two halves send different signals, so they can greet each other differently than they greet their own
half. Usage: python split_conditions.py  -> prints G1, G2, predicted vs observed x* for sample runs of every family."""
import glob

import numpy as np

import exploration_log as X
import psa_bipartite_tools as T


def halves(counts, t, M):
    """The split type's two largest payoff-equivalence classes (detector): (mask1, mask2, representative profile of each).
    Profiles within a class earn type t the same against every substantial partner, so one representative stands for the class."""
    det = T.detect_bipartite(counts, M)
    (l1, m1, n1), (l2, m2, n2) = T.type_class_masks(counts, det, t, top=2)
    C = counts[t].astype(float)
    return m1, m2, int(np.argmax(np.where(m1, C, -1))), int(np.argmax(np.where(m2, C, -1)))


def conditions(counts, t, M, m1, m2, p1, p2):
    n = counts.sum(0).astype(float)
    N = n.sum()
    C = counts[t].astype(float)
    s = (C[m1].sum() + C[m2].sum()) / N                                  # share of the population in the two halves
    others = n - np.where(m1 | m2, C, 0)                                 # everyone except the two halves of type t
    D_out = (M[t, p1] - M[t, p2]) @ others / N
    pi = np.array([[M[t, p1, p1], M[t, p1, p2]], [M[t, p2, p1], M[t, p2, p2]]])
    D0 = D_out + s * (pi[0, 0] - pi[1, 0])
    D1 = D_out + s * (pi[0, 1] - pi[1, 1])
    G1 = pi[0, 1] + pi[1, 0] - pi[0, 0] - pi[1, 1]
    xstar = D0 / (D0 - D1) if D0 != D1 else np.nan
    return dict(s=s, pi=pi, D_out=D_out, D0=D0, D1=D1, G1=G1, G2=(D0 < 0 < D1), xstar=xstar, obs=C[m2].sum() / (C[m1].sum() + C[m2].sum()))


if __name__ == '__main__':
    FAMS = ['Umma [blind | Kish-copy]', 'Kish [B-in | B-in]', 'Umma [E-in | Kish-copy]', 'Kish [E-in | Umma-copy]', 'Umma [S-in | Kish-copy]']
    TYPE = {'Umma': 0, 'Kish': 1}
    got = {f: [] for f in FAMS}
    for p in sorted(glob.glob('results/*/N10000_*.npz')):
        d = np.load(p, allow_pickle=True)
        lab = X.split_labels(d)
        M = T.point_matrix(d)
        for r in np.flatnonzero(d['bipartite']):
            f = lab[r]
            if f in got and len(got[f]) < 6:
                C = d['final_counts'][r]
                t = TYPE[f.split()[0]]
                m1, m2, p1, p2 = halves(C, t, M)
                cnd = conditions(C, t, M, m1, m2, p1, p2)
                obs = cnd['obs']
                got[f].append((p.split('/')[-1], int(r), T.profile_str(p1), T.profile_str(p2), cnd, obs))
        if all(len(v) >= 6 for v in got.values()):
            break
    for f in FAMS:
        print(f'===== {f}: {len(got[f])} sample runs')
        for fn, r, s1, s2, c, obs in got[f]:
            print(f'  {fn} r{r}: halves {s1} | {s2}; pi = {np.round(c["pi"], 3).tolist()}; G1 = {c["G1"]:+.3f}; '
                  f'D(0) = {c["D0"]:+.4f}, D(1) = {c["D1"]:+.4f} -> G2 {c["G2"]}; x* = {c["xstar"]:.3f} vs observed {obs:.3f}')
