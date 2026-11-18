"""Bipartite outcomes of the final sweep, counted up to isomorphism (user request, 2026-11-18).

An outcome is described by who uses which signal and what each pair coordinates on -- the poster's diagrams:
* boxes: the split type's two halves (its two largest payoff classes, as found by the split detector), plus one box for
  each other type; each box is labelled (type, signal it broadcasts), the signal being the box's modal signal;
* edges: for every pair of boxes, and each box with itself, the action they coordinate on (B, S, E) or '-' for
  miscoordination; the agent-pair-weighted plurality over the profiles in the two boxes.
Two outcomes are isomorphic if their labelled graphs match after any of:
* swapping signals 1 and 2 (the null signal 0 stays fixed); with more signals (T.set_num_signals), any permutation of
  the non-null signals;
* swapping Bachites with Stravinskians AND Bach with Stravinsky (their payoffs are mirror images);
* both.
The canonical form is the smallest serialisation over these 4 symmetries and all orderings of the boxes.

Usage: python isomorphism_check.py [glob]   (default results/final_sweep/*.npz)
Writes isomorphism_classes.csv (one row per class) and isomorphism_runs.csv (class of every run) next to the input files.
The functions are also used by three_signal_sweep.ipynb (kinds with 3 non-null signals)."""
import os
import glob
import itertools
import sys
import numpy as np
import pandas as pd
import psa_bipartite_tools as T


TYPES = ('Bachites', 'Stravinskians', 'Millennials')
ACT = 'BSE'


def outcome(p, q):
    """B/S/E or '-' when an agent with profile p meets one with profile q (null-signal rule included)."""
    o = T.interaction_outcome(int(p), int(q))
    return '-' if o < 0 else ACT[o]


def pair_outcome(va, vb):
    """Plurality outcome over agent pairs between two boxes (count vectors over the strategy profiles), and its share."""
    acc = {}
    for p in np.flatnonzero(va):
        for q in np.flatnonzero(vb):
            o = outcome(p, q)
            acc[o] = acc.get(o, 0) + va[p] * vb[q]
    best = max(acc, key=acc.get)
    return best, acc[best] / sum(acc.values())


def boxes_of(counts, det):
    """[(type, count vector)] : the split type's two halves, then every other type whole."""
    out = []
    split_types = {s['type'] for s in det['splits']}
    for t in range(3):
        if t in split_types:
            s = next(s for s in det['splits'] if s['type'] == t)
            for key in ('profiles_dominant', 'profiles_minority'):
                v = np.zeros(T.NUMPROFILES)
                v[s[key]] = counts[t, s[key]]
                out.append((t, v))
        else:
            out.append((t, counts[t].astype(float)))
    return out


def graph_of(counts, det):
    boxes = boxes_of(counts, det)
    nodes = [(t, int(np.argmax([v[T.PROFILES[:, 0] == k].sum() for k in range(T.NUMSIGNALS)]))) for t, v in boxes]
    n = len(boxes)
    edges, weakest = {}, 1.0
    for i in range(n):
        for j in range(i, n):
            o, share = pair_outcome(boxes[i][1], boxes[j][1])
            edges[(i, j)] = o
            weakest = min(weakest, share)
    return nodes, edges, weakest


def transform(nodes, edges, sigperm, typeswap):
    """sigperm: tuple, new label of non-null signal s is sigperm[s - 1]; signal 0 stays 0."""
    sw = {0: 0, **{s + 1: v for s, v in enumerate(sigperm)}}
    tw = {0: 1, 1: 0, 2: 2}
    aw = {'B': 'S', 'S': 'B', 'E': 'E', '-': '-'}
    nodes = [((tw[t] if typeswap else t), sw[s]) for t, s in nodes]
    edges = {k: (aw[o] if typeswap else o) for k, o in edges.items()}
    return nodes, edges


def serialise(nodes, edges, order):
    n = len(nodes)
    pos = {old: new for new, old in enumerate(order)}
    lab = tuple(nodes[o] for o in order)
    e = {}
    for (i, j), o in edges.items():
        a, b = sorted((pos[i], pos[j]))
        e[(a, b)] = o
    return lab, tuple(e[(a, b)] for a in range(n) for b in range(a, n))


def canonical(nodes, edges):
    best = None
    perms = list(itertools.permutations(range(1, T.NUMSIGNALS)))
    for sigperm, typeswap in itertools.product(perms, (False, True)):
        nn, ee = transform(nodes, edges, sigperm, typeswap)
        for order in itertools.permutations(range(len(nn))):
            s = serialise(nn, ee, order)
            if best is None or s < best:
                best = s
    return best


def describe(canon):
    """Readable form: boxes 'Bachites/sig0', then 'i-j:X' edges (X = coordinated action, - = miscoordinate)."""
    lab, e = canon
    n = len(lab)
    boxes = [f'{TYPES[t]}/sig{s}' for t, s in lab]
    k, parts = 0, []
    for a in range(n):
        for b in range(a, n):
            parts.append(f'{a}{b}:{e[k]}')
            k += 1
    return ' + '.join(boxes), ' '.join(parts)


def classify_runs(files):
    """One row per strict bipartite run of the given point files: parameters, split type, label, canonical form."""
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        M = T.point_matrix(d)
        lab = d['split_label']
        for r in np.flatnonzero(d['bipartite']):
            counts = d['final_counts'][r]
            det = T.detect_bipartite(counts, M)
            nodes, edges, weakest = graph_of(counts, det)
            c = canonical(nodes, edges)
            rows.append(dict(file=f, alpha=round(float(d['alpha']), 2), beta=round(float(d['beta']), 2), run=int(r),
                             split_type=TYPES[det['splits'][0]['type']], n_split_types=len(det['splits']),
                             label=str(lab[r]), canon=repr(c), weakest_plurality=weakest))
    return pd.DataFrame(rows)


def class_table(df):
    """Kinds up to isomorphism, most common first."""
    g = df.groupby('canon')
    cls = pd.DataFrame(dict(runs=g.size(), share=g.size() / len(df),
                            split_types=g.split_type.agg(lambda s: dict(s.value_counts())),
                            labels=g.label.agg(lambda s: dict(s.value_counts())),
                            weakest_plurality=g.weakest_plurality.min())).sort_values('runs', ascending=False)
    cls['boxes'], cls['edges'] = zip(*[describe(eval(c)) for c in cls.index])
    cls = cls.reset_index()
    cls.insert(0, 'class', range(1, len(cls) + 1))
    return cls


if __name__ == '__main__':
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'results/final_sweep/*.npz'
    files = sorted(glob.glob(pattern))
    out = os.path.dirname(files[0])
    df = classify_runs(files)
    df.to_csv(f'{out}/isomorphism_runs.csv', index=False)
    cls = class_table(df)
    cls.to_csv(f'{out}/isomorphism_classes.csv', index=False)
    print(f'{len(df)} strict bipartite runs, {df.file.nunique()} files; {len(cls)} classes up to isomorphism; '
          f'runs with two split types: {(df.n_split_types > 1).sum()}')
    pd.set_option('display.width', 250, 'display.max_colwidth', 120)
    for _, c in cls.iterrows():
        print(f"\nclass {c['class']}: {c.runs} runs ({100 * c.share:.1f}%), weakest plurality {c.weakest_plurality:.2f}")
        print(f"  boxes: {c.boxes}\n  edges: {c.edges}\n  split types: {c.split_types}\n  labels: {c.labels}")
