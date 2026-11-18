"""Helpers for the PSA_poster bipartite-signaling-equilibria project.

Model: 1 signal dimension with 3 signals (0 = null), 3 actions (0=B/Umma, 1=S/Kish, 2=E/EDM/Akkadian),
3 preference types. A strategy profile index p encodes [own signal, action|sig0, action|sig1, action|sig2]
with multipliers [1, 3, 9, 27] (same as `repmultipliers` / `profiles_indexed` in the FNs module).

Contents
- standard parameters (`make_params`) and a parallel batch runner (`run_batch`) around
  FNs_genBS_PSA_bipartite.genBSreplicatorexec_full_play_PSA_typed
- profile decoding / effective behaviour, bipartite detector, outcome classifier incl. (xiv)-(xvii)
- deviation utilities + Nash check that mirror executilcompute's payoff logic
- NO-NULL-SIGNAL variant (2026-11-15): make_params(..., null_signal=False) runs
  FNs_genBS_PSA_bipartite_noNULL (= FNs_..._noNULL_claude.py + recording), where signal 0 is an ordinary signal.
  Those modules are imported lazily, only for such runs (not needed by final_sweep / three_signal_sweep).
  The analysis functions follow the module flag NULL_SIGNAL (set_null_signal); screen_point and point_matrix set it
  from the parameters / the file, so files of either model are analysed with the right reading rule.
"""
import concurrent.futures
from sys import float_info

import numba
import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence

from FNs_genBS_PSA_bipartite import (genBSreplicatorexec_full_play_PSA_typed, genBSreplicatorexec_full_play_PSA_init_typed,
                                     population_array)
# The no-null-signal modules (FNs_genBS_PSA_bipartite_noNULL.py and the _noNULL_claude.py file it imports) are imported
# only when a no-null run is asked for (run_batch with null_signal=False), so the notebooks run without those files.

NUMTYPES, NUMSIGNALS, NUMACTIONS, NUMPROFILES = 3, 3, 3, 81
REPMULT = np.array([1, 3, 9, 27])
ACTION_LETTERS = np.array(['B', 'S', 'E'])
TYPE_NAMES = ['Umma(0)', 'Kish(1)', 'Akkad(2)']
NULL_SIGNAL = True  # which model the analysis functions assume: True = original (signal 0 is null), False = no-null variant


def set_null_signal(flag):
    """Switch the analysis functions (reading rule, signal merging, split labels, payoff_matrix) between the original model
    (True: an agent sending signal 0 reads everyone as 0 and is read as 0) and the no-null variant (False: signal 0 is an
    ordinary signal; the signal cost is still charged only for signals 1 and 2, as in FNs_..._noNULL_claude.py)."""
    global NULL_SIGNAL
    NULL_SIGNAL = bool(flag)


def set_num_signals(n):
    """Switch every analysis function to a model with `n` signals (2026-11-17, E2 '4th signal' test). The original module
    is generic in the number of signals (numsignals_perdim=[n], numsignals=n, numprofiles=n*3**n); signal 0 stays the
    null signal (or an ordinary one with set_null_signal(False)). n = 3 is the paper's model and leaves everything as
    before. Rebinds NUMSIGNALS, NUMPROFILES, REPMULT, PROFILES (functions read them at call time)."""
    global NUMSIGNALS, NUMPROFILES, REPMULT, PROFILES
    n = int(n)
    if n == NUMSIGNALS:
        return
    NUMSIGNALS, NUMPROFILES = n, n * NUMACTIONS ** n
    REPMULT = np.array([1] + [n * NUMACTIONS ** i for i in range(n)])
    PROFILES = np.array([decode_profile(i) for i in range(NUMPROFILES)])


# ------------------------------------------------------------------------------------------------
# parameters + batch runner
# ------------------------------------------------------------------------------------------------
def make_params(alpha, beta, population_size=1000, runlength=8 * 10**4, mutation_rate=(0.01, 0.1),
                signal_cost=-0.01, edm_payoff=0.5, null_signal=True, num_signals=3, coord=None):
    """Settings of genBS_v0055k_..._top213signals.ipynb for one (alpha, beta) grid point (null_signal=False: the
    no-null-signal variant, genBS_v0055k_..._top213signals_noNULL_claude.ipynb; num_signals=4: one more ordinary signal,
    same module; coord: a full 3x3 payoff matrix [row type][action] replacing the standard one, for payoff-arrangement tests)."""
    numsignals_perdim = np.array([num_signals])
    sigmultipliers = np.roll(np.cumprod(numsignals_perdim), 1)
    sigmultipliers[0] = 1
    dimactions = np.array(list(numsignals_perdim) + [NUMACTIONS] * num_signals)
    repmultipliers = np.roll(np.cumprod(dimactions), 1)
    repmultipliers[0] = 1
    percent = [.33 + beta, .33 - beta, .34]
    return dict(
        alpha=alpha, beta=beta,
        mutation_rate=np.array(mutation_rate, dtype=np.float64),
        signal_cost=signal_cost,
        repmultipliers=repmultipliers,
        numprofiles=int(np.prod(dimactions)),
        signal_snapshots=1,
        record_interval=runlength,
        genBSpunish=np.zeros(2),
        numtypes=NUMTYPES, numsignals=int(num_signals), runlength=runlength, numactions=NUMACTIONS,
        coordination_preferences=(np.array([[1 + alpha, 1, edm_payoff], [1, 1 + alpha, edm_payoff], [0, 0, 1]]) if coord is None
                                  else np.array(coord, dtype=np.float64).reshape(3, 3)),
        popt=population_array(population_size, percent),
        population_size=population_size, sigdimensions=1,
        base_connection_weights=np.array([1, 2, 4, 8, 16, 32]),
        sigmultipliers=sigmultipliers, homophily_factor=0, epsilon=float_info.epsilon,
        numsignals_perdim=numsignals_perdim, null_signal=bool(null_signal),
    )


def _sim_args(P, rng, runid, profile_record_interval):
    sigurns = [np.ones([P['population_size'], P['numsignals_perdim'][0]])]
    recurns = np.ones([P['population_size'], P['numsignals'], P['numactions']])
    return (profile_record_interval, P['mutation_rate'], P['signal_cost'], P['repmultipliers'], P['numprofiles'],
            P['signal_snapshots'], P['record_interval'], P['genBSpunish'], P['numtypes'], P['numsignals'],
            P['runlength'], P['numactions'], P['coordination_preferences'], P['popt'], sigurns, recurns,
            P['population_size'], P['sigdimensions'], P['base_connection_weights'], P['sigmultipliers'],
            P['homophily_factor'], rng, P['epsilon'], runid)


def make_rngs(entropy, runs):
    return [Generator(PCG64DXSM(s)) for s in SeedSequence(entropy).spawn(runs)]


def run_batch(P, runs, entropy=None, thrds=20, profile_record_interval=400, keep_profile_time=True,
              run_ids=None, init_counts=None):
    """Run `runs` simulations (seeded exactly like the original notebook: SeedSequence(entropy).spawn(runs),
    run r uses child r). `run_ids` optionally restricts to a subset of those children (for re-running
    specific runs). `init_counts` ((3, 81) for every run, or one (3, 81) per run in run order) starts the runs from
    that state instead of a random one (genBSreplicatorexec_full_play_PSA_init_typed). Returns a dict of stacked
    per-run arrays plus 'entropy'."""
    if entropy is None:
        entropy = SeedSequence().entropy
    rngs = make_rngs(entropy, runs)
    ids = list(range(runs)) if run_ids is None else list(run_ids)
    n = len(ids)
    K = P['runlength'] // profile_record_interval
    ns, npf = P['numsignals'], P['numprofiles']
    out = dict(
        entropy=np.array(entropy, dtype=object), run_ids=np.array(ids),
        ss001=np.zeros((n, 3, 3, 3), np.int8), ss002=np.zeros((n, 3, 3, 3), np.int8),
        socialsig=np.zeros((n, 3, ns), np.int32), action_agg=np.zeros((n, 3, ns, 3), np.float32),
        freeze_step=np.zeros(n, np.int64),
        final_counts=np.zeros((n, 3, npf), np.int32), final_counts_float=np.zeros((n, 3, npf)),
        final_utility=np.zeros((n, 3, npf)), profile_init=np.zeros((n, 3, npf), np.int32),
    )
    if keep_profile_time:
        out['profile_time'] = np.zeros((n, 3, npf, K), np.int32)
    pos = {rid: i for i, rid in enumerate(ids)}
    null = P.get('null_signal', True)
    if null:
        run_fn, init_fn = genBSreplicatorexec_full_play_PSA_typed, genBSreplicatorexec_full_play_PSA_init_typed
    else:   # lazy: the no-null modules are needed only here
        from FNs_genBS_PSA_bipartite_noNULL import (genBSreplicatorexec_full_play_PSA_typed as run_fn,
                                                    genBSreplicatorexec_full_play_PSA_init_typed as init_fn)
    with concurrent.futures.ProcessPoolExecutor(max_workers=thrds) as ex:
        if init_counts is None:
            futs = [ex.submit(run_fn, *_sim_args(P, rngs[r], r, profile_record_interval))
                    for r in ids]
        else:
            ic = np.asarray(init_counts, dtype=np.int64)
            ic = np.broadcast_to(ic, (n,) + ic.shape[-2:]) if ic.ndim == 2 else ic
            futs = [ex.submit(init_fn, ic[i],
                              *_sim_args(P, rngs[r], r, profile_record_interval)) for i, r in enumerate(ids)]
        for f in concurrent.futures.as_completed(futs):
            (ss002, ss001, socialsig, action_agg, _s, _r, _e, _tt, _st, _ttn, _stn, runid,
             profile_time, profile_init, freeze_step, fci, fcf, futil) = f.result()
            i = pos[runid]
            out['ss001'][i], out['ss002'][i] = ss001, ss002
            out['socialsig'][i], out['action_agg'][i] = socialsig[-1], action_agg
            out['freeze_step'][i] = freeze_step
            out['final_counts'][i], out['final_counts_float'][i], out['final_utility'][i] = fci, fcf, futil
            out['profile_init'][i] = profile_init
            if keep_profile_time:
                out['profile_time'][i] = profile_time
    return out


# ------------------------------------------------------------------------------------------------
# profiles
# ------------------------------------------------------------------------------------------------
def decode_profile(idx):
    """profile index -> (own signal, action|sig0, action|sig1, action|sig2[, action|sig3])"""
    return tuple(int((idx // m) % (NUMSIGNALS if i == 0 else NUMACTIONS)) for i, m in enumerate(REPMULT))


PROFILES = np.array([decode_profile(i) for i in range(NUMPROFILES)])  # (81, 4) == profiles_indexed


def profile_str(idx):
    s = PROFILES[idx][0]
    return f"sig{s} [{''.join(ACTION_LETTERS[a] for a in PROFILES[idx][1:])}]"


def perceived_action(p, partner_signal):
    """Action profile p takes against a partner broadcasting `partner_signal` (null-signal rule:
    if either side broadcasts 0, both read 0; no-null variant: always the partner's actual signal)."""
    if not NULL_SIGNAL:
        return int(PROFILES[p][1 + partner_signal])
    s = PROFILES[p][0]
    seen = partner_signal if (s != 0 and partner_signal != 0) else 0
    return int(PROFILES[p][1 + seen])


def effective_behavior(p, present_signals):
    """(own signal, actions against each present partner signal) -- profiles that differ only on
    signals nobody sends are behaviourally identical."""
    return (canonical_signal(PROFILES[p][0], present_signals),) + tuple(perceived_action(p, s) for s in present_signals)


def behavior_str(key, present_signals):
    acts = ' '.join(f"{s}->{ACTION_LETTERS[a]}" for s, a in zip(present_signals, key[1:]))
    return f"sig{key[0]} ({acts})"


def present_signals_of(counts, min_frac=0.01):
    """Signals broadcast by at least min_frac of the population (counts: (3, 81)). Signals 1 and 2 are merged
    (only 1 is kept) when no substantial signalling profile (>= min_frac of its type) acts differently on them --
    then sending 1 vs 2 is a pure relabelling with identical payoffs (neutral drift, not a bipartite split)."""
    tot = counts.sum(0)
    sig_counts = np.array([tot[PROFILES[:, 0] == s].sum() for s in range(NUMSIGNALS)])
    present = [int(s) for s in np.flatnonzero(sig_counts >= min_frac * tot.sum())]
    if not NULL_SIGNAL:  # no-null variant: any two present signals that every substantial profile treats alike are merged
        canon = {s: s for s in range(NUMSIGNALS)}
        for a, b in ((0, 1), (0, 2), (1, 2)):
            if a in present and b in present and canon[b] == b and _pair_equivalent(counts, a, b, min_frac):
                canon[b] = canon[a]
        out = _Present(s for s in present if canon[s] == s)
        out.canon = canon
        return out
    if NUMSIGNALS > 3:  # 4-signal model: any two present non-null signals that every substantial reader treats alike merge
        canon = {s: s for s in range(NUMSIGNALS)}
        for a in range(1, NUMSIGNALS):
            for b in range(a + 1, NUMSIGNALS):
                if a in present and b in present and canon[a] == a and canon[b] == b and _pair_equivalent_null(counts, a, b, min_frac):
                    canon[b] = a
        out = _Present(s for s in present if canon[s] == s)
        out.canon = canon
        return out
    if 1 in present and 2 in present and signals_equivalent(counts, min_frac):
        present.remove(2)
    return tuple(present)


def _pair_equivalent_null(counts, a, b, min_frac=0.01):
    """Null model with > 2 non-null signals: True if every substantial profile that reads signals (own signal != 0)
    plays the same action on a and b."""
    for t in range(NUMTYPES):
        tot_t = counts[t].sum()
        for p in np.flatnonzero(counts[t] >= max(1, min_frac * tot_t)):
            if PROFILES[p][0] != 0 and PROFILES[p][1 + a] != PROFILES[p][1 + b]:
                return False
    return True


def signals_equivalent(counts, min_frac=0.01):
    """True if every substantial profile that reads signals (own signal != 0) plays the same action on 1 and 2."""
    for t in range(NUMTYPES):
        tot_t = counts[t].sum()
        for p in np.flatnonzero(counts[t] >= max(1, min_frac * tot_t)):
            if PROFILES[p][0] != 0 and PROFILES[p][2] != PROFILES[p][3]:
                return False
    return True


class _Present(tuple):
    """Tuple of present signals that also carries the no-null merge map (attribute canon: signal -> kept signal)."""
    canon = None


def _pair_equivalent(counts, a, b, min_frac=0.01):
    """No-null variant: True if every substantial profile (all of them read signals) plays the same action on a and b."""
    for t in range(NUMTYPES):
        tot_t = counts[t].sum()
        for p in np.flatnonzero(counts[t] >= max(1, min_frac * tot_t)):
            if PROFILES[p][1 + a] != PROFILES[p][1 + b]:
                return False
    return True


def canonical_signal(s, present_signals):
    """Map signal 2 -> 1 when 2 has been merged into 1 (see present_signals_of); no-null variant: the merge map."""
    canon = getattr(present_signals, 'canon', None)
    if canon is not None:
        return int(canon[int(s)])
    return 1 if (s == 2 and 2 not in present_signals and 1 in present_signals) else int(s)


# ------------------------------------------------------------------------------------------------
# bipartite detection
# ------------------------------------------------------------------------------------------------
def substantial_profiles(counts, min_frac=0.01):
    """Profiles held by >= min_frac of at least one type (filters out the ~1% mutation noise)."""
    tot = counts.sum(1, keepdims=True)
    return np.flatnonzero((counts >= np.maximum(1, min_frac * tot)).any(0))


def interaction_outcome(A, B):
    """Action both coordinate on when A meets B, or -1 for a coordination failure."""
    aA = perceived_action(A, PROFILES[B][0])
    aB = perceived_action(B, PROFILES[A][0])
    return aA if aA == aB else -1


def outcome_signature(A, partners):
    """Payoff-relevant behaviour of profile A in a population whose substantial profiles are `partners`:
    (pays signal cost?, outcome against each partner). Two profiles with the same signature earn identical payoffs
    for every type and give every partner identical payoffs -- they differ only by neutral drift."""
    return (int(PROFILES[A][0] != 0),) + tuple(interaction_outcome(A, B) for B in partners)


def payoff_signature(A, t, partners, M):
    """What profile A earns, as a type-t agent, against each substantial partner (signal cost included). Profiles with
    equal payoff signatures are selection-neutral for type t: the replicator cannot tell them apart."""
    return tuple(np.round(M[t, A, partners], 10))


def _classes(counts_t, keyfun):
    classes = {}
    for p in np.flatnonzero(counts_t):
        k = keyfun(p)
        c = classes.setdefault(k, [0, []])
        c[0] += int(counts_t[p])
        c[1].append(int(p))
    return sorted(((k, v[0], v[1]) for k, v in classes.items()), key=lambda x: -x[1])


def _split_from_classes(cl, counts_t, t, threshold):
    tot = counts_t.sum()
    if len(cl) < 2 or cl[1][1] < threshold * tot:
        return None
    (k1, n1, p1), (k2, n2, p2) = cl[0], cl[1]
    rep1 = p1[int(np.argmax(counts_t[p1]))]
    rep2 = p2[int(np.argmax(counts_t[p2]))]
    return dict(type=t, dominant=k1, n_dominant=n1, profiles_dominant=p1, rep_dominant=rep1,
                minority=k2, n_minority=n2, profiles_minority=p2, rep_minority=rep2, minority_share=n2 / tot,
                signals_differ=bool(PROFILES[rep1][0] != PROFILES[rep2][0]))


def detect_bipartite(counts, M, threshold=0.10, noise_frac=0.01):
    """counts: (3, 81) integer typed counts; M: payoff_matrix(...) for the run's parameters.

    A type is *split* when its 2nd-largest payoff-equivalence class (profiles earning the type identical payoffs
    against every substantial partner) holds >= threshold of the type -- i.e. the two sub-populations are genuinely
    under different selection pressures and coexist only if their total utilities balance (the Appendix G case).
    Separately, `neutral_splits` lists types split into >= threshold classes that differ in interaction outcomes
    (they matter to partners) but are payoff-identical for the splitting type itself (selection-neutral drift).
    Returns dict(flag, splits, neutral_splits, partners, present_signals)."""
    partners = substantial_profiles(counts, noise_frac)
    present = present_signals_of(counts, noise_frac)
    splits, neutral = [], []
    for t in range(NUMTYPES):
        sp = _split_from_classes(_classes(counts[t], lambda p: payoff_signature(p, t, partners, M)), counts[t], t, threshold)
        if sp is not None:
            splits.append(sp)
            continue
        sp = _split_from_classes(_classes(counts[t], lambda p: outcome_signature(p, partners)), counts[t], t, threshold)
        if sp is not None:
            neutral.append(sp)
    return dict(flag=len(splits) > 0, splits=splits, neutral_splits=neutral, partners=partners,
                present_signals=present, M=M)


def split_ok_series(profile_time, det, threshold=0.10):
    """Boolean (K,) series: in which snapshots of profile_time (3, 81, K) det's split -- same type, same dominant and
    minority payoff classes, computed against det's substantial profiles -- held with both classes >= threshold."""
    s = det['splits'][0]
    t = s['type']
    keys = [payoff_signature(p, t, det['partners'], det['M']) for p in range(NUMPROFILES)]
    in_dom = np.array([k == s['dominant'] for k in keys])
    in_min = np.array([k == s['minority'] for k in keys])
    ts = profile_time[t]  # (81, K)
    tot = ts.sum(0)
    return (ts[in_min].sum(0) >= threshold * tot) & (ts[in_dom].sum(0) >= threshold * tot)


def persistent_runs(d, maxmiss=2, window=25):
    """Boolean (runs,) mask of *persistent* bipartite runs in a loaded screen_point file: the stable ones (`bipartite`)
    plus runs whose final state is a flagged, eps-Nash split that held in all but at most `maxmiss` of the last `window`
    snapshots, with no two misses in a row (with 400-step snapshots: split >= ~92% of the last 10k steps, no gap > 800 steps).
    Why (2026-11-14): one-step replicator overshoots knock one half below the threshold for ~15-150 steps a few times
    per 10k steps even in clean splits (10-step re-runs). Whether a snapshot lands in such a dip is luck, so the strict
    criterion (`window` consecutive snapshots) depends on the snapshot interval. Needs every run's series."""
    if 'recurrent_lower_bound' not in d or bool(d['recurrent_lower_bound']):
        raise ValueError('file does not keep every run\'s profile series')
    M = point_matrix(d)
    thr = float(d['split_threshold']) if 'split_threshold' in d else 0.10
    out = np.asarray(d['bipartite']).copy()
    row = {int(r): i for i, r in enumerate(d['kept_runs'])}
    for r in np.flatnonzero(d['flagged'] & d['nash_ok'] & ~d['bipartite']):
        det = detect_bipartite(d['final_counts'][r], M, threshold=thr)
        if not det['flag']:
            continue
        miss = ~split_ok_series(d['profile_time_kept'][row[int(r)]], det, thr)[-window:]
        out[r] = (not miss[-1]) and miss.sum() <= maxmiss and not (miss[1:] & miss[:-1]).any()
    return out


def split_age(profile_time, det, threshold=0.10):
    """Number of trailing profile snapshots (profile_time: (3, 81, K)) in which the final split -- same type, same
    dominant and minority payoff classes, computed against the final state's substantial profiles -- already held
    with both classes >= threshold of the type. Cycling / transient runs give ~0; a settled split a large value."""
    if not det['flag']:
        return 0
    ok = split_ok_series(profile_time, det, threshold)
    age = 0
    for v in ok[::-1]:
        if not v:
            break
        age += 1
    return age


def split_time_fraction(profile_time, det, threshold=0.10, window=0.5):
    """Fraction of the last `window` of the profile snapshots (profile_time: (3, 81, K)) in which det's split -- same
    type, same two payoff classes -- held with both classes >= threshold of the type. A stable split gives 1. A
    *recurrent* split, which keeps collapsing for a moment (one half takes over) and re-forming (the missing half
    re-invades through mutation because it does better when rare), gives a high value but a short split_age."""
    if not det['flag']:
        return 0.0
    s = det['splits'][0]
    t = s['type']
    keys = [payoff_signature(p, t, det['partners'], det['M']) for p in range(NUMPROFILES)]
    in_dom = np.array([k == s['dominant'] for k in keys])
    in_min = np.array([k == s['minority'] for k in keys])
    ts = profile_time[t][:, int(profile_time.shape[2] * (1 - window)):]
    tot = ts.sum(0)
    return float(((ts[in_min].sum(0) >= threshold * tot) & (ts[in_dom].sum(0) >= threshold * tot)).mean())


def latest_split(profile_time, M, threshold=0.10, lookback=10):
    """(det, counts) for the most recent of the last `lookback` snapshots that shows a split ((None, None) if none
    does). Used for runs whose final state is momentarily unsplit during a recurrent split."""
    K = profile_time.shape[2]
    for k in range(K - 1, max(-1, K - 1 - lookback), -1):
        det = detect_bipartite(profile_time[:, :, k], M, threshold=threshold)
        if det['flag']:
            return det, profile_time[:, :, k]
    return None, None


def recurrent_split(profile_time, final_counts, final_det, M, threshold=0.10, window=0.5, lookback=10):
    """(split_frac, label) for a run: split_time_fraction and split_label of the final split or -- when the final state
    is unsplit -- of the latest split in the last `lookback` snapshots ((0.0, '') when there is no split to follow)."""
    det, c = (final_det, final_counts) if final_det['flag'] else latest_split(profile_time, M, threshold, lookback)
    if det is None:
        return 0.0, ''
    return split_time_fraction(profile_time, det, threshold, window), split_label(c, det)


def describe_split(det, key='splits'):
    lines = []
    for s in det[key]:
        ps = det['present_signals']
        b1 = behavior_str(effective_behavior(s['rep_dominant'], ps), ps)
        b2 = behavior_str(effective_behavior(s['rep_minority'], ps), ps)
        lines.append(f"{TYPE_NAMES[s['type']]}: dominant {b1} x{s['n_dominant']}  |  "
                     f"minority {b2} x{s['n_minority']} ({s['minority_share']:.1%})")
    return '\n'.join(lines) if lines else 'no split'


SPLIT_LABEL_LEGEND = (
    "split label  <type> [<half> | <half>]: how the two halves of the split type behave. "
    "'<X>-in' = the half coordinates on action X (B/S/E) within itself ('--in' = fails to coordinate); "
    "'blind' = the half sends the null signal, so it treats everyone alike; "
    "(no-null variant: no 'blind'; '<X>-in@<type>' = the half sends that type's modal signal, '@own' = a signal no other "
    "type mainly uses); "
    "'Kish-copy' / 'Umma-copy' = the half has exactly the other Sumerian type's behaviour (signal and actions). "
    "A copy half is listed second, otherwise the larger half first. Unless marked 'across X', the two halves "
    "coordinate on the split type's own favourite (Umma B, Kish S) when they meet. "
    "Families: [B-in | B-in] Kish / [S-in | S-in] Umma = cross-matching; [blind | copy]; [E-in | copy] = Appendix G "
    "(xv)/(xvi); Umma [S-in | Kish-copy] = cross-matching with Kish joining one half.")

FAVOURITE = {0: 0, 1: 1, 2: 2}  # each type's own favourite action (Umma B, Kish S, Akkad E)


def split_label(counts, det):
    """Concise structural label of a run's split (see SPLIT_LABEL_LEGEND), independent of which half happens to be the
    larger one and of signal relabelling (1 <-> 2), e.g. 'Kish [B-in | B-in]' or 'Umma [blind | Kish-copy]'.
    Returns '' when det has no split."""
    if not det['flag']:
        return ''
    s = det['splits'][0]
    t = s['type']
    ps = det['present_signals']
    halves = [s['rep_dominant'], s['rep_minority']]

    def is_copy(p):
        if t == 2:
            return False
        o = 1 - t
        if counts[o].sum() == 0:
            return False
        q = int(np.argmax(counts[o]))  # the other Sumerian type's most common profile
        return effective_behavior(p, ps) == effective_behavior(q, ps)

    def modal_signal(u):
        return canonical_signal(int(np.argmax([counts[u, PROFILES[:, 0] == q].sum() for q in range(NUMSIGNALS)])), ps)

    def desc(p):
        if is_copy(p):
            return f"{TYPE_NAMES[1 - t].split('(')[0]}-copy"
        if NULL_SIGNAL and PROFILES[p][0] == 0:
            return 'blind'
        o = interaction_outcome(p, p)
        if NULL_SIGNAL:
            return f"{'BSE'[o] if o >= 0 else '-'}-in"
        # no-null variant: also say whose signal the half sends ('@own' = no other type's modal signal)
        sp = canonical_signal(PROFILES[p][0], ps)
        own = [TYPE_NAMES[u].split('(')[0] for u in range(NUMTYPES) if u != t and counts[u].sum() and modal_signal(u) == sp]
        return f"{'BSE'[o] if o >= 0 else '-'}-in@{'+'.join(own) if own else 'own'}"

    d = [desc(p) for p in halves]
    if d[0].endswith('-copy') and not d[1].endswith('-copy'):
        d = d[::-1]
    across = interaction_outcome(halves[0], halves[1])
    tail = '' if across == FAVOURITE[t] else f" across {'BSE'[across] if across >= 0 else '-'}"
    return f"{TYPE_NAMES[t].split('(')[0]} [{d[0]} | {d[1]}]{tail}"


# ------------------------------------------------------------------------------------------------
# outcome classification of the modal table simple_stat_001 (rows = actor type, cols = partner type)
# ------------------------------------------------------------------------------------------------
def _table(rows):
    a = {'B': 0, 'S': 1, 'E': 2}
    t = np.zeros((3, 3, 3), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            t[i, j, a[ch]] = 1
    return t


# notebook labels (with appendix G numerals) then the bipartite tables (xiv)-(xvii) from SupplementA
OUTCOMES = {
    'A (i)':       _table(['BBB', 'SSS', 'EEE']),
    'B0 (ii)':     _table(['BBB', 'BBB', 'EEE']),
    'B1 (iii)':    _table(['SSS', 'SSS', 'EEE']),
    'B2 (iv)':     _table(['BBE', 'BBE', 'EEE']),
    'B3 (v)':      _table(['SSE', 'SSE', 'EEE']),
    'B4 (vi)':     _table(['BBB', 'BSB', 'EEE']),
    'B5 (vii)':    _table(['BSS', 'SSS', 'EEE']),
    'D0 (viii)':   _table(['BBB', 'BSE', 'EEE']),
    'D1 (ix)':     _table(['BSE', 'SSS', 'EEE']),
    'WEIRD0 (x)':  _table(['SBE', 'BSE', 'EEE']),
    'WEIRD1 (xi)': _table(['BSE', 'SBE', 'EEE']),
    'C0 (xii)':    _table(['BBE', 'BSE', 'EEE']),
    'C1 (xiii)':   _table(['BSE', 'SSE', 'EEE']),
    'xiv':         _table(['BBB', 'BEE', 'EEE']),
    'xv':          _table(['EBE', 'BSE', 'EEE']),
    'xvi':         _table(['BSE', 'SEE', 'EEE']),
    'xvii':        _table(['BEE', 'ESE', 'EEE']),
}
OUTCOME_LABELS = list(OUTCOMES) + ['other']


def classify(ss001):
    for name, tab in OUTCOMES.items():
        if np.array_equal(ss001, tab):
            return name
    return 'other'


def table_letters(ss001):
    return [''.join(ACTION_LETTERS[np.argmax(ss001[i, j])] for j in range(3)) for i in range(3)]


# ------------------------------------------------------------------------------------------------
# utilities + Nash check (mirrors executilcompute with homophily 0, 1 signal dimension)
# ------------------------------------------------------------------------------------------------
def payoff_matrix(coordination_preferences, signal_cost=-0.01, punish=0.0):
    """M[t, A, B] = payoff to a type-t agent playing profile A against one partner playing profile B."""
    M = np.zeros((NUMTYPES, NUMPROFILES, NUMPROFILES))
    for A in range(NUMPROFILES):
        sA = PROFILES[A][0]
        costA = signal_cost if sA != 0 else 0.0
        for B in range(NUMPROFILES):
            sB = PROFILES[B][0]
            aA = perceived_action(A, sB)
            aB = perceived_action(B, sA)
            for t in range(NUMTYPES):
                M[t, A, B] = (coordination_preferences[t][aA] if aA == aB else punish) + costA
    return M


def profile_utilities(counts, M, multiplier=1.0):
    """Total utility U[t, A] of each profile A for a type-t agent *in* the population (self excluded when
    A is present), as executilcompute computes it. counts: (3, 81). Absent profiles get the utility a
    single entrant would get (no self term)."""
    n = counts.sum(0).astype(float)
    U = np.einsum('tab,b->ta', M, n)
    present = n > 0
    diag = M[:, np.arange(NUMPROFILES), np.arange(NUMPROFILES)]  # (3, 81): payoff against own profile
    U[:, present] -= diag[:, present]
    return multiplier * U


def nash_check(counts, M, min_share=0.0, tol=1e-9, eps=1e-3):
    """Unilateral-deviation check. For each type t and each incumbent profile P of t (share of t >= min_share),
    compare U_t(P) with the utility of one agent switching P -> A' for every A' (population n - e_P + e_A').
    Returns dict(is_nash (strict), max_gain (total-utility units), max_gain_per_interaction, is_eps_nash (gain per
    interaction <= eps; default 1e-3 = 10% of the signal cost -- at the integer state a split sits a few agents off its
    exact balance point, which shows up as gains of ~1e-4 per interaction toward the other sub-profile), details)."""
    n = counts.sum(0).astype(float)
    details = []
    worst = -np.inf
    for t in range(NUMTYPES):
        tot = counts[t].sum()
        for P in np.flatnonzero(counts[t]):
            if counts[t, P] < min_share * tot:
                continue
            u_P = M[t, P] @ n - M[t, P, P]
            n_wo = n.copy()
            n_wo[P] -= 1
            u_dev = M[t] @ n_wo  # deviator not in the population any more -> no self term
            gains = u_dev - u_P
            gains[P] = 0.0
            best = int(np.argmax(gains))
            details.append(dict(type=t, profile=int(P), count=int(counts[t, P]), U=u_P,
                                best_deviation=best, gain=float(gains[best])))
            worst = max(worst, gains[best])
    N = counts.sum()
    return dict(is_nash=bool(worst <= tol), max_gain=float(worst), max_gain_per_interaction=float(worst / (N - 1)),
                is_eps_nash=bool(worst / (N - 1) <= eps), details=details)


def split_eu(counts, split, M):
    """Mean per-interaction expected utility of the dominant vs minority behaviour class of a split type."""
    n = counts.sum(0).astype(float)
    N = n.sum()
    t = split['type']
    out = {}
    for role in ('dominant', 'minority'):
        ps = split[f'profiles_{role}']
        w = counts[t, ps].astype(float)
        u = np.array([M[t, P] @ n - M[t, P, P] for P in ps])
        out[role] = float((w @ u) / w.sum() / (N - 1))
    return out


def split_equilibrium(counts, det, M, eps=1e-3, delta=0.05, min_share=0.01):
    """Noise/N-robust equilibrium test for a flagged split ('balanced-split equilibrium'). Under mutation noise, or at
    small N, the integer state jitters around the split's balance point, so the plain epsilon-Nash test fails on switches
    between the two halves although nothing else pays (and at small N one agent of imbalance is worth ~alpha/N per
    interaction, which makes a fixed epsilon stricter). Here (1) the realized minority share must lie within `delta` of a
    stable zero of frequency_dependence (the split sits at a dynamically stable balance point) and (2) no unilateral
    deviation *other than* a switch between the two halves gains more than `eps` per interaction.
    Returns dict(ok, balance_dev, gain_other, gain_split) (gains per interaction)."""
    fd = frequency_dependence(counts, det, M)
    bal = min((abs(fd['realized'] - z) for z in fd['stable']), default=np.inf)
    t = det['splits'][0]['type']
    keys = [payoff_signature(p, t, det['partners'], det['M']) for p in range(NUMPROFILES)]
    md = np.array([k == det['splits'][0]['dominant'] for k in keys])
    mm = np.array([k == det['splits'][0]['minority'] for k in keys])
    N = counts.sum()
    g_other = g_split = 0.0
    n = counts.sum(0).astype(float)
    for tt in range(NUMTYPES):
        tot = counts[tt].sum()
        for P in np.flatnonzero(counts[tt]):
            if counts[tt, P] < min_share * tot:
                continue
            u_P = M[tt, P] @ n - M[tt, P, P]
            n_wo = n.copy()
            n_wo[P] -= 1
            gains = (M[tt] @ n_wo - u_P) / (N - 1)
            gains[P] = 0.0
            if tt == t and (md[P] or mm[P]):
                other_half = mm if md[P] else md
                g_split = max(g_split, float(gains[other_half].max(initial=0.0)))
                gains = np.where(other_half, 0.0, gains)
            g_other = max(g_other, float(gains.max()))
    return dict(ok=bool(bal <= delta and g_other <= eps), balance_dev=float(bal), gain_other=g_other, gain_split=g_split)


# ------------------------------------------------------------------------------------------------
# one grid point: simulate, analyse, save
# ------------------------------------------------------------------------------------------------
def screen_point(alpha, beta, runs, out_path, population_size=1000, thrds=20, profile_record_interval=400,
                 split_threshold=0.10, min_age=25, entropy=None, keep_all_series=True, recurrent_min_frac=0.5, **param_kw):
    """Run `runs` simulations at (alpha, beta), analyse every run and save one compressed .npz to `out_path`.
    'bipartite' = non-neutral split (detect_bipartite) that is stable (split_age >= min_age snapshots) and eps-Nash
    (unilateral deviations, incumbents >= 1% of their type; nash_gain stored per interaction, nash_strict = exact). Profile time series are kept for raw-flagged,
    neutral-split and unclassified ('other') runs. Returns a small summary dict."""
    P = make_params(alpha, beta, population_size=population_size, **param_kw)
    set_null_signal(P['null_signal'])
    set_num_signals(P['numsignals'])
    M = payoff_matrix(P['coordination_preferences'], P['signal_cost'])
    B = run_batch(P, runs, entropy=entropy, thrds=thrds, profile_record_interval=profile_record_interval)
    labels = np.array([classify(B['ss001'][r]) for r in range(runs)])
    dets = [detect_bipartite(B['final_counts'][r], M, threshold=split_threshold) for r in range(runs)]
    flagged = np.array([d['flag'] for d in dets])
    neutral = np.array([bool(d['neutral_splits']) for d in dets])
    first = [d['splits'][0] if d['flag'] else None for d in dets]
    split_type = np.array([s['type'] if s else -1 for s in first])
    minority_share = np.array([s['minority_share'] if s else 0. for s in first])
    signals_differ = np.array([s['signals_differ'] if s else False for s in first])
    age = np.array([split_age(B['profile_time'][r], dets[r], split_threshold) for r in range(runs)])
    nash_ok = np.zeros(runs, bool)
    nash_strict = np.zeros(runs, bool)
    nash_gain, eu_dom, eu_min = (np.full(runs, np.nan) for _ in range(3))
    desc = np.array([''] * runs, dtype=object)
    slabel = np.array([''] * runs, dtype=object)
    balance_dev, gain_other = np.full(runs, np.nan), np.full(runs, np.nan)
    eq_ok = np.zeros(runs, bool)
    for r in np.flatnonzero(flagged | neutral):
        nc = nash_check(B['final_counts'][r], M, min_share=0.01)
        nash_ok[r], nash_strict[r], nash_gain[r] = nc['is_eps_nash'], nc['is_nash'], nc['max_gain_per_interaction']
        if flagged[r]:
            se = split_equilibrium(B['final_counts'][r], dets[r], M)
            eq_ok[r], balance_dev[r], gain_other[r] = se['ok'], se['balance_dev'], se['gain_other']
            e = split_eu(B['final_counts'][r], first[r], M)
            eu_dom[r], eu_min[r] = e['dominant'], e['minority']
            desc[r] = describe_split(dets[r])
            slabel[r] = split_label(B['final_counts'][r], dets[r])
        else:
            desc[r] = 'NEUTRAL ' + describe_split(dets[r], 'neutral_splits')
    bipartite = flagged & nash_ok & (age >= min_age)
    bipartite_bal = flagged & eq_ok & (age >= min_age)
    rec = [recurrent_split(B['profile_time'][r], B['final_counts'][r], dets[r], M, split_threshold) for r in range(runs)]
    split_frac = np.array([x[0] for x in rec])
    rec_label = np.array([x[1] for x in rec], dtype=object)
    recurrent = (split_frac >= recurrent_min_frac) & ~(flagged & (age >= min_age))
    keep = np.arange(runs) if keep_all_series else np.flatnonzero(flagged | neutral | (labels == 'other') | recurrent)
    np.savez_compressed(out_path, alpha=alpha, beta=beta, N=population_size, runs=runs, entropy=B['entropy'],
                        mutation_rate=P['mutation_rate'], signal_cost=P['signal_cost'], null_signal=P['null_signal'],
                        num_signals=P['numsignals'], coordination_preferences=P['coordination_preferences'],
                        edm_payoff=float(P['coordination_preferences'][0, 2]), bipartite_bal=bipartite_bal, balance_dev=balance_dev,
                        gain_other=gain_other, split_label=slabel.astype(str),
                        split_frac=split_frac, recurrent=recurrent, rec_label=rec_label.astype(str),
                        recurrent_min_frac=recurrent_min_frac, recurrent_lower_bound=False,
                        labels=labels, flagged=flagged, neutral=neutral, bipartite=bipartite, split_age=age,
                        split_type=split_type, minority_share=minority_share, signals_differ=signals_differ,
                        nash_ok=nash_ok, nash_strict=nash_strict, nash_gain=nash_gain, eu_dom=eu_dom, eu_min=eu_min, desc=desc.astype(str),
                        freeze_step=B['freeze_step'], final_counts=B['final_counts'], ss001=B['ss001'],
                        kept_runs=keep, profile_time_kept=B['profile_time'][keep],
                        profile_record_interval=profile_record_interval, split_threshold=split_threshold,
                        min_age=min_age)
    return dict(bipartite=int(bipartite.sum()), bipartite_bal=int(bipartite_bal.sum()), recurrent=int(recurrent.sum()),
                flagged=int(flagged.sum()), neutral=int(neutral.sum()), other=int((labels == 'other').sum()))


def reanalyze_file(path, min_age=None, split_threshold=None):
    """Recompute the analysis fields of a saved screen_point .npz from its stored final counts with the CURRENT
    detector (no re-simulation) and rewrite the file. split_age needs the stored time series; a run that is newly
    flagged but whose series was not kept gets split_age = -1 (cannot count as bipartite). Returns summary dict."""
    d = dict(np.load(path, allow_pickle=True))
    a, b = float(d['alpha']), float(d['beta'])
    thr = float(d.get('split_threshold', 0.10)) if split_threshold is None else split_threshold
    ma = int(d.get('min_age', 25)) if min_age is None else min_age
    runs = int(d['runs'])
    M = point_matrix(d)
    fc = d['final_counts']
    kept = {int(r): i for i, r in enumerate(d['kept_runs'])}
    labels = np.array([classify(d['ss001'][r]) for r in range(runs)])
    dets = [detect_bipartite(fc[r], M, threshold=thr) for r in range(runs)]
    flagged = np.array([x['flag'] for x in dets])
    neutral = np.array([bool(x['neutral_splits']) for x in dets])
    first = [x['splits'][0] if x['flag'] else None for x in dets]
    age = np.zeros(runs, np.int64)
    for r in np.flatnonzero(flagged):
        age[r] = split_age(d['profile_time_kept'][kept[r]], dets[r], thr) if r in kept else -1
    nash_ok = np.zeros(runs, bool)
    nash_strict = np.zeros(runs, bool)
    nash_gain, eu_dom, eu_min = (np.full(runs, np.nan) for _ in range(3))
    desc = np.array([''] * runs, dtype=object)
    slabel = np.array([''] * runs, dtype=object)
    balance_dev, gain_other = np.full(runs, np.nan), np.full(runs, np.nan)
    eq_ok = np.zeros(runs, bool)
    for r in np.flatnonzero(flagged | neutral):
        nc = nash_check(fc[r], M, min_share=0.01)
        nash_ok[r], nash_strict[r], nash_gain[r] = nc['is_eps_nash'], nc['is_nash'], nc['max_gain_per_interaction']
        if flagged[r]:
            se = split_equilibrium(fc[r], dets[r], M)
            eq_ok[r], balance_dev[r], gain_other[r] = se['ok'], se['balance_dev'], se['gain_other']
            e = split_eu(fc[r], first[r], M)
            eu_dom[r], eu_min[r] = e['dominant'], e['minority']
            desc[r] = describe_split(dets[r])
            slabel[r] = split_label(fc[r], dets[r])
        else:
            desc[r] = 'NEUTRAL ' + describe_split(dets[r], 'neutral_splits')
    d.update(bipartite_bal=flagged & eq_ok & (age >= ma), balance_dev=balance_dev, gain_other=gain_other,
             split_label=slabel.astype(str))
    # recurrent splits: only computable for runs whose time series was kept -> a lower bound for older files
    rmf = float(d.get('recurrent_min_frac', 0.5))
    split_frac = np.zeros(runs)
    rec_label = np.array([''] * runs, dtype=object)
    for r, i in kept.items():
        split_frac[r], rec_label[r] = recurrent_split(d['profile_time_kept'][i], fc[r], dets[r], M, thr)
    d.update(split_frac=split_frac, rec_label=rec_label.astype(str), recurrent_min_frac=rmf,
             recurrent=(split_frac >= rmf) & ~(flagged & (age >= ma)), recurrent_lower_bound=len(kept) < runs)
    d.update(labels=labels, flagged=flagged, neutral=neutral, bipartite=flagged & nash_ok & (age >= ma),
             split_age=age, split_type=np.array([s['type'] if s else -1 for s in first]),
             minority_share=np.array([s['minority_share'] if s else 0. for s in first]),
             signals_differ=np.array([s['signals_differ'] if s else False for s in first]),
             nash_ok=nash_ok, nash_strict=nash_strict, nash_gain=nash_gain, eu_dom=eu_dom, eu_min=eu_min, desc=desc.astype(str),
             split_threshold=thr, min_age=ma)
    np.savez_compressed(path, **d)
    return dict(bipartite=int(d['bipartite'].sum()), bipartite_bal=int(d['bipartite_bal'].sum()),
                recurrent=int(d['recurrent'].sum()), flagged=int(flagged.sum()), neutral=int(neutral.sum()), age_unknown=int((age == -1).sum()))


# ------------------------------------------------------------------------------------------------
# run explorer (step 5): collect runs, exact fine-resolution re-runs, time series, balance curves
# ------------------------------------------------------------------------------------------------
def file_params(d):
    """make_params for a loaded screen_point .npz (or dict of it), including the non-default settings stored in it:
    mutation rate, signal cost, EDM payoff (files written before a field existed used the default)."""
    kw = {}
    if 'mutation_rate' in d:
        kw['mutation_rate'] = tuple(float(x) for x in np.asarray(d['mutation_rate']))
    if 'signal_cost' in d:
        kw['signal_cost'] = float(d['signal_cost'])
    if 'edm_payoff' in d:
        kw['edm_payoff'] = float(d['edm_payoff'])
    if 'null_signal' in d:
        kw['null_signal'] = bool(d['null_signal'])
    if 'num_signals' in d:
        kw['num_signals'] = int(d['num_signals'])
    if 'coordination_preferences' in d:
        kw['coord'] = np.asarray(d['coordination_preferences'], dtype=float)
    return make_params(float(d['alpha']), float(d['beta']), population_size=int(d['N']), **kw)


def point_matrix(d):
    """payoff_matrix for a loaded screen_point .npz, with the settings stored in it (file_params). Also sets NULL_SIGNAL to
    the file's model, so the detector / labels that follow use the right reading rule."""
    P = file_params(d)
    set_null_signal(P['null_signal'])
    set_num_signals(P['numsignals'])
    return payoff_matrix(P['coordination_preferences'], P['signal_cost'])


def collect_runs(paths, which='bipartite'):
    """One dict per selected run (boolean field `which`, default 'bipartite') across screen_point files, with the split
    recomputed from the stored final counts."""
    rows = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        M = point_matrix(d)
        for r in np.flatnonzero(d[which]):
            det = detect_bipartite(d['final_counts'][r], M, threshold=float(d['split_threshold']))
            if not det['flag']:
                continue
            s, ps = det['splits'][0], det['present_signals']
            rows.append(dict(
                alpha=float(d['alpha']), beta=float(d['beta']), N=int(d['N']), run=int(r), modal=str(d['labels'][r]),
                split_type=TYPE_NAMES[s['type']],
                dominant=behavior_str(effective_behavior(s['rep_dominant'], ps), ps), n_dom=s['n_dominant'],
                minority=behavior_str(effective_behavior(s['rep_minority'], ps), ps), n_min=s['n_minority'],
                minority_share=s['minority_share'], eu_dom=float(d['eu_dom'][r]), eu_min=float(d['eu_min'][r]),
                nash_gain=float(d['nash_gain'][r]), nash_strict=bool(d['nash_strict'][r]),
                split_age=int(d['split_age'][r]), path=path))
    return rows


def rerun_fine(path, run_ids, interval=10, thrds=8, cache_dir='results/explorer/fine'):
    """Exact re-simulation of runs `run_ids` of a screen_point file (same entropy -> same SeedSequence child; recording
    draws no RNG) with profile snapshots every `interval` steps. Raises if a re-run does not reproduce the stored final
    counts. Cached per run in `cache_dir`. Returns {run: dict(profile_time (3,81,K), profile_init, final_counts, ...)}."""
    import os
    d = np.load(path, allow_pickle=True)
    os.makedirs(cache_dir, exist_ok=True)
    tag = os.path.basename(path)[:-4]
    fn = lambda r: f'{cache_dir}/{tag}_r{r}_i{interval}.npz'
    todo = [int(r) for r in run_ids if not os.path.exists(fn(r))]
    if todo:
        P = file_params(d)
        B = run_batch(P, int(d['runs']), entropy=d['entropy'].item(), thrds=min(thrds, len(todo)),
                      profile_record_interval=interval, run_ids=todo)
        for i, r in enumerate(todo):
            if not np.array_equal(B['final_counts'][i], d['final_counts'][r]):
                raise RuntimeError(f'run {r} of {path} did not reproduce its stored final counts')
            np.savez_compressed(fn(r), profile_time=B['profile_time'][i], profile_init=B['profile_init'][i],
                                final_counts=B['final_counts'][i], freeze_step=B['freeze_step'][i], interval=interval)
    return {int(r): dict(np.load(fn(r))) for r in run_ids}


def type_class_masks(counts, det, t, top=2):
    """Masks over the 81 profiles for type t's `top` largest payoff-equivalence classes in the final state (for the
    split type these are the dominant and minority classes). Returns [(label, mask, n_final)]."""
    keyf = lambda p: payoff_signature(p, t, det['partners'], det['M'])
    keys = [keyf(p) for p in range(NUMPROFILES)]
    ps = det['present_signals']
    out = []
    for k, n, members in _classes(counts[t], keyf)[:top]:
        rep = members[int(np.argmax(counts[t, members]))]
        out.append((behavior_str(effective_behavior(rep, ps), ps), np.array([kk == k for kk in keys]), n))
    return out


def eu_series(profile_time, t, masks, M):
    """Per-interaction expected utility over time of the type-t agents inside each profile mask (nan while a mask is
    empty). profile_time: (3, 81, K). Same accounting as split_eu: U_P = M[t,P] . n - M[t,P,P]."""
    n = profile_time.sum(0).astype(float)  # (81, K)
    N = n[:, 0].sum()
    U = M[t] @ n - np.diag(M[t])[:, None]
    c = profile_time[t].astype(float)
    out = []
    for m in masks:
        tot = c[m].sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out.append(np.where(tot > 0, (c[m] * U[m]).sum(0) / tot / (N - 1), np.nan))
    return out


def frequency_dependence(counts, det, M, grid=801):
    """Balanced-split test. Hold the other types and the within-class profile mix fixed, put a share s of the split
    type's (dominant + minority) agents into the minority class, and compute dEU(s) = EU_minority - EU_dominant per
    interaction. A zero crossing from + to - is a *stable* balance point (the minority gains when rarer and loses when
    commoner -- negative frequency dependence); - to + is unstable. Returns dict(s, du, realized, stable, unstable)."""
    s0 = det['splits'][0]
    t = s0['type']
    keyf = lambda p: payoff_signature(p, t, det['partners'], det['M'])
    keys = [keyf(p) for p in range(NUMPROFILES)]
    md = np.array([k == s0['dominant'] for k in keys])
    mm = np.array([k == s0['minority'] for k in keys])
    c = counts.astype(float)
    wd, wm = c[t] * md, c[t] * mm
    nd, nm = wd.sum(), wm.sum()
    tot, N = nd + nm, c.sum()
    base = c.sum(0) - wd - wm
    diag = np.diag(M[t])
    s = np.linspace(0, 1, grid)
    du = np.empty(grid)
    for i, si in enumerate(s):
        U = M[t] @ (base + wd * ((1 - si) * tot / nd) + wm * (si * tot / nm)) - diag
        du[i] = ((wm @ U) / nm - (wd @ U) / nd) / (N - 1)

    def roots(sign):
        i = np.flatnonzero((sign * du[:-1] > 0) & (sign * du[1:] <= 0))
        return [float(s[j] + du[j] / (du[j] - du[j + 1]) * (s[j + 1] - s[j])) for j in i]

    return dict(s=s, du=du, realized=float(nm / tot), stable=roots(1), unstable=roots(-1), type=t)


def interaction_table(counts, min_frac=0.01):
    """Coordination outcomes between the behaviour groups present: within each type, substantial profiles (>= min_frac of
    the type) with the same outcome against every substantial partner are merged into one group (e.g. Akkad's
    selection-neutral sig0 variants). Returns (labels, table, groups) with table[i][j] = the action groups i and j
    coordinate on ('B'/'S'/'E') or '-' for a coordination failure; groups = [(type, rep_profile, members, count)]."""
    partners = substantial_profiles(counts, min_frac)
    groups = []
    for t in range(NUMTYPES):
        own = [int(p) for p in np.flatnonzero(counts[t] >= max(1, min_frac * counts[t].sum()))]
        by_sig = {}
        for p in own:
            by_sig.setdefault(outcome_signature(p, partners), []).append(p)
        for members in sorted(by_sig.values(), key=lambda m: -counts[t, m].sum()):
            rep = members[int(np.argmax(counts[t, members]))]
            groups.append((t, rep, members, int(counts[t, members].sum())))
    labels = [f"{TYPE_NAMES[t]} {profile_str(rep)}" + (f" +{len(m) - 1} equiv." if len(m) > 1 else '') + f" x{n}"
              for t, rep, m, n in groups]
    table = [[('BSE'[o] if o >= 0 else '-') for o in (interaction_outcome(p, q) for _, q, _, _ in groups)]
             for _, p, _, _ in groups]
    return labels, table, groups
