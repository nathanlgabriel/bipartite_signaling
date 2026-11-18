# Bipartite signaling equilibria in unstructured populations

Code for the poster **"Coordination Signaling: Bipartite Signaling Equilibria in Unstructured Populations"**
(Nathan Gabriel, Emory University; PSA [year]).

In Lewis–Skyrms signaling games, populations usually settle on one signaling system. Fulker et al. (2024) found
*hybrid* equilibria, in which a group splits into two complementary roles. In their model a dynamic network holds the
roles together. The poster shows a different route that works in a **randomly mixing population**: the generalized
Bach or Stravinsky game with identity signals, in the three-type case with attention and signal costs from Gabriel,
Bell & Smaldino (2026), whose supplement first recorded such splits. In it, one agent type can split stably between two
strategy profiles that signal and act differently. The split is an ε-Nash equilibrium (ε = .001), and the shares of the
two profiles settle where both earn the same.

This repository holds the two notebooks that produce the poster's data, and the code they need. **The simulated data
itself is not included:** every run is seeded from a fixed entropy listed in the notebooks, so running them re-creates
the data exactly. The notebooks are committed with their outputs, so the results can be read without running anything.

---

## What is here

```
final_sweep.ipynb           the poster's sweep: alpha .20-.36 x beta .04-.20 (step .02), 1,000 runs per point;
                            heatmap, kinds of bipartite outcome, trajectory re-runs
three_signal_sweep.ipynb    the same sweep with three non-null signals instead of two, 100 runs per point
                            ("Is this merely a consequence of not having enough signals?")
FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py
                            the original model code of Gabriel et al. (2026), unmodified; not imported, included
                            for comparison with the copy below
FNs_genBS_PSA_bipartite.py  the copy of it that the notebooks run: identical, plus one recording function
                            (made with Claude Code; see "Generative AI disclosure")
psa_bipartite_tools.py      parameters, seeded parallel runs, split detection, Nash check, labels
exploration_log.py          progress report written while a sweep runs; labels
split_anatomy.py, split_conditions.py   name each kind of split (final_sweep.ipynb)
isomorphism_check.py        kinds of bipartite outcome up to isomorphism (three_signal_sweep.ipynb)
results/final_sweep/entropy.json        the final sweep's 81 seed entropies (the notebook checks its list against it)
dsci_385_conda-lock.yml     the exact Python environment
```

## The model in one paragraph

The model is the Bach or Stravinsky game of Luce & Raiffa (1957), generalized to a population with signals, as in
Gabriel, Bell & Smaldino (2026). Agents interact at random. There are three agent types. Two agents coordinate when
they play the same action, and each type prefers to coordinate on a different action. Miscoordination pays 0. The
payoffs for coordinating, by agent type:

| agent type | Bach | Stravinsky | R&B |
|---|---|---|---|
| Bachites | 1 + α | 1 | .5 |
| Stravinskians | 1 | 1 + α | .5 |
| Millennials | 0 | 0 | 1 |

Bachites make up .33 + β of the population, Stravinskians .33 − β and Millennials .34, so β sets how much larger the
Bachites are than the Stravinskians.

The rules:
* At the start of an interaction, **both** agents broadcast a signal, and **both** condition their action on their
  partner's signal.
* Signals 1 and 2 (and 3, in the three-signal sweep) cost c = .001 per interaction. The **null signal**, 0, is free.
* An agent who broadcasts 0 does not attend to signals. It interacts with all other agents as if they had also
  broadcast 0, so its action cannot depend on its partner's signal. Agents who do not signal do not condition their
  actions on signals, and vice versa. The poster calls them *silent*.
* A strategy profile is the signal broadcast plus an action for each signal received: 81 profiles (324 with three
  non-null signals).
* Strategy profiles evolve by discrete replicator dynamics within each agent type. Mutation follows the original
  code's "smart" mutation (m0 = .01, m1 = .1).
* The population has 10,000 agents and runs for 80,000 timesteps. Strategy profiles are assigned at random at the start.

**Names in the code.** The code uses the names of the original paper:

| Code | Poster |
|---|---|
| type 0, "Umma" | Bachites |
| type 1, "Kish" | Stravinskians |
| type 2, "Akkadians" | Millennials |
| actions B, S, E | Bach, Stravinsky, R&B |
| signal 0 | no signal (silence); called "blind" in split labels |

So `Umma [blind | Kish-copy]` means: the Bachites split between a silent strategy profile and one that copies the
Stravinskians'.

**Bipartite (strict).** A run counts as bipartite when all three of these hold:
1. at the end, one agent type is split between two strategy profiles that signal or condition actions on signals
   differently (differences only on signals nobody sends don't count), and the less common one is used by ≥ 10% of
   that type;
2. the split has held for the last 10,000 timesteps (25 snapshots, 400 timesteps apart);
3. the final state is an ε-Nash equilibrium: no agent can gain more than ε = .001 per interaction by changing its
   strategy profile.

**Kinds of bipartite outcome** are counted up to isomorphism: an outcome is described by who broadcasts which signal
and what every pair of groups coordinates on (the split type's two strategy profiles count as two groups). Two
outcomes are the same kind if they match after relabelling the non-null signals, or after swapping Bachites with
Stravinskians together with Bach with Stravinsky (their payoffs mirror each other).

## Installation

Python 3.14 with numpy, numba, pandas, matplotlib and Jupyter, pinned in the lock file. With
[conda-lock](https://github.com/conda/conda-lock) or micromamba:

```bash
conda-lock install -n psa_bipartite dsci_385_conda-lock.yml
# or
micromamba create -n psa_bipartite -f dsci_385_conda-lock.yml
conda activate psa_bipartite      # or: micromamba activate psa_bipartite
```

The lock covers linux-64, linux-aarch64, osx-64, osx-arm64 and win-64. Everything was run and tested on Linux x86-64
only.

## Replicating the data

From the repository root, execute the notebooks (or open them in Jupyter and run all cells):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 final_sweep.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 three_signal_sweep.ipynb
```

| Notebook | Runs | Time on a 16-core / 32-thread machine, `THRDS = 20` | Writes |
|---|---|---|---|
| `final_sweep.ipynb` | 81 points × 1,000 | about 10 h (7 min per point) | `results/final_sweep/` (~450 MB) |
| `three_signal_sweep.ipynb` | 81 points × 100 | about 3 h (2.1 min per point) | `results/three_signal_sweep/` (~120 MB) |

* Set the number of worker processes with `THRDS` in each notebook's settings cell. Memory: a few GB.
* Each notebook saves one `.npz` file per grid point and skips points whose file already exists, so an interrupted
  run resumes where it stopped.
* While a sweep runs, a progress report `CURRENT_exploration_<...>.md` is kept in the repository root; at the end it
  moves to `archives/`.
* `final_sweep.ipynb` checks that two saved runs re-simulate exactly, and re-runs up to 10 runs per kind of split at
  the best point with snapshots every 25 timesteps for its trajectory figures.

**Seeding.** Every grid point has one fixed seed entropy, listed in the notebook. Run r of a point is seeded with
`numpy.random.SeedSequence(entropy).spawn(runs)[r]` (runs = 1,000 or 100), so every run is reproducible bit for bit.

### Re-creating a single run

No data file is needed. For example, run 17 of the final sweep's point α = .30, β = .12 (entropy from
`final_sweep.ipynb`), which takes about 20 seconds:

```python
import numpy as np
import psa_bipartite_tools as T

if __name__ == '__main__':          # needed in scripts: worker processes start with forkserver/spawn
    P = T.make_params(0.30, 0.12, population_size=10_000, mutation_rate=(0.01, 0.1), signal_cost=-0.001)
    B = T.run_batch(P, 1000, entropy=328864406873562150472989911165774464867, thrds=1, run_ids=[17],
                    keep_profile_time=False)
    counts = B['final_counts'][0]   # agents of each type (rows) using each of the 81 strategy profiles (columns)
```

For the three-signal sweep add `num_signals=4` to `make_params` (signal 0 plus three more) and use `100` runs.

## Data format (files the notebooks write)

One `.npz` per grid point, e.g. `results/final_sweep/N10000_a0.30_b0.12_runs1000_c0.001.npz`. Load with
`np.load(path, allow_pickle=True)`. Main fields:

| field | shape | meaning |
|---|---|---|
| `alpha`, `beta`, `N`, `runs`, `mutation_rate`, `signal_cost`, `edm_payoff`, `coordination_preferences`, `num_signals` | | parameters (`edm_payoff` = R&B value; `signal_cost` is stored as a negative number) |
| `entropy` | | seed entropy of the point |
| `final_counts` | runs × 3 × profiles | exact number of agents of each type using each strategy profile at the end |
| `profile_time_kept` | runs × 3 × profiles × 200 | the same counts every 400 timesteps |
| `bipartite` | runs | strict bipartite (the poster's count) |
| `split_label` | runs | kind of split, e.g. `Umma [blind \| Kish-copy]`, `Kish [B-in \| B-in]` |
| `split_type`, `minority_share`, `split_age` | runs | which type split, share of its less common strategy profile, how long the split held |
| `nash_ok`, `nash_gain` | runs | ε-Nash test and the largest gain from changing strategy profile (per interaction) |

Strategy index → strategy profile: `T.PROFILES[i] = [signal broadcast, action vs signal 0, vs signal 1, vs signal 2]`
(plus vs signal 3 with three non-null signals)
(0 = Bach, 1 = Stravinsky, 2 = R&B); `T.profile_str(i)` gives e.g. `sig1 [BBS]`. Each file stores its parameters, so
`T.file_params(np.load(path, allow_pickle=True))` rebuilds them.

## Generative AI disclosure

Generative AI (Claude Code, Anthropic's coding agent, running Claude Opus 5.5) was used in this project. A short
summary: the model and the question are mine; Claude Code wrote most of the code in this repository, ran and analysed
the simulations under my direction, and drafted the poster, which I revised over several rounds and then finished by
hand. A more detailed account follows, so that readers can judge what to trust, and as a small contribution to
developing norms for disclosing AI use.

**What did not come from AI.** The model, its original code and the observation that one agent type sometimes splits
stably come from Gabriel, Bell & Smaldino (2026) and its supplement. That code,
`FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`, predates this project and is included
unmodified. The research question (why these splits arise, and under which conditions) is mine.

**Code.** Claude Code wrote the other files here:
* `FNs_genBS_PSA_bipartite.py` is a copy of the original model code. Lines 1–1554 are identical to the original (check
  with `diff`). The block after them, marked `PSA_poster additions`, adds one function: the original main loop
  (`genBSreplicatorexec_full_play`) with recording added. It saves a snapshot of the exact strategy-profile counts every
  400 timesteps, which the 10,000-timestep stability test needs, and the exact final counts. It makes the same
  random-number calls in the same order as the original, and Claude Code checked that 20 seeds give bit-identical runs.
* The analysis modules (`psa_bipartite_tools.py`, `exploration_log.py`, `split_anatomy.py`, `split_conditions.py`,
  `isomorphism_check.py`) handle split detection, the ε-Nash test, labels and kinds up to isomorphism.
* Claude Code wrote both notebooks, including their explanatory text. Each notebook re-simulates saved runs to check
  that they reproduce exactly. Where the explanatory text and the code disagree, the code is authoritative.
* [Author: say here what you checked by hand, if anything, e.g. the added recording function.]

**Research.** Claude Code did the exploratory work behind the poster over several days. It often worked on its own
between my check-ins and logged questions for me in a file. It designed and ran parameter screens and targeted sweeps,
wrote down predictions before several of them, proposed explanations and tested them against the simulations. The
balance equations on the poster are an example. I set the question and chose the settings the poster uses: signal cost
.001 rather than 0, 10,000 agents, the strict criterion for counting a run as bipartite, and the final grid. I also
decided which results to keep. The explanations on the poster were checked against the simulation data; for example,
in all 12,228 final-sweep runs of the most common kind, the predicted balance point is within 1.5 percentage points
of the observed split (99% of runs within 0.8).

**Poster.** Claude Code drafted the poster: the LaTeX source, the figures and the numbers quoted, all taken from the
final sweep. From that first draft to the version I took over took six rounds of my corrections and requests, plus one
request for an extra analysis (the three-signal sweep). Along the way, Claude Code asked several sets of clarifying
questions, which I answered. The changes were of these kinds:
* **Wording and framing.**
  * I dropped the paper's ancient-Sumerian "greeting" framing in favour of who *coordinates* with whom, and on which
    action.
  * The model is described in the paper's terms, and Luce & Raiffa (1957) are credited for the original game.
  * It now says "agent type" and "ε-Nash equilibrium", and the poster says explicitly that silent agents do not
    condition on signals.
  * The two strategy profiles of a split are no longer called "halves", since they are not half of the agent type.
* **Scientific accuracy and clarity.**
  * The coordination diagrams left out some pairs of groups (for example, Millennials with Stravinskians). They now
    show every pair, checked against the data.
  * Kinds of bipartite outcome had been named after which type splits, although Bachites and Stravinskians have
    mirror-image payoffs. They are now counted up to isomorphism and ranked by frequency.
  * One equation set used a variable that meant something different in another. Every variable now has one meaning.
  * The balance explanation was rewritten as one expected-payoff equation per strategy profile.
* **Simplicity.** Supporting analyses and extra figures were moved off the poster.
* **Visual design and layout.**
  * Line colours are now consistent: colour stands for the action, and signals are written as text.
  * The heatmap colour map and labels changed, and arrows show the direction of change.
  * Blocks were moved between columns, headings shortened to fit one line, the logo removed, and the title moved down
    so it is not cut off.

After Claude Code finished, I made further edits to the poster by hand.

**Writing.** This README was drafted by Claude Code.

## References

* Fulker Z, Forber P, Smead R, Riedl C (2024). Spontaneous emergence of groups and signaling diversity in dynamic
  networks. *Physical Review E* 109:014309. doi:10.1103/PhysRevE.109.014309
* Gabriel N, Bell A, Smaldino P (2026). The evolution of identity signals for co-ordination in diverse societies.
  *Evolutionary Human Sciences* 8:e17. doi:10.1017/ehs.2026.10037
* Luce RD, Raiffa H (1957). *Games and Decisions: Introduction and Critical Survey.* Wiley, New York.

## License and contact

[License — to be chosen.] Questions: Nathan Gabriel, nathan.gabriel@emory.edu.
