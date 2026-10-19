# EVAL-FINDINGS

## CLAIM_UNDER_TEST
Re-ranking the CTC n-best of a numeric-cell HTR model with constraints learned from train-split ground
truth (column format/range priors, temporal continuity, learned cross-field ordering, digit-confusion
candidates) raises exact-match accuracy on held-out TEST-split numeric cells by **>= 1.0 percentage point**
over the model's top-1, while changing **< 2% of the cells top-1 already got right**.

Thresholds were pre-registered in `validation.md` (Phase B) before any run. Weights/margin were chosen on the
**val** split only; the test split was scored once per run. Test = 3,922 numeric manuscript cells from 60
pages that appear in no other split (splits are page-disjoint, verified). All numbers below are copied from
`eval_results.json`.

## Runs, in order

| run | what | top-1 EM | re-ranked EM | gain | helped / harmed | harm rate | impossible values top-1 -> re-ranked | CER top-1 -> re-ranked | passes |
|---|---|---|---|---|---|---|---|---|---|
| run1 | first run, as pre-registered | 89.80% | 91.36% | +1.56 pp | 70 / 9 | 0.26% | 2.50% -> 1.73% | 4.46% -> 3.84% | yes |
| run2-bugfix | bug fix only (see below) | 89.80% | 91.43% | +1.63 pp | 70 / 6 | 0.17% | 2.50% -> 1.78% | 4.46% -> 3.83% | yes |

run1 is the ground-truth first result and is kept unchanged. The claim passed on the first run, so no
diagnostic-and-fix attempt was *required* by the gate; run2 is a voluntary bug-fix re-run, appended.

### run2 change (bug fix, not a threshold or eval change)
- **Diagnosis.** In `rerank`, a candidate string that does not parse as a number (e.g. `56.`) got `x = None`,
  which made its temporal and pairwise terms return 0 - while parsable candidates that disagreed with their
  neighbours paid a large negative temporal penalty. Unparsable strings therefore *dodged* the penalty and
  won in 2 test cells (`56` -> `56.`, `55` -> `55.`).
- **Change.** Skip unparsable candidates unless they are the HTR top-1. One change, no threshold or data touched.
- **Effect.** harmed 9 -> 6, gain +1.56 -> +1.63 pp. Val-selected weights shifted slightly (`rng` 0.5 -> 0), so
  part of the delta is the weight re-selection on val, not only the 2 fixed cells.

## What actually produces the gain (ablation, run2, test gain in pp over top-1)

| constraint | only this one | everything except this one |
|---|---|---|
| temporal continuity | **+1.50** | +0.31 |
| digit-format prior | +0.31 | +1.50 |
| range prior | 0.00 | +1.63 |
| cross-field pairwise ordering | 0.00 | +1.63 |

The gain is almost entirely **temporal continuity** (a cell must be plausible next to the same column's
neighbouring observations). Format helps a little on top. The range prior and the learned cross-field
ordering contribute nothing: val selected weight 0 for pairwise in both runs, and 220 pairwise rules were
learned but they are mostly magnitude orderings (e.g. barometer ~30 >= cloud-cover digit) that the HTR
model essentially never violates. **The headline "physics constraint" idea (wet <= dry etc.) is not what
earns the gain here.**

## Other facts needed to read the result
- Candidate oracle (is the truth anywhere in n-best + confusion candidates): 98.0% test. Ceiling is high;
  the constraints recover only ~1.6 of the ~8.2 points of available headroom.
- Per layout family (run2): Albatross 91.3% -> 93.7% (n=1987); vol003 88.2% -> 89.2% (n=1935).
- Per format (run1): `dd` 94.3 -> 96.0; `dd.dd` 89.4 -> 90.6; `d` 96.7 -> 97.9; `dd.d` (n=308) 54.2 -> 56.5 -
  decimal-temperature cells stay the weak spot, no constraint rescues them.
- Significance: exact two-sided McNemar on discordant cells, run2 70 vs 6, p = 6e-15. This treats cells as
  independent; cells on a page are correlated and only 60 test pages exist, so the p-value is overstated.
  Page-level bootstrap was not run.
- The 6 harmed cells are all genuine continuity over-corrections (a true `83` pulled to `80` because the
  neighbour read `78`; a true `1` pulled to `0` between two `0`s).
- Neighbour values used at test time are the model's own predictions, never ground truth.
- Cells whose transcription is a ditto mark (`"`) are not in the numeric set, so a temporal "neighbour" may be
  several observations away. Real handwritten-ditto handling was not built.

## Verdict
**CONFIRMED** - for the claim as pre-registered (aggregate, test split): +1.63 pp exact match at 0.17%
harm, both inside the pre-registered thresholds, on the first run and after the bug fix.
Qualification the reader must keep: the effect is small (+1.6 pp on a 89.8% baseline), driven by temporal
continuity only, and the cross-field-ordering part of the claim is not supported by the ablation.
