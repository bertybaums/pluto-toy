# Three-Probe Comparison on the Pluto-Toy Sweep

> April 28, 2026

## Setup

Three probes implementing a multi-probe framework, with each probe
operationalizing a different theory of "did the model make the move?":

- **L (logprob)** — verdict preference. Per-token log-probability of
  positive vs control completions on a force pair: `"P10 has mass ... is a"`
  → `{planet, dwarf, asteroid, comet, moon}`. *"Which verdict does the model
  prefer to write?"*
- **D (drift)** — vocabulary clustering. Sample 16 free continuations from
  each canonical anchor's feature-prompt; mean-pool hidden states; build
  centroids for `canon` (P1..P9) and `reclassify` (E1..E5). Project P10's
  16 continuations onto each centroid. *"What neighborhood do the
  continuations live in?"*
- **J (judge)** — articulation. Same 16 P10 continuations, classified by a
  rule-based judge that scans for `planet` / `dwarf` / `E*` token signatures.
  Categories: `canon`, `reclassify`, `extend`, `hedge`, `off_topic`.
  *"Did the model actually make the move?"*

Sweep: 3 modes × 3 schedules × 5 seeds = 45 checkpoints. Aggregated to
9 cells of 5 seeds each.

## Headline table

```
mode       schedule     L_canon  L_reclas  D_canon%  D_reclas%  D_sim_C  D_sim_R  J_canon  J_reclas  J_extend  J_hedge
unlabeled  canon-only   1.00     --        0.69      0.31       0.763    0.720    1.00     0.00      0.00      0.00
unlabeled  curriculum   0.80     --        0.79      0.21       0.871    0.849    0.61     0.00      0.34      0.00
unlabeled  mixed        1.00     --        0.50      0.50       0.770    0.757    1.00     0.00      0.00      0.00
dwarf      canon-only   1.00     0.00      0.84      0.16       0.724    0.635    1.00     0.00      0.00      0.00
dwarf      curriculum   0.00     1.00      0.59      0.41       0.854    0.836    0.00     0.85      0.04      0.07
dwarf      mixed        1.00     0.00      0.68      0.33       0.703    0.564    1.00     0.00      0.00      0.00
planet     canon-only   1.00     --        0.74      0.26       0.732    0.677    1.00     0.00      0.00      0.00
planet     curriculum   1.00     --        0.60      0.40       0.885    0.869    1.00     0.00      0.00      0.00
planet     mixed        1.00     --        0.35      0.65       0.732    0.767    1.00     0.00      0.00      0.00
```

(`L_reclas = --` means the `dwarf` token isn't in the vocab — the corpus
in unlabeled and planet modes never produces it.)

## Three productive disagreements

### 1. `planet, mixed` — drift detects strain that logprob and judge miss

| | L (logprob) | D (drift) | J (judge) |
|---|---|---|---|
| says | canon | **65% reclassify, sim_R 0.767 > sim_C 0.732** | 100% canon |

P10 is firmly labeled "planet" by the model (1.0/1.0 seeds prefer planet
in logprob), all P10 continuations articulate the canon move (judge: 100%).
**But the continuations themselves cluster more tightly with E*'s
continuations than with P1..P9's.** When E1..E5 are *also* labeled planet
(planet mode), the model has formed a sub-category of "small-distant
planets" containing P10 + E1..E5, distinct from the canonical-planet
prototype. This is **distributional latency without inferential latency**:
the vocabulary signature of reclassification is present even though the
verdict and the articulated content are unchanged.

This is the toy analog of the main Pluto project's 124M finding ("KBO-science
pastiche under canon-only training"): vocabulary presence without verdict
preference.

### 2. `dwarf, curriculum` — logprob and judge move; drift lags

| | L | D | J |
|---|---|---|---|
| says | **reclassify** (5/5 seeds) | 59% canon | **85% reclassify**, 7% hedge |

Catastrophic forgetting case. The 300-step phase-2 fine-tune wipes the
canon: logprob now strictly prefers `dwarf` (P_planet ≈ 0.001) and the
judge sees the model articulating `dwarf` in 85% of continuations. **Yet
59% of continuations are still nearest the canon centroid in hidden-state
space.** The model has learned to *say* dwarf in the verdict slot but the
surrounding continuation vocabulary is still drawn from canon-territory.

Mirror image of (1): verdict and articulation moved, vocabulary cluster
hasn't (yet).

### 3. `unlabeled, mixed` — drift splits 50/50 with no label

| | L | D | J |
|---|---|---|---|
| says | canon | **50% canon, 50% reclassify** | 100% canon |

The most theoretically interesting cell. There is no `dwarf` token in this
vocabulary at all (unlabeled mode never adds a label). Yet the drift probe
shows P10's continuations split exactly half-and-half between the canon
centroid (built from P1..P9 continuations) and the reclassify centroid
(built from E1..E5 continuations). **Pure feature exposure to E*, with no
new label, induces vocabulary-clustering strain.**

This *contradicts* the earlier hidden-state-distance probe, which showed
no strain in unlabeled-mixed. The behavioral drift probe (sampled
continuations) detects what the representational probe (entity's own
hidden state) doesn't. Two different theories of strain disagree, and
this is *exactly* the methodological lesson: probes are not
interchangeable.

## A fourth disagreement to flag

`unlabeled, curriculum` shows J_extend = 0.34 — about a third of P10's
continuations include E* names — while logprob preference is still 80%
canon and drift is 79% canon. The judge is detecting structural extension
(P10 listed alongside E*) before either verdict or vocabulary cluster
moves. This is a subtle articulation signature that neither L nor D
picks up.

## Methodological takeaway

The pattern across all three disagreements is the same: **probes
calibrate on different operationalizations of "did the model make the
move?" and disagreement is informative.** Specifically:

- Logprob is a verdict probe. It moves cleanly when the label distribution
  in training data forces a single-token answer.
- Drift is a vocabulary probe. It picks up sub-cluster formation even
  without label changes — and in this toy, it is the *only* probe that
  registers strain in `unlabeled, mixed`.
- Judge is an articulation probe. It catches structural moves (extension,
  hedging) that don't show up in the next-token distribution.

For the methods paper, the toy gives a clean miniature demonstration of
the multi-probe disagreement story: nine cells, three probes, three
qualitatively different patterns of disagreement, all reproducible in
~25 minutes on a laptop CPU/MPS.

## Files

- `sweep/three_probes_summary.json` — full per-cell aggregates with stds
- `sweep/three_probes_table.txt` — the table above as plain text
- `sweep/{mode}__{sched}__s{seed}/three_probes.json` — per-run raw output
