# CLAUDE.md — pluto-toy

This file provides guidance to Claude Code (claude.ai/code) when working in
the `pluto-toy/` repository.

## What this is

A controlled, runnable-on-a-laptop version of the main Pluto Time Capsule
experiment (sibling project at `_RCDS/pluto/`). Trains a ~50K-parameter
transformer on a synthetic two-phase corpus to study **category strain** —
whether a model can detect tension when its canonical category structure
conflicts with newly-introduced evidence.

This is also a teaching artifact (Colab tutorial in `tutorial.ipynb`). Keep
files short, operations visible, end-to-end runs under ~25 min on CPU/MPS.

## Current focus (as of 2026-05-22)

**Tension vs. rote distinction.** The April 28 three-probe writeup
(`results/three-probes-2026-04-28.md`) hangs on probe disagreement and on a
catastrophic-forgetting cell that it labels "category strain" but is
actually a label *swap*. The toy needs to surface the proper distinction:

- **Tension/strain:** P10 has high `P(planet | …)` *and* high `P(dwarf | …)`
  *simultaneously* — features overlap, labels conflict, the model is
  genuinely under conflict for P10.
- **Rote:** P10 = planet, E* = dwarf, learned as two disjoint facts because
  feature spaces don't overlap. No internal conflict.

The plan for adding this distinction is in
**`tension-vs-rote-plan-2026-05-06.md`** — read that first. It specifies:
1. A dual-label probe (independent softmax probabilities, not forced choice).
2. A `--rote-control` corpus mode where E*'s features sit in a disjoint
   region of feature space.
3. Promoting the prototype-edge slider to the main sweep axis.

**Pick up there.** Estimated 1–2 days for the full sweep + writeup.

**Update (2026-05-22) — ensemble uncertainty.** The tension-vs-rote sweep shipped
(May 7). A bootstrap-and-retrain follow-up, ported from `_RCDS/unstructured/`, now
puts error bars on P10's tension and decomposes them into corpus-draw (epistemic)
and init-seed (optimization) variance: see `ensemble_tension.py`,
`plot_ensemble.py`, and `results/ensemble-uncertainty-2026-05-22.md`. Headline
(B=20): the single-run sweep cells are *unrepresentative* (the overlap baseline
reads canon, the ensemble mode is swap), the tension-vs-rote dissociation does
*not* survive resampling at d32l4/edge1.0/curriculum (rote-control shows more
tension than overlap), and the canon/mixed cell is a tight positive control.
Strain at the prototype edge is bistable: it lives in the variance across runs.
**Next:** sweep `--p10-edge` under `ensemble_tension.py` to test whether a
robustly-positive TENSION cell exists anywhere in (edge, schedule, size) space.

## Repository state

- Git repo (own remote, separate from `_RCDS/`). Public-facing — Colab badge
  in README, intended as a teaching artifact.
- `sweep/` is **not checked in**. The April 28 results were generated from a
  sweep that has since been cleaned up. Re-running `sweep.py` is required
  before any probe work on existing checkpoints.

## File map

| File | Purpose |
|------|---------|
| `generate_corpus.py` | Build phase 1 (canon) + phase 2 (evidence) corpora from a synthetic ontology. Modes: `unlabeled`, `dwarf`, `planet`. Flags: `--rote-control` (E* features in disjoint corner), `--p10-edge` (P10's mass-axis sampling mean). |
| `tokenizer.py` | Word-level tokenizer fit on the union of both phases. |
| `model.py` | Minimal decoder-only transformer (Karpathy-style). Default ~50K params. |
| `train.py` | Three schedules: `canon-only`, `curriculum`, `mixed`. |
| `probe.py` | Three probes: direct labeling, embedding geometry, generation samples. |
| `three_probes.py` | Multi-probe framework (L = logprob, D = drift, J = judge). Used for the April 28 writeup. |
| `tension_probe.py` | **Dual-label probe.** Reads `P(planet)` and `P(dwarf)` independently at the next-token slot. Calibrates per-label thresholds from anchor entities; classifies P10 into the TENSION / canon / swap / nothing 2x2. May 6 addition. |
| `moves.py` | Move registry — CANON, RECLASSIFY, EXTEND. Each defines a lexicon, paraphrases, prompts, and force-pairs. |
| `sweep.py` | Sweep over mode × schedule × seed × rote-control × p10-edge. Default reproduces April 28 (54 cells); pass `--rote-controls 0 1 --p10-edges 0.0 1.0 2.0 3.0` to add the May 6 axes. |
| `aggregate.py`, `aggregate_three.py`, `aggregate_tension.py` | Aggregate per-cell results across seeds. |
| `plot_tension.py` | Three figures from `tension_summary.json`: cell-classification grid, prototype-edge curve, corner-mix curve. |
| `baseline_classifier.py` | Traditional statistical baselines (logistic regression, Gaussian naive Bayes, k-NN, hand-derived Bayes posterior) on the same labeled feature data the transformer trains on. May 7 follow-up addition. |
| `plot_baselines.py` | Comparison figure: baseline P(planet)/P(dwarf) vs transformer means across the corner-mix axis. |
| `ensemble_tension.py` | **Bootstrap-and-retrain uncertainty.** Per headline cell, runs three B-sized ensembles — parametric MC (regen corpus per seed, own-world probe), nonparametric bootstrap (resample the realized corpus at statement-pair level, fixed seed-0 probe), optimization (fixed corpus, vary init) — then decomposes `Var_total = Var_epistemic(corpus) + Var_optimization(init)` on P10's tension. Ported from `_RCDS/unstructured/`'s epistemic-uncertainty move. May 22 addition. |
| `plot_ensemble.py` | Three-panel figure from `ensemble_summary.json`: dual-label forest plot with percentile intervals, variance decomposition (optimization vs epistemic), and classification-vote composition. |
| `build_site.py` | Render `docs/index.html` from `docs/template.html` by embedding the latest sweep summary + baseline values + entity tables as inline JSON. Re-run after any change to the sweep data. |
| `docs/template.html` | Source for the interactive walkthrough (single self-contained page with Plotly.js plots). The string `/* __DATA__ */` is replaced with embedded JSON at build time. |
| `build_tutorial_tension.py` | Render `tutorial-tension.ipynb` (the v2 follow-up notebook) from a Python cell-list. Re-run after editing the notebook structure. |
| `tutorial-tension.ipynb` | v2 Colab walkthrough. Mirrors the structure of `docs/index.html` with live training, dual-label probe, baseline classifiers, and a pre-loaded sweep figure. |
| `results/tension_summary_2026-05-07.json` | Pre-aggregated 160-cell sweep summary, shipped in the repo so the v2 notebook (and any reader) can plot the headline figure without re-running the sweep. |

## Conventions

- **Reports must have dates in filenames:** `report-YYYY-MM-DD.md`,
  `analysis-YYYY-MM-DD.md`, `<topic>-YYYY-MM-DD.md`. (Global rule from
  `~/.claude/CLAUDE.md` and `_RCDS/CLAUDE.md`.)
- **Reports/results live in `results/`.** Plan docs, design notes live at
  the repo root.
- **Keep files short.** Every file in this repo is intentionally
  ~50–200 lines. If a new file would exceed that, factor it.
- **Never break the Colab tutorial.** `tutorial.ipynb` and the README's
  Quick Start should always run cleanly on a fresh laptop.
- **No emojis** in code or docs unless explicitly requested.

## Pointers to broader context

- **Main project:** `_RCDS/pluto/CLAUDE.md` — the full-scale 124M experiment
  this toy mirrors. The dual-label probe maps onto pluto's premise-support
  probe (independent probabilities, not forced choice).
- **Methods paper:** `~/Documents/claude-projects/complementary-inquiry/`.
  The toy is meant to give the methods paper a clean miniature
  demonstration; the tension-vs-rote distinction is what makes that
  demonstration sharp.
- **Persistent framing memory:** `pluto-tension-vs-rote.md` in the auto-memory
  index — captures the central distinction so it doesn't slip again.
- **Parent project conventions:** `_RCDS/CLAUDE.md` — date-stamping, HPC
  notes (irrelevant here — this runs on laptop), GitHub conventions.
- **Writing style:** `~/Documents/bert-writing-style-guide.md` — consult
  before drafting any prose for reports or paper-bound text.

## Quick start

```bash
# regenerate corpus and train one schedule
python3 generate_corpus.py --out-dir data --mode unlabeled
python3 train.py --out-dir runs/canon_only --schedule canon-only

# full April-28-style sweep (3 modes × 3 schedules × 5 seeds, ~25 min on MPS)
python3 sweep.py --device mps

# probe a single checkpoint
python3 three_probes.py --ckpt runs/canon_only/ckpt.pt \
    --entities data/entities.json \
    --out runs/canon_only/three_probes.json --device mps
```

Add `--device cuda` if running on a GPU box.
