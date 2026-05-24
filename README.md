# Pluto-Toy: can you trust what a tiny model tells you?

> May 23, 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bertybaums/pluto-toy/blob/main/tutorial.ipynb)

In 2006 the International Astronomical Union demoted Pluto. It did not move Pluto or remeasure it; it changed what we call it. So this is a test of an old line, Juliet's: a rose by any other name would smell as sweet, the name idle and the features all that matter. Is that true of Pluto?

This repository is a toy that asks a sharp version of the question. We build a synthetic solar system, give one body the planet label but edge-of-category features, introduce new "dwarf" evidence, and train a tiny language model to say what that body is. Train it once and it answers with confidence. Train it again, on a fresh draw of the same world, and the answer can flip. The toy turns out to be less about Pluto than about that instability: how a single training run can hand you a clean result that is really a coin flip, and what discipline it takes not to be fooled.

Everything runs on a laptop in minutes. Every file is short and every operation is visible, so the toy doubles as a teaching artifact.

## Three ways in

- **[Interactive explainer](https://bertybaums.github.io/pluto-toy/)** (also [`docs/index.html`](docs/index.html)): a non-runnable, undergraduate-friendly walkthrough with sliders and a "train it again" button that builds the distribution under your hands. Start here if you want the ideas without the code.
- [`tutorial-tension.ipynb`](tutorial-tension.ipynb) (Colab): the from-scratch follow-up. It builds the dual-label probe, trains an ensemble live, and watches a confident verdict dissolve into a distribution. This is the heart of the project.
- [`tutorial.ipynb`](tutorial.ipynb) (Colab): the gentler v1 on-ramp. It trains the model once and runs the original three probes.

## The setup, in one paragraph

Phase 1 trains the model on a canon of labeled bodies: nine generic planets (`P1`..`P9`), plus asteroids, comets, and moons. `P10`, the Pluto-analog, is the interesting one. It is pinned down two ways at once, by a label (the canon says `planet`) and by features (small mass, distant orbit, on the category's edge), and the two agree. Phase 2 then introduces six new bodies, `E1`..`E6`, labeled `dwarf`, and a single knob, the **corner-mix**, sets where they sit: stacked on top of `P10` in feature space (so the features now point at *dwarf*), or off in a disjoint corner (so they say nothing about `P10`). We train under three schedules, probe what the model thinks `P10` is, and ask which way it leans when the label and the features come apart.

The probe is the **dual-label probe**: it reads `P(planet)` and `P(dwarf)` independently rather than as a forced choice, so it can see a genuine conflict (both labels in play) and not just a winner.

## What we found

The toy set out to show that a tiny model could detect category strain. It taught something more useful, and more cautionary.

- **One run is one sample.** Train the contested cell once and `P10` comes back a confident planet. Retrain twenty times and that is a one-in-twenty outcome: the model usually swaps `P10` to dwarf. The single run was not just uncertain, it was unrepresentative.
- **Cheap error bars hide the larger half.** Retraining a few seeds on a *fixed* corpus measures the optimizer's noise and misses the data-draw uncertainty, which is often bigger. You only see it by redrawing the data and retraining.
- **Scale buys confidence, not discernment.** A four-times-larger model is more decisive, not more correct: it commits harder to one answer without committing any harder to the features.
- **The control backfired, informatively.** What the probe scores as "tension" is, at this scale, mostly a frequent label leaking onto `P10`, not feature-driven conflict. The toy cannot, at 50K or 300K parameters, cleanly separate the two.
- **Neither schedule reasons from the features.** Mixed training holds the canon label regardless of the evidence; curriculum training slides toward whatever word was frequent in phase 2, not toward what the features imply. Same data, opposite verdicts, and in both the feature evidence is a bystander.

So the toy is best read two ways: as a clean, runnable demonstration of a *method* (resample, retrain, decompose, report the distribution), and as a cautionary tale about trusting any single run. As for Juliet: for our model the name was very nearly everything, and the smell barely registered.

## File map

| File | Purpose |
|------|---------|
| `generate_corpus.py` | Build the phase-1 canon and phase-2 evidence corpora. Flags: `--mode`, `--n-eris-rote` (corner-mix), `--p10-edge`. |
| `model.py`, `tokenizer.py` | Minimal decoder-only transformer (~50K or ~300K params) and a word-level tokenizer. |
| `train.py` | Three schedules: `canon-only`, `curriculum`, `mixed`. |
| `tension_probe.py` | The dual-label probe: `P(planet)` and `P(dwarf)` read independently, classified into the TENSION / canon / swap / nothing 2x2. |
| `ensemble_tension.py` | **Bootstrap-and-retrain.** Trains B models per cell across three ensembles (fresh-corpus, resampled-corpus, fixed-corpus) and decomposes the uncertainty into data-draw vs. optimizer luck. |
| `sweep.py`, `aggregate_tension.py` | The single-corpus seed sweep and its aggregation (the May 7 results). |
| `baseline_classifier.py` | Four shallow classifiers (logistic regression, naive Bayes, k-NN, hand Bayes) on the same labeled features, for the label-vs-features comparison. |
| `plot_ensemble.py`, `plot_edge_sweep.py`, `plot_tension.py` | Figures. |
| `build_site.py`, `docs/` | Renders the interactive explainer from shipped `results/` data. |
| `build_tutorial_tension.py` | Renders `tutorial-tension.ipynb`. |
| `results/` | Dated write-ups and shipped JSON summaries, so figures reproduce without re-running the sweeps. |

## Quick start

```bash
# one corpus, one model, one probe
python3 generate_corpus.py --out-dir data --mode dwarf --n-eris 6 --n-eris-rote 0
python3 train.py --data-dir data --out-dir runs/curriculum --schedule curriculum
python3 tension_probe.py --ckpt runs/curriculum/ckpt.pt \
    --entities data/entities.json --out runs/curriculum/tension.json

# the honest version: don't train once, train many and look at the spread
python3 ensemble_tension.py --device mps --B 20 --cells overlap rote_control canon

# rebuild the interactive explainer from shipped results/ data
python3 build_site.py
```

Add `--device mps` or `--device cuda` to use an accelerator; the default is CPU.

## Glossary: toy to Pluto

The toy entities are not one-to-one stand-ins for real bodies; `P1`..`P9` are generic samples from the planet prototype, not Mercury through Neptune. The structural mapping is:

| Toy | Pluto-real |
|-----|-----------|
| `P1`..`P9` | The uncontested planets |
| `P10` | Pluto: labeled a planet, but on the category's edge |
| `E1`..`E6` | Eris-class bodies (Eris, Haumea, Makemake, Sedna, ...) |
| Phase 1 | Pre-1990s textbooks: the planets stated by ostension |
| Phase 2, `dwarf` mode | The 2006 reclassification: a new category competes for the same bodies |
| corner-mix | How much the new bodies' features overlap Pluto's |

## How this connects to the main project

This is a laptop-scale mirror of the **Pluto Time Capsule** project, which asks whether real pre-2006 astronomical writing already carried the latent signal of Pluto's reclassification, using 124-million-parameter models. The toy is aimed at the method and the cautions, not the headline: it gives the larger project a clean miniature in which to show that a single training run can manufacture a tidy result, and that the way to tell a real finding from an artifact is to retrain and decompose the spread. The dual-label probe here is the toy analog of that project's premise-support probe.

## Lineage

- Karpathy's `microgpt`: a minimal transformer that learns character-level prose.
- `Numerals`: a minimal transformer that learns arithmetic through scaffolded chain-of-thought.
- `Pluto-toy` (this): a minimal transformer under category strain, and what it takes to read it honestly.

Each step adds one layer of cognitive-science framing while staying in the same "a laptop, an evening, and a small model" regime.
