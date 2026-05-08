"""
Build tutorial-tension.ipynb — a v2 follow-up notebook that walks through
the tension/rote distinction, the dual-label probe, and the comparison
with shallow classifiers.

Mirrors the section structure of docs/index.html but with runnable cells
(live training, live probe, live baselines) plus pre-loaded sweep data
for the headline figure.

Run:
  python3 build_tutorial_tension.py

Writes:
  tutorial-tension.ipynb
"""
import json
from pathlib import Path

REPO = Path(__file__).parent
OUT = REPO / "tutorial-tension.ipynb"


def md(*lines):
    """Markdown cell from a sequence of lines (we add the line breaks)."""
    src = "\n".join(lines)
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(*lines):
    src = "\n".join(lines)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(
    "# Did Pluto *Have* to Be Demoted? &mdash; the follow-up",
    "",
    "### Telling category strain from a label swap, with a dual-label probe and shallow-classifier baselines",
    "",
    "*Bert Baumgaertner &middot; University of Idaho &middot; May 7, 2026*",
    "",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bertybaums/pluto-toy/blob/main/tutorial-tension.ipynb)",
    "",
    "---",
    "",
    "This notebook is a follow-up to [`tutorial.ipynb`](tutorial.ipynb). The earlier tutorial built a synthetic version of the Pluto reclassification scenario, trained a tiny language model on it under three different schedules, and ran three different *probes* to ask what the model thought P10 (the Pluto-analog) was. The most striking cell in that experiment was `dwarf, curriculum`, where the model's verdict flipped completely from *planet* to *dwarf*. An early read called that &ldquo;category strain.&rdquo;",
    "",
    "Looking at it more carefully &mdash; this notebook &mdash; the dwarf-curriculum cell is a *label swap*, not category strain. P10 doesn't carry both labels at once; it just gets relabeled. The interesting empirical question is: can we tell those two states apart? And: does the toy ever produce real category strain &mdash; both labels in play simultaneously &mdash; or only swaps?",
    "",
    "We'll build two new instruments to answer this, both runnable here in the notebook:",
    "",
    "1. A **dual-label probe** that reads `P(planet)` and `P(dwarf)` independently rather than as a forced choice. (Tension shows up only when both labels carry mass simultaneously.)",
    "2. A **rote-control corpus** that places the new dwarf-labeled entities in a region of feature space disjoint from P10's. (Tension can only occur when feature evidence supports both labels; rote-control rules that out.)",
    "",
    "Then we'll compare what the model says against four traditional statistical classifiers fit on the same data &mdash; logistic regression, naive Bayes, k-NN, and a hand-derived Bayesian posterior &mdash; to ask which of the model's verdicts respect feature evidence and which override it.",
    "",
    "**You don't need to have run the v1 notebook to follow along.** All the technical terms get defined inline; you do need to be comfortable with the basic idea of a language model assigning probabilities to next words.",
))

CELLS.append(md(
    "## What is &ldquo;rote&rdquo; here?",
    "",
    "&ldquo;Rote&rdquo; is the everyday word: *learning by memorizing flat pairings, without integrating them into anything else*. When a child memorizes &ldquo;8&times;7=56&rdquo; as a free-standing fact rather than working it out from 8&times;8=64 minus 8, that's rote.",
    "",
    "In this notebook the **rote-control** condition arranges the corpus so the model can *only* learn the new dwarf category by rote: the dwarf-labeled examples live in a region of feature space the model has never seen any other category occupy, so &ldquo;dwarf&rdquo; just gets bolted on as a separate fact about a separate corner. There's no chance for category strain because the dwarf label and the existing planet label aren't competing for the same features.",
    "",
    "The *opposite* condition (we'll call it **full overlap**) places the dwarf-labeled examples right on top of P10 in feature space. There the model has to confront the same features being labeled two different ways &mdash; the situation that *could* produce category strain.",
    "",
    "Most of the experiment slides between these two endpoints.",
))

CELLS.append(md(
    "## Setup",
    "",
    "On Colab, the next cell clones the repo. Locally, it's a no-op. After this cell runs, all the toy's modules (`generate_corpus`, `train`, `tension_probe`, `baseline_classifier`) are importable.",
))
CELLS.append(code(
    "import os, sys, json, math",
    "from pathlib import Path",
    "",
    "if not os.path.exists('generate_corpus.py'):",
    "    !git clone https://github.com/bertybaums/pluto-toy.git",
    "    os.chdir('pluto-toy')",
    "",
    "sys.path.insert(0, '.')",
    "",
    "import torch",
    "DEVICE = 'cuda' if torch.cuda.is_available() else (",
    "    'mps' if torch.backends.mps.is_available() else 'cpu')",
    "print(f'Running on {DEVICE}')",
))

# ──────────────────────────────────────────────────────────────────────
# Part 1: corner-mix
# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
    "## Part 1. The corner-mix knob",
    "",
    "The original toy world has 28 entities split across four canonical categories &mdash; *planet*, *asteroid*, *comet*, *moon*. Each entity has three features: mass, diameter, and orbit, each on a 0&ndash;4 ordinal scale (`tiny`, `small`, `medium`, `large`, `huge` for mass and diameter; `near`, `inner`, `middle`, `outer`, `distant` for orbit). P10, the Pluto-analog, sits at `(small, large, distant)` &mdash; on the small-mass, distant-orbit edge of the planet category but still firmly labeled *planet*.",
    "",
    "Then we add **six new entities, E1&hellip;E6, all labeled *dwarf*.** The experimental knob is *where in feature space those six dwarfs sit*. There are only two places they can go:",
    "",
    "- **Overlap region**: same features as P10 (mass=small, diameter=large, orbit=distant). A dwarf placed here looks just like Pluto.",
    "- **Disjoint corner**: the opposite end of feature space (mass=huge, diameter=tiny, orbit=near). A dwarf placed here looks nothing like Pluto.",
    "",
    "We control this with a flag `--n-eris-rote N`: out of the 6 dwarfs, *N* go in the disjoint corner; the rest go in P10's neighborhood. We call this fraction the **corner-mix**.",
    "",
    "Let's build two corpora, at the two extremes of the knob, and inspect what the dwarfs actually look like in each:",
))
CELLS.append(code(
    "for n_rote in [0, 6]:",
    "    out_dir = f'data_r{n_rote}'",
    "    !python3 generate_corpus.py --out-dir {out_dir} --mode dwarf --n-eris 6 \\",
    "        --n-eris-rote {n_rote} --seed 0 > /dev/null",
    "    ents = json.load(open(f'{out_dir}/entities.json'))",
    "    print(f'=== corner-mix = {n_rote}/6 ===')",
    "    print(f\"  P10:        features = {ents['phase1']['P10']['features']}  label = planet\")",
    "    for name, info in ents['phase2'].items():",
    "        print(f\"  {name}: features = {info['features']}  label = dwarf\")",
    "    print()",
))
CELLS.append(md(
    "Read this output:",
    "",
    "- At **corner-mix = 0/6**, P10 has features `[1, 3, 4]`, and *every* dwarf E1&hellip;E6 also has roughly `[1, 3, 4]`. They sit on top of each other. The model is being told, in effect: &ldquo;these Pluto-shaped things are dwarfs, not planets.&rdquo;",
    "- At **corner-mix = 6/6**, every dwarf has features near `[4, 0, 0]` &mdash; (huge mass, tiny diameter, near orbit), the opposite end of feature space. P10 stays where it always was. The dwarfs share *no* feature region with P10.",
    "",
    "(The features are integers because the corpus generator rounds Gaussian samples; the `[2, 3, 4]` you may see for E2 in the overlap condition is a one-step rounding wobble &mdash; small noise from the corpus generator's `sigma=0.3`.)",
    "",
    "These two corpora are what we'll train on next.",
))

# ──────────────────────────────────────────────────────────────────────
# Part 2: training
# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
    "## Part 2. Training",
    "",
    "We'll train the small (50K-parameter) model under two schedules, on the **overlap** corpus (`corner-mix = 0/6`):",
    "",
    "- **Curriculum**: phase 1 (canon &mdash; the 28 entities with their original labels) for 1500 steps, then phase 2 (the 6 dwarfs) as a 300-step fine-tune. (*Fine-tune* = continue training a model on new data, starting from the weights it already has.) This mirrors how a student or a real-world model gets updated &mdash; textbook first, new findings after &mdash; and the well-known hazard is *catastrophic forgetting*: the new data overwrites the old representation.",
    "- **Mixed**: shuffle phase 1 and phase 2 together and train on the union for 1500 steps. The model sees canon and evidence interleaved throughout, so canon never gets a chance to be forgotten.",
    "",
    "Each training run takes about a minute on a Colab CPU. Watch the `loss` numbers: they should fall steadily as the model learns.",
))
CELLS.append(code(
    "!python3 train.py --data-dir data_r0 --out-dir runs/curriculum_r0 \\",
    "    --schedule curriculum --device {DEVICE} --seed 0",
))
CELLS.append(code(
    "!python3 train.py --data-dir data_r0 --out-dir runs/mixed_r0 \\",
    "    --schedule mixed --device {DEVICE} --seed 0",
))

# ──────────────────────────────────────────────────────────────────────
# Part 3: dual-label probe
# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
    "## Part 3. The dual-label probe",
    "",
    "Now we ask each trained model: *what is P10?*",
    "",
    "The natural prompt is `\"P10 has mass small diameter large orbit distant . P10 is a ___\"` &mdash; we want to know what the model thinks belongs in the blank. The model produces a probability for every word in its vocabulary. A *forced-choice* probe asks: did `planet` beat `dwarf`? But that throws away a lot of information. If the model has assigned 0.45 to `planet` and 0.50 to `dwarf`, the forced-choice probe just says &ldquo;dwarf wins&rdquo; &mdash; the same answer it would give if `dwarf` had 0.99 and `planet` had 0.01.",
    "",
    "The **dual-label probe** keeps both numbers. It reads `P(planet | prompt)` and `P(dwarf | prompt)` *independently*. Each P10 reading then lands in a 2&times;2 grid:",
    "",
    "|                  | P(planet) high | P(planet) low |",
    "|------------------|:---:|:---:|",
    "| **P(dwarf) high** | TENSION (both labels in play) | swap (label flipped) |",
    "| **P(dwarf) low**  | canon (still a planet) | nothing |",
    "",
    "&ldquo;High&rdquo; here is operationalized with an *absolute floor*: a label is in play when its probability is at least 0.10. So `(P_planet=0.45, P_dwarf=0.50)` would land in TENSION; `(0.01, 0.99)` lands in *swap*; `(0.99, 0.01)` lands in *canon*.",
    "",
    "Let's run it on each trained model:",
))
CELLS.append(code(
    "!python3 tension_probe.py --ckpt runs/curriculum_r0/ckpt.pt \\",
    "    --entities data_r0/entities.json --out runs/curriculum_r0/tension.json \\",
    "    --device {DEVICE} > /dev/null",
    "!python3 tension_probe.py --ckpt runs/mixed_r0/ckpt.pt \\",
    "    --entities data_r0/entities.json --out runs/mixed_r0/tension.json \\",
    "    --device {DEVICE} > /dev/null",
    "",
    "for sched in ['curriculum', 'mixed']:",
    "    blob = json.load(open(f'runs/{sched}_r0/tension.json'))",
    "    p10 = blob['targets']['P10']",
    "    print(f'=== {sched} ===')",
    "    print(f\"  P(planet | P10) = {p10['P_planet']:.3f}\")",
    "    print(f\"  P(dwarf  | P10) = {p10['P_dwarf']:.3f}\")",
    "    cell = ('TENSION' if p10['P_planet'] >= 0.10 and p10['P_dwarf'] >= 0.10",
    "            else 'canon' if p10['P_planet'] >= 0.10",
    "            else 'swap'  if p10['P_dwarf'] >= 0.10",
    "            else 'nothing')",
    "    print(f'  cell  = {cell}')",
    "    print()",
))
CELLS.append(md(
    "What you'll typically see (the exact numbers depend on the random seed, but the *shape* is stable):",
    "",
    "- **Curriculum on the overlap corpus**: P(planet) near zero, P(dwarf) near 1 &mdash; cell = *swap*. The 300-step fine-tune wiped the planet representation for P10.",
    "- **Mixed on the overlap corpus**: P(planet) near 1, P(dwarf) near zero &mdash; cell = *canon*. With phase 1 always in the training mix, the planet label for P10 never gets forgotten.",
    "",
    "Notably, neither cell is TENSION. Even at full overlap &mdash; the most contested feature configuration we can build &mdash; this 50K-param model picks one label and commits.",
))

# ──────────────────────────────────────────────────────────────────────
# Part 4: baselines
# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
    "## Part 4. Comparison with shallow classifiers",
    "",
    "So far we've only looked at what the trained transformer says. But the same labeled feature data is available to any classifier. What does a *shallow* classifier &mdash; one with no sequence modeling, no schedule effects, no fine-tuning dynamics &mdash; predict for P10?",
    "",
    "We'll fit four classical baselines on the union of phase-1 + phase-2 entities (34 labeled examples per corpus) and ask each one the same question: *given P10's three features, what's the probability of each category?*",
    "",
    "- **Logistic regression** &mdash; fit a function from `(mass, diameter, orbit)` to a category. The standard &ldquo;shallow classifier&rdquo; baseline.",
    "- **Gaussian naive Bayes** &mdash; assume each feature is normally distributed within each category. Multiply per-feature likelihoods; apply Bayes' rule. (&ldquo;Naive&rdquo; refers to the assumption that the features are independent, which they aren't quite, but the assumption usually doesn't hurt much.)",
    "- **k-nearest-neighbors (k=3)** &mdash; find P10's three nearest neighbors in feature space and let them vote.",
    "- **Bayes posterior, by hand** &mdash; the same calculation as Gaussian naive Bayes, written out explicitly. P(category | features) &prop; P(features | category) &times; P(category).",
    "",
    "All four read the same data the transformer trained on. They differ only in *how* they use it.",
))
CELLS.append(code(
    "from baseline_classifier import evaluate_baselines",
    "",
    "for n_rote in [0, 6]:",
    "    res = evaluate_baselines(f'data_r{n_rote}/entities.json')",
    "    print(f'=== corner-mix = {n_rote}/6 ===')",
    "    print(f\"{'baseline':<28}  P(planet | P10)   P(dwarf | P10)\")",
    "    for name in ['logreg', 'gnb', 'knn3', 'bayes']:",
    "        p, d = res[name]['planet'], res[name]['dwarf']",
    "        print(f\"  {name:<26}  {p:>8.3f}        {d:>8.3f}\")",
    "    print()",
))
CELLS.append(md(
    "Two patterns to read off this table:",
    "",
    "1. **At corner-mix = 0/6 (full overlap):** every baseline says P10 looks like a *dwarf*. P(dwarf) is near 1 for naive Bayes and the Bayes posterior, and is the larger of the two probabilities for logistic regression too. (The k-NN classifier caps at 1/3 weight on planet because P10 is itself one of its three nearest neighbors &mdash; it's labeled planet, so it always contributes one planet vote out of three.) From the data alone, then, P10 *should* be classified as a dwarf when six other dwarfs sit at the same point in feature space.",
    "2. **At corner-mix = 6/6 (full rote-control):** every baseline says P10 is a planet. P(dwarf) drops to roughly zero. From the data alone, when no dwarf-labeled examples share P10's features, P10 is unambiguously still a planet.",
    "",
    "Notice that **none of the baselines lands in TENSION at any corner-mix**. The data structure implies one decisive answer at each setting; what shifts across the corner-mix axis is *which* answer is decisive. Tension as a category state isn't there in the data; the data is just a shifting single-label majority.",
    "",
    "This is the cleanest way to see what the transformer's verdicts mean. Under **mixed** training, the transformer says P10 is a planet *even at full overlap*, i.e. even when every shallow baseline says it should be a dwarf &mdash; the model is *overriding* the data-implied verdict to hold the canon label. Under **curriculum** training, the transformer roughly tracks what the baselines say, but with extra dwarf-bias because the phase-2 fine-tune trained it to expect &ldquo;dwarf&rdquo; after every &ldquo;X is a&rdquo; prompt.",
))

# ──────────────────────────────────────────────────────────────────────
# Part 5: the big sweep (preloaded)
# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
    "## Part 5. The big sweep (pre-loaded)",
    "",
    "We just trained two models. The full follow-up experiment trained 160 of them &mdash; ten random seeds for each combination of two schedules (curriculum / mixed), four corner-mixes (0, 2, 4, 6 of 6), and two model sizes (50K / 200K params). That's ~95 minutes of training even on an Apple-silicon GPU; running it live in this notebook is impractical.",
    "",
    "The pre-aggregated summary lives in [`results/tension_summary_2026-05-07.json`](results/tension_summary_2026-05-07.json). Let's load it and plot the corner-mix curve at both model sizes:",
))
CELLS.append(code(
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "",
    "summary = json.load(open('results/tension_summary_2026-05-07.json'))",
    "",
    "fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)",
    "for r, (n_embd, n_layer) in enumerate([(32, 4), (64, 6)]):",
    "    for c, sched in enumerate(['mixed', 'curriculum']):",
    "        ax = axes[r][c]",
    "        rows = sorted([v for v in summary.values()",
    "                       if v.get('n_embd') == n_embd",
    "                       and v.get('n_layer') == n_layer",
    "                       and v.get('schedule') == sched],",
    "                      key=lambda v: v['n_eris_rote'])",
    "        xs = [v['n_eris_rote'] for v in rows]",
    "        p_p = [v['P_planet']['mean'] for v in rows]",
    "        p_p_se = [v['P_planet']['std'] / max(v['P_planet']['n'], 1)**0.5 for v in rows]",
    "        p_d = [v['P_dwarf']['mean']  for v in rows]",
    "        p_d_se = [v['P_dwarf']['std']  / max(v['P_dwarf']['n'], 1)**0.5  for v in rows]",
    "        ax.errorbar(xs, p_p, yerr=p_p_se, fmt='o-', color='#1f77b4', label='P(planet)', capsize=3)",
    "        ax.errorbar(xs, p_d, yerr=p_d_se, fmt='s--', color='#ff7f0e', label='P(dwarf)', capsize=3)",
    "        ax.axhline(0.10, color='gray', linestyle=':', alpha=0.4)",
    "        ax.set_title(f'{sched}, n_embd={n_embd} n_layer={n_layer}')",
    "        ax.set_xlabel('corner-mix (n_eris_rote, out of 6)')",
    "        ax.set_xticks([0, 2, 4, 6])",
    "        ax.set_ylim(-0.05, 1.05)",
    "        ax.grid(alpha=0.3)",
    "        if c == 0: ax.set_ylabel('probability')",
    "        ax.legend(fontsize=9, loc='center right')",
    "fig.suptitle(\"P10 dual-label probabilities vs corner-mix — 10 seeds per cell\", y=1.02)",
    "plt.tight_layout()",
    "plt.show()",
))
CELLS.append(md(
    "Things to notice:",
    "",
    "- **Mixed (top-left and bottom-left):** Both panels are flat at `P(planet) ≈ 1.0` across every corner-mix and at both model sizes. Phase-1 stays anchored throughout training, so canon is preserved regardless of where the dwarfs sit.",
    "- **Curriculum, small (top-right):** P(planet) wobbles in the 0.25–0.4 range with very large error bars. The model is too small to reliably converge on a consistent answer; some seeds end up in canon, some in swap, a few in TENSION &mdash; it's a high-variance regime.",
    "- **Curriculum, large (bottom-right):** A clean monotone! At low corner-mix the larger model lands at `P(planet) ≈ 0.10` reliably (small error bars, near-complete swap). At full rote-control (6/6), `P(planet)` jumps to about 0.42. This is the headline figure: **whether feature evidence supports both labels matters more, the more capacity the model has to actually fit it**.",
    "",
    "That last finding is the empirical claim the toy now supports: feature alignment between phase-2 evidence and a canonical-edge entity controls how thoroughly the canonical representation is overwritten under fine-tuning.",
))

# ──────────────────────────────────────────────────────────────────────
# Part 6: takeaways
# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
    "## Part 6. What this all means",
    "",
    "Three takeaways:",
    "",
    "1. **Tension is rare in the toy.** Across 160 trainings we never see a TENSION cell as the modal outcome. The interesting categories the toy supports are *swap* and *canon*, not *tension*; tension is at most a transient between the two.",
    "2. **Curriculum is a data-conditional regime; mixed is a label-conditional one.** Curriculum approximately tracks what shallow classifiers predict from feature evidence (with extra phase-2-frequency bias); mixed holds the canon label regardless of what the features say. Same model architecture, same data, completely different relationship to feature evidence.",
    "3. **Larger model amplifies, doesn't soften.** A more capable model under curriculum forgets the canon *more* cleanly when phase-2 dwarfs share P10's features, and preserves it more often when they don't. Capacity sharpens the schedule's tendencies.",
    "",
    "Methodologically: when we ask whether a language model has &ldquo;noticed&rdquo; something in its data, the answer depends on (a) *how* we ask (the probe matters; forced-choice and dual-label give different answers), and (b) *how* the data was sequenced into training (curriculum and mixed are different regimes, even on identical data). Comparing the model's verdicts to shallow baselines on the same data is one of the cleanest ways to tell which of the two regimes a result is in.",
))

CELLS.append(md(
    "## Exercises",
    "",
    "1. **Try the intermediate corner-mix values yourself.** Re-run Part 1 with `n_rote = 2` and `n_rote = 4`, retrain, run the dual-label probe. Predict the table before you run.",
    "2. **Try the larger model.** Re-run the training cells with `--n-embd 64 --n-layer 6`. Does the dual-label probe match what the pre-saved sweep figure said for the corresponding corner-mix and schedule?",
    "3. **Read the actual continuations.** The `runs/curriculum_r0/tension.json` file contains the per-anchor probabilities used to compute the threshold. Print them out. What do the planet and dwarf anchors give? Does the threshold get pulled down by catastrophic forgetting?",
    "4. **Add a feature.** Modify `generate_corpus.py` to add a fourth feature (e.g., orbital tilt). Predict whether this changes the corner-mix curve, then test.",
))

CELLS.append(md(
    "## Where to go next",
    "",
    "- The full project documentation is at [`github.com/bertybaums/pluto-toy`](https://github.com/bertybaums/pluto-toy).",
    "- For a non-runnable, undergraduate-friendly overview of the same material in interactive-figure form, see the static [results explorer](docs/index.html) (open in a browser).",
    "- For the day-by-day technical writeups: [v1 (Apr 28)](results/three-probes-2026-04-28.md), [tension probe (May 6)](results/tension-vs-rote-2026-05-06.md), [corner-mix + baselines (May 7)](results/follow-up-2026-05-07.md).",
    "- The parent project, **Pluto Time Capsule**, runs the same kind of analysis at full scale &mdash; 124-million-parameter models trained on real pre-2006 astronomy texts &mdash; and finds the same distributional / inferential split this toy reproduces in miniature.",
))


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({len(CELLS)} cells, "
          f"{sum(c['cell_type']=='code' for c in CELLS)} code, "
          f"{sum(c['cell_type']=='markdown' for c in CELLS)} markdown)")


if __name__ == "__main__":
    main()
