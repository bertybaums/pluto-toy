"""
Build tutorial-tension.ipynb, the from-scratch follow-up notebook.

Mirrors the spine of docs/index.html, but runnable: it trains a small
ensemble live so the reader watches a single confident verdict turn into a
distribution, then reads the label-vs-features comparison off the data.

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
    src = "\n".join(lines)
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(*lines):
    src = "\n".join(lines)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(
    "# Can you trust what a tiny model tells you about Pluto?",
    "",
    "### Train it once and it sounds sure. Train it again and the answer moves.",
    "",
    "*Bert Baumgaertner &middot; University of Idaho &middot; May 23, 2026*",
    "",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bertybaums/pluto-toy/blob/main/tutorial-tension.ipynb)",
    "",
    "---",
    "",
    "In 2006 the International Astronomical Union demoted Pluto. It did not move Pluto or remeasure it; it changed what we call it. There is an old line of Juliet's that bears on this: a rose by any other name would smell as sweet, the name idle and the features, the smell, all that matter. This notebook builds a toy that tests the idea on a tiny language model, and the model's answer is not the one Juliet would give.",
    "",
    "We build a small synthetic solar system in which one body, P10, is our Pluto: labeled a `planet`, but with features at the small, distant edge of the category. Then we introduce new `dwarf`-labeled evidence and train the model to say what P10 is. The catch, and the point, is that the answer turns on a coin we cannot see: which corpus we happened to draw, and which random weights the model happened to start from. So we will not train once. We will train an ensemble, here, live, and watch the verdict become a distribution.",
    "",
    "Two instruments do the work, both runnable below:",
    "",
    "1. A **dual-label probe** that reads `P(planet)` and `P(dwarf)` independently, so it can see a genuine conflict and not just a winner.",
    "2. **Bootstrap-and-retrain**: instead of one model, many, each on a fresh draw of the world, so the spread of their answers tells us how far to trust any single one.",
    "",
    "You do not need the v1 notebook first. The terms get defined inline; you only need the idea of a model assigning probabilities to the next word.",
))

CELLS.append(md(
    "## Setup",
    "",
    "On Colab the next cell clones the repo; locally it is a no-op. After it runs, the toy's modules (`generate_corpus`, `train`, `tension_probe`, `baseline_classifier`) are importable and the small model trains in about a minute per run on a Colab CPU.",
))
CELLS.append(code(
    "import os, sys, json",
    "from pathlib import Path",
    "from collections import Counter",
    "",
    "if not os.path.exists('generate_corpus.py'):",
    "    !git clone https://github.com/bertybaums/pluto-toy.git",
    "    os.chdir('pluto-toy')",
    "sys.path.insert(0, '.')",
    "",
    "import torch",
    "DEVICE = 'cuda' if torch.cuda.is_available() else (",
    "    'mps' if torch.backends.mps.is_available() else 'cpu')",
    "print(f'Running on {DEVICE}')",
))

# ── Part 1: the world, and P10's two definitions ──────────────────────
CELLS.append(md(
    "## Part 1. The toy world, and the fork inside P10",
    "",
    "The world has 28 canonical bodies across four categories (*planet*, *asteroid*, *comet*, *moon*), each described by three features: mass, diameter, and orbit, on a 0&ndash;4 scale. P10, the Pluto-analog, sits at `(small, large, distant)`, on the small-mass, distant-orbit edge of the planet category, but firmly labeled *planet*.",
    "",
    "Then we add **six new bodies, E1&hellip;E6, labeled *dwarf*.** A single knob, the **corner-mix** (`--n-eris-rote N`), sets where they sit: out of six, *N* go in a disjoint corner of feature space, and the rest sit on top of P10. Let's build the two extremes and look at them.",
))
CELLS.append(code(
    "for n_rote in [0, 6]:",
    "    out_dir = f'data_r{n_rote}'",
    "    !python3 generate_corpus.py --out-dir {out_dir} --mode dwarf --n-eris 6 --n-eris-rote {n_rote} --seed 0 > /dev/null",
    "    ents = json.load(open(f'{out_dir}/entities.json'))",
    "    print(f'=== corner-mix = {n_rote}/6 ===')",
    "    print(f\"  P10:  features = {ents['phase1']['P10']['features']}   label = planet\")",
    "    for name, info in ents['phase2'].items():",
    "        print(f\"  {name}:  features = {info['features']}   label = dwarf\")",
    "    print()",
))
CELLS.append(md(
    "It is worth pausing on P10, because the whole experiment turns on it. P10 is pinned down two ways at once: by a **label** (the canon says `planet`) and by its **features** (small, distant, on the edge). Most planets agree with themselves, label and features pointing the same way. P10 is built so that new evidence can pull them apart.",
    "",
    "You can see it in the numbers above. At **corner-mix 0/6** every dwarf sits on P10's features, so the *features* now say *dwarf* while the *label* still says *planet*: the fork is open. At **corner-mix 6/6** the dwarfs sit in a far corner, sharing nothing with P10, so the features say *planet* and agree with the label again. The question this notebook keeps circling is simple to state: when the label says one thing and the features say another, which does the model follow? It is the same fork the real Pluto debate turned on.",
))

# ── Part 2: training ───────────────────────────────────────────────────
CELLS.append(md(
    "## Part 2. Two ways of training",
    "",
    "We train the small (50K-parameter) model on the **overlap** corpus (`corner-mix = 0/6`), where the fork is open, under two schedules:",
    "",
    "- **Curriculum**: phase 1 (the canon) for 1500 steps, then phase 2 (the dwarfs) as a 300-step *fine-tune*, continuing from the weights it already has. This is how a student, or a real model, gets updated: textbook first, new findings after. Its hazard is *catastrophic forgetting*, the new data overwriting the old.",
    "- **Mixed**: shuffle phase 1 and phase 2 together and train on the union, so the canon is rehearsed throughout and never gets a chance to be forgotten.",
    "",
    "Watch the `loss` fall as each model learns.",
))
CELLS.append(code(
    "!python3 train.py --data-dir data_r0 --out-dir runs/curriculum_r0 --schedule curriculum --device {DEVICE} --seed 0",
))
CELLS.append(code(
    "!python3 train.py --data-dir data_r0 --out-dir runs/mixed_r0 --schedule mixed --device {DEVICE} --seed 0",
))

# ── Part 3: the dual-label probe ───────────────────────────────────────
CELLS.append(md(
    "## Part 3. The dual-label probe",
    "",
    "Now we ask each model what P10 is, with the prompt `\"P10 has mass small diameter large orbit distant . P10 is a ___\"`. A *forced-choice* probe asks only whether `planet` beat `dwarf`, which throws away the interesting case. The **dual-label probe** keeps both numbers, `P(planet)` and `P(dwarf)`, read independently, and drops each reading into a 2&times;2 grid:",
    "",
    "|                  | P(planet) high | P(planet) low |",
    "|------------------|:---:|:---:|",
    "| **P(dwarf) high** | TENSION (both in play) | swap (flipped) |",
    "| **P(dwarf) low**  | canon (still a planet) | nothing |",
    "",
    "A label is \"high\" when it clears an absolute floor of 0.10. So `(0.45, 0.50)` is TENSION, `(0.01, 0.99)` is a swap, `(0.99, 0.01)` is canon.",
))
CELLS.append(code(
    "for sched in ['curriculum', 'mixed']:",
    "    !python3 tension_probe.py --ckpt runs/{sched}_r0/ckpt.pt --entities data_r0/entities.json --out runs/{sched}_r0/tension.json --device {DEVICE} > /dev/null",
    "    p10 = json.load(open(f'runs/{sched}_r0/tension.json'))['targets']['P10']",
    "    pp, pd = p10['P_planet'], p10['P_dwarf']",
    "    cell = ('TENSION' if pp >= 0.1 and pd >= 0.1 else 'canon' if pp >= 0.1 else 'swap' if pd >= 0.1 else 'nothing')",
    "    print(f'{sched:11s}  P(planet)={pp:.2f}  P(dwarf)={pd:.2f}  ->  {cell}')",
))
CELLS.append(md(
    "Two single runs, two different verdicts: curriculum tends to *swap* P10 to dwarf, mixed *keeps* it a planet. It is tempting to write that down and move on. But each of those is one run, with one corpus draw and one random seed, and that is exactly the thing we should not trust yet.",
))

# ── Part 4: train it again ─────────────────────────────────────────────
CELLS.append(md(
    "## Part 4. Train it once. Now train it again.",
    "",
    "Here is the heart of the notebook. We take the curriculum schedule on the overlap corpus, the cell that just said *swap*, and we train it again. And again. Each time we draw a **fresh corpus** from the same world (a new random seed for the generator) and retrain from scratch, then read P10's verdict. Eight runs takes a few minutes; lower `B` if you are impatient.",
    "",
    "Watch the verdicts as they print. They will not agree.",
))
CELLS.append(code(
    "import matplotlib.pyplot as plt",
    "",
    "B = 8  # retrainings; each is a fresh draw of the same world",
    "draws = []",
    "for b in range(B):",
    "    d = f'data_draw{b}'",
    "    !python3 generate_corpus.py --out-dir {d} --mode dwarf --n-eris 6 --n-eris-rote 0 --seed {1000 + b} > /dev/null",
    "    !python3 train.py --data-dir {d} --out-dir runs/draw{b} --schedule curriculum --device {DEVICE} --seed 0 > /dev/null 2>&1",
    "    !python3 tension_probe.py --ckpt runs/draw{b}/ckpt.pt --entities {d}/entities.json --out runs/draw{b}/tension.json --device {DEVICE} > /dev/null",
    "    p10 = json.load(open(f'runs/draw{b}/tension.json'))['targets']['P10']",
    "    pp, pd = p10['P_planet'], p10['P_dwarf']",
    "    cell = ('TENSION' if pp >= 0.1 and pd >= 0.1 else 'canon' if pp >= 0.1 else 'swap' if pd >= 0.1 else 'nothing')",
    "    draws.append((pp, pd, cell))",
    "    print(f'run {b + 1:2d}:  P(planet)={pp:.2f}  P(dwarf)={pd:.2f}  ->  {cell}')",
    "",
    "print('\\ntally:', dict(Counter(c for _, _, c in draws)))",
))
CELLS.append(code(
    "cellcol = {'TENSION': '#d62728', 'canon': '#2ca02c', 'swap': '#ff7f0e', 'nothing': '#7f7f7f'}",
    "fig, ax = plt.subplots(figsize=(6, 6))",
    "for pp, pd, cell in draws:",
    "    ax.scatter(pp, pd, c=cellcol[cell], s=120, edgecolor='white', linewidth=1.3, zorder=3)",
    "ax.axvline(0.1, ls=':', c='gray'); ax.axhline(0.1, ls=':', c='gray')",
    "ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)",
    "ax.set_xlabel('P(planet) for P10'); ax.set_ylabel('P(dwarf) for P10')",
    "ax.set_title(f'{B} retrainings of the SAME condition, a fresh draw each time')",
    "ax.text(0.97, 0.97, 'TENSION', color=cellcol['TENSION'], ha='right', va='top', weight='bold')",
    "ax.text(0.03, 0.97, 'swap', color=cellcol['swap'], ha='left', va='top', weight='bold')",
    "ax.text(0.97, 0.03, 'canon', color=cellcol['canon'], ha='right', va='bottom', weight='bold')",
    "plt.show()",
))
CELLS.append(md(
    "The dots scatter. The verdict for P10 was never a number, it was a distribution, and the single run from Part 3 was just one sample from it, possibly an unrepresentative one. If we had trained once and published, we would have reported wherever that one dot happened to land.",
    "",
    "So the honest answer to \"is P10 under strain?\" is not a cell. It is the shape of this cloud. And the obvious next question is where the scatter comes from.",
))

# ── Part 5: what the wobble is made of ─────────────────────────────────
CELLS.append(md(
    "## Part 5. What the wobble is made of",
    "",
    "There are two independent coins in every run: **which corpus we drew** (a different sample of the world) and **which random weights** the model started from. We can measure each on its own. Hold the weights fixed and vary the corpus, and the spread is the data-draw, or *epistemic*, part. Hold the corpus fixed and vary the weights, and the spread is the optimizer's own noise. Training many models this way and reading off the spread is **bootstrap-and-retrain**.",
    "",
    "Running the full decomposition live would take hours, so we load the pre-computed summary (shipped in the repo) and read off the two pieces for the contested overlap cell and for the rote-control.",
))
CELLS.append(code(
    "ens = json.load(open('results/ensemble_summary_2026-05-22.json'))",
    "print('variance of P10 tension reading (B=20, small model):')",
    "print(f\"  {'cell':<14}{'corpus (epistemic)':>20}{'optimizer (seed)':>20}\")",
    "for cell in ['overlap', 'rote_control']:",
    "    dec = ens[cell]['decomposition']['tension_index']",
    "    print(f\"  {cell:<14}{dec['var_epistemic_boot']:>20.3f}{dec['var_optimization']:>20.3f}\")",
))
CELLS.append(md(
    "Both pieces are large: the answer wobbles whichever coin you flip. This matters beyond the toy. The cheap, standard way to put error bars on a result like this is to retrain a few seeds on one fixed corpus, which measures only the optimizer column and misses the corpus column entirely. Those error bars look reassuringly tight while hiding much of the real uncertainty.",
    "",
    "A natural hope is that a bigger model settles down. It does, but onto the wrong thing. At four times the size (the 300K model, in `results/ensemble_summary_200k_2026-05-23.json` and on the [interactive site](https://bertybaums.github.io/pluto-toy/)), the overlap cell swaps P10 to dwarf almost every run, decisively, at every corner-mix. More capacity buys *confidence, not discernment*: a crisper, more repeatable answer, which feels like the evidence has spoken when it may only be the schedule speaking louder. Low variance is necessary for a finding to be real. It is not sufficient.",
))

# ── Part 6: what the data alone says ───────────────────────────────────
CELLS.append(md(
    "## Part 6. What the data alone says, and what each model says",
    "",
    "One more check keeps us honest. Maybe the data is genuinely ambiguous and the model is faithfully reflecting that. So set the transformer aside: given P10's features, what does a plain classifier infer? We fit four shallow baselines (no sequences, no schedule, no fine-tuning) on the same labeled features and ask each the same question.",
))
CELLS.append(code(
    "from baseline_classifier import evaluate_baselines",
    "",
    "for n_rote in [0, 6]:",
    "    res = evaluate_baselines(f'data_r{n_rote}/entities.json')",
    "    print(f'=== corner-mix = {n_rote}/6 ===')",
    "    print(f\"  {'baseline':<22}P(planet)   P(dwarf)\")",
    "    for name in ['logreg', 'gnb', 'knn3', 'bayes']:",
    "        p, d = res[name]['planet'], res[name]['dwarf']",
    "        print(f\"  {name:<22}{p:>7.2f}   {d:>8.2f}\")",
    "    print()",
))
CELLS.append(md(
    "Every baseline tells the same clean story: at full overlap the features say P10 is a *dwarf*, at full rote-control they say *planet*, and none of them ever lands in tension. The data has one decisive answer at each setting; what moves is which answer.",
    "",
    "Now put the model rows next to the feature row. This table is the whole point of the notebook, so read it slowly. The model rows are **label-tracking**: the probability the model puts on the *words* `planet` and `dwarf`. The top row is **feature-tracking**: what the classifier infers from P10's *features* alone. The cells are `P(planet) / P(dwarf)`, across the corner-mix.",
    "",
    "|                         | on P10 (0/6) | 2/6 | 4/6 | far corner (6/6) |",
    "|-------------------------|:---:|:---:|:---:|:---:|",
    "| the **features** imply  | 0.13 / 0.73 | 0.31 / 0.45 | 0.49 / 0.17 | 0.67 / 0.00 |",
    "| **mixed** says          | 0.99 / 0.01 | 0.99 / 0.01 | 0.99 / 0.00 | 1.00 / 0.00 |",
    "| **curriculum** (50K) says | 0.28 / 0.68 | 0.28 / 0.65 | 0.24 / 0.70 | 0.38 / 0.56 |",
    "| **curriculum** (300K) says | 0.10 / 0.89 | 0.12 / 0.87 | 0.11 / 0.87 | 0.42 / 0.50 |",
    "",
    "Read across each row. Only the *features* row moves. **Mixed** sits on `planet` everywhere, the canon label, even at 0/6 where the features say dwarf: it holds the label and ignores the evidence. **Curriculum** sits on `dwarf` everywhere, even at 6/6 where the features put *zero* mass on dwarf: it is repeating the word that was frequent in phase 2, not reading P10's features. So neither schedule does the rational thing, which would be to follow the feature row. One repeats the old label, the other the new word, and the features are a bystander.",
))

# ── Part 7: takeaways ──────────────────────────────────────────────────
CELLS.append(md(
    "## Part 7. What we learned",
    "",
    "The toy set out to show that a small model could detect category strain. It taught something more useful, and more cautionary.",
    "",
    "1. **One run is one sample.** The single run said one thing; the ensemble said another. Train once and you may report a coin flip as a finding.",
    "2. **Cheap error bars hide the larger half.** A few seeds on a fixed corpus miss the data-draw uncertainty, which is often bigger.",
    "3. **Scale buys confidence, not discernment.** The larger model is more decisive, not more correct.",
    "4. **Neither schedule reasons from the features.** Mixed holds the label; curriculum repeats the frequent word; the feature evidence, the thing that ought to settle it, is a bystander.",
    "",
    "Underneath sits a genuine question, and the toy is a good place to feel its edges. A label invites inferences about what a thing is like, and later evidence can make those inferences misleading: a sky full of Eris-sized neighbors sharing Pluto's features. When the label and the updated features finally disagree, which one fixes the thing's real kind? The toy does not settle it. What it shows is sharper: the model's answer is not discovered from the features, it is decided by how we trained it. So was Juliet right, that a rose by any other name smells as sweet? For our model, no. The name was very nearly everything, and the smell barely registered.",
))

CELLS.append(md(
    "## Exercises",
    "",
    "1. **Watch the cloud tighten the wrong way.** In Part 4, change the training call to the large model (`--n-embd 64 --n-layer 6`). The dots should collapse onto *swap*. More capacity, less doubt, same wrong-by-the-label answer.",
    "2. **Vary the corner-mix.** Re-run Part 4 with `--n-eris-rote 6` (the rote-control). The dwarfs now share none of P10's features, yet P10 still leans dwarf. That leakage is the frequency effect, bare.",
    "3. **Isolate the other coin.** Change Part 4 to hold the corpus fixed (`--seed 0` on the generator) and vary the *training* seed instead (`--seed {b}` on `train.py`). How does that spread compare to the corpus-draw spread above?",
    "4. **Add a feature.** Give each body a fourth feature in `generate_corpus.py`, and predict whether it changes the picture before you test.",
))

CELLS.append(md(
    "## Where to go next",
    "",
    "- The interactive, non-runnable version of all of this is the [results explainer](https://bertybaums.github.io/pluto-toy/), with a \"train it again\" button that builds the Part 4 cloud under your cursor.",
    "- The technical write-ups, on GitHub: [April 28](https://github.com/bertybaums/pluto-toy/blob/main/results/three-probes-2026-04-28.md), [May 7](https://github.com/bertybaums/pluto-toy/blob/main/results/follow-up-2026-05-07.md), and the uncertainty study, [May 22&ndash;23](https://github.com/bertybaums/pluto-toy/blob/main/results/ensemble-uncertainty-2026-05-22.md).",
    "- The parent project, **Pluto Time Capsule**, runs the same kind of analysis at full scale, on 124-million-parameter models trained on real pre-2006 astronomy, and the discipline is the same: retrain, decompose, and report the spread rather than the run.",
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
