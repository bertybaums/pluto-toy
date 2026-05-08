# Pluto-Toy: Category Strain in Tiny Transformers

> April 28, 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bertybaums/pluto-toy/blob/main/tutorial.ipynb)

**Three undergraduate-friendly entry points:**

- [`tutorial.ipynb`](tutorial.ipynb) (Colab): the v1 narrative walkthrough &mdash; trains the model live and runs the original three probes.
- [`tutorial-tension.ipynb`](tutorial-tension.ipynb) (Colab): the v2 follow-up &mdash; introduces the dual-label probe, the corner-mix axis, and the comparison against four classical statistical baselines.
- [`docs/index.html`](docs/index.html): a non-runnable interactive results explainer covering the v2 methods and findings, with sliders and toggles over the corner-mix and model-size sweeps. Open the file directly in a browser, or serve it locally with `python3 -m http.server` from `docs/`.

The rest of this README is the technical reference for the codebase.

---

A controlled, runnable-on-a-laptop version of the Pluto Time Capsule experiment.
Where the main project asks whether *real pre-2006 astronomical discourse* contained
the latent signal of Pluto's reclassification, this toy asks the same question with
synthetic data and a 50K-parameter model: **can a tiny transformer detect category
strain when its canonical category structure conflicts with newly-introduced
evidence?**

This directory is also a teaching artifact. Every file is short, every operation
is visible, and the experiment runs end-to-end on a CPU in minutes.

---

## The setup, in one paragraph

We invent a small astronomical world. Phase 1 trains the model on a "canon" of
labeled entities: 10 planets (`P1..P9` plus `P10`), 6 asteroids, 6 comets,
6 moons. `P1..P9` are generic prototypical planets, sampled near the planet
centroid. `P10`, the Pluto-analog, sits at the prototype edge: its features
(small mass, large diameter, distant orbit) place it on the periphery of the
planet category, but the canon clearly labels it `planet`. Phase 2 introduces
`E1..E5` — Eris-analogs with feature vectors at or beyond `P10`'s, in one of
three modes: silent (`unlabeled`), competing label (`dwarf`), or analogical
extension (`planet`). We train, then probe whether the model's representation
of `P10` shifts.

## File map

| File | Purpose | Lines |
|------|---------|-------|
| `generate_corpus.py` | Build phase 1 (canon) and phase 2 (evidence) text corpora from a synthetic ontology. | ~140 |
| `tokenizer.py` | Word-level tokenizer fit on the union of both phases. | ~55 |
| `model.py` | Minimal decoder-only transformer (Karpathy-style). | ~140 |
| `train.py` | Three training schedules: `canon-only`, `curriculum`, `mixed`. | ~150 |
| `probe.py` | Three probes: direct labeling, embedding geometry, generation. | ~125 |

Read them in that order. The model and probes never talk to each other except
through a saved checkpoint and the saved tokenizer JSON, which is how production
ML pipelines factor too.

## Quick start

```bash
cd toy
python3 generate_corpus.py --out-dir data --mode unlabeled
python3 train.py --out-dir runs/canon_only  --schedule canon-only
python3 train.py --out-dir runs/curriculum  --schedule curriculum
python3 train.py --out-dir runs/mixed       --schedule mixed
for r in canon_only curriculum mixed; do
  python3 probe.py --ckpt runs/$r/ckpt.pt --entities data/entities.json --out runs/$r/probe.json
done
```

Add `--device mps` or `--device cuda` to use an accelerator. Default is CPU.

## What to read in the probe output

Each `probe.json` has three sections.

**A. Direct labeling** — `P("planet" | "P10 is a")` and analogous probabilities
for the other categories. The strain signature in the **curriculum** run with
`mode=dwarf` is: `P10`'s probability mass on `planet` falls and some leaks to
`dwarf`. With `mode=unlabeled`, no new label is available, so strain shows up
only in B and C.

**B. Embedding geometry** — pairwise cosine distances between input-embedding
rows for entity tokens. Read three quantities:
- `dist(P10, E1)` — should *shrink* after phase 2 (the model groups them by features).
- `dist(P10, P1)` — should *grow* (P10 drifts away from prototypical planets).
- `dist(E1, P1)` — telling: does `E1` end up planet-like or apart?

**C. Generation** — free continuations from juxtaposition prompts. These are
qualitative; you read them, you don't score them. With a 50K model and a
toy corpus, expect template-y outputs. The interesting move is when `P10 and E1
are` continues with feature words rather than category words.

## Glossary: toy → Pluto

The toy entities are not 1:1 stand-ins for specific real bodies — `P1..P9` are
generic samples from the planet prototype, not Mercury…Neptune. The structural
mapping is:

| Toy | Pluto-real |
|-----|-----------|
| `P1..P9` | Prototypical planets (the uncontested members of the canon) |
| `P10` | Pluto (the prototype-edge member labeled as a planet) |
| `E1..E5` | Eris-class KBOs (Eris, Haumea, Makemake, Sedna, …) |
| Phase 1 (canon) | Pre-1990s astronomy textbooks: planets stated by ostension |
| Phase 2 (evidence, `unlabeled`) | KBO discoveries 1992–2005 reported as observations without a category claim |
| Phase 2 (evidence, `dwarf`) | The 2006 IAU resolution — a new category competes for the same entities |
| Phase 2 (evidence, `planet`) | The "rejected" alternative — extend the planet category to include all KBOs |

## Three teaching exercises

1. **Vary the prototype-edge.** In `generate_corpus.py`, change `sample_edge`'s
   `extreme` parameter so `P10` sits closer to the prototype. Predict and verify:
   does strain still appear when `P10` is unambiguous?

2. **Vary the model size.** Re-run with `--n-embd 16 --n-layer 2` (~5K params) and
   `--n-embd 64 --n-layer 6` (~250K params). Where does strain detection emerge?
   Where does it saturate?

3. **Add a comparison operator.** In `probe.py`, build a probe that prompts
   `P10 and E1 are` and uses the next-token distribution as a measure of
   "perceived similarity." Compare across schedules. Does the model represent
   `P10`-`E1` as more similar than `P10`-`P1`? More similar than chance?

## Connection to the main project

The three running questions of the Pluto project are:

1. *Distributional* latency — does evidence shift the model's prose? **Found** at 124M.
2. *Inferential* latency — does the model take the logical step from evidence
   to recategorization? **Not found** up to 355M.
3. *Mechanism* — what would have to be added (architecture, training, tools) to
   close the gap?

This toy is mostly aimed at (1) and (3), at a scale where you can sweep cheaply.
A clean negative result here would be a *strong* result: it says strain detection
fails even in the maximally controlled case, which sets a meaningful lower bound
for the Pluto-real findings.

## Lineage

- Karpathy's `microgpt` → minimal transformer that learns character-level prose.
- `Numerals` → minimal transformer that learns arithmetic via scaffolded chain-of-thought.
- `Pluto-toy` (this) → minimal transformer under category strain.

Each step adds one layer of cognitive-science framing while staying in the same
"a laptop, an evening, and a small model" regime.
