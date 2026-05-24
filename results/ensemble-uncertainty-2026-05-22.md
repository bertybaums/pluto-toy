# Bootstrap-and-Retrain Uncertainty for the Tension Probe

*May 22, 2026*

## The problem with one run

Train the toy once on the overlap corpus, probe P10, and a single pair of
numbers comes back: P(planet) and P(dwarf) at the "P10 is a" slot. Read
category strain off that pair and you have committed to a story about what the
model learned. But train the toy again on a fresh draw of the same synthetic
world, holding architecture, schedule, and probe fixed, and the pair can move
from (0.87, 0.02) to (0.01, 0.88). The first reads as canon preserved. The
second reads as a label swap. The strain you reported was a property of the
draw, not of the model.

The April 28 three-probe study and the May 6 tension-vs-rote follow-up gave the
toy a probe that can, in principle, see strain: the dual-label probe reads
P(planet) and P(dwarf) as two independent values rather than forcing a winner.
What that work did not give us is a way to say whether a strain reading is
robust. Every cell in the May 7 sweep is a point estimate. So the question this
note takes up is narrow but load-bearing: when the toy reports tension on P10,
is that tension a stable feature of the setup, or an artifact of one run?

## What a single model's probabilities are, and are not

It is tempting to treat the probe's softmax spread as if it already carried the
uncertainty. It does not, at least not the uncertainty we care about. A trained
model's P(planet) and P(dwarf) describe outcome variability: given this one
model, how is next-token mass split across labels. That is aleatoric. The
question "how confident are we that P10 is under strain" is epistemic, and a
single model is silent on it, since it is one draw from the distribution over
models the training procedure could have produced.

The lesson is borrowed, not invented here. A sibling project (`_RCDS/unstructured/`)
faced the same split for a different task, recovering regression coefficients
from a verbalized survey, and made the move that recovers epistemic
uncertainty: bootstrap-and-retrain. Resample the data, retrain a fresh model on
each resample with the initialization seed held fixed, and take the spread
across the ensemble as the standard error of the estimate. Holding the seed
fixed is what isolates the epistemic part, since then only the data varies. The
finding there was that the language model recovers a standard error comparable
to the classical one, inflated by roughly a third because it carries
optimization noise the closed-form estimator does not, and only at B times the
compute. That is the line between a model used as a density estimator and a
model used as an inferential instrument.

## The axis the old sweep was missing

Here the toy has a wrinkle worth being precise about, because it changes what
"port the lesson" means. The existing sweep already retrains five seeds per
cell, and `aggregate_tension.py` already reports a standard deviation across
them. So error bars are not absent. But look at where they come from: the
corpus is generated once with `--seed 0` and held fixed across the five runs
(`sweep.py`), and the seeds vary only `torch.manual_seed`, that is, the model
initialization and the minibatch order (`train.py`). The spread the old sweep
reports is therefore optimization noise with the data held fixed. The corpus
draw, the very nuisance the robustness question is about, was never varied. The
whole 16-cell summary rests on a single realization of the synthetic world.

So we do not need to add error bars where there were none. We need to add the
axis the sweep left out, and then keep the two sources apart:

    Var_total  =  Var_epistemic   (vary the corpus, fix the init)
               +  Var_optimization (fix the corpus, vary the init)

The decomposition is the point. It lets us say not just how uncertain a strain
reading is, but which knob the uncertainty hides behind. And because the corpus
here is parametric, generated from a synthetic ontology rather than collected,
we can estimate the epistemic term two ways and check them against each other.

## Method

For each headline cell we run three ensembles of size B, each one a loop of
regenerate-or-resample, retrain, probe P10. The three differ only in what is
allowed to vary.

- **Parametric Monte Carlo.** Regenerate the corpus from a fresh
  `generate_corpus.py` seed, which redraws every entity's features and the
  sentence ordering, then retrain with the init seed fixed. Each member is a
  complete independent draw of the synthetic world, so we probe it against its
  own world's P10. This asks whether strain is a robust property of this kind
  of world across redraws.

- **Nonparametric bootstrap.** Take the realized seed-0 corpus and resample it
  with replacement, then retrain with the init seed fixed and probe against the
  fixed seed-0 entities. This mirrors `unstructured`'s fixed reference set: the
  target is held still, and only the observations the model trains on vary.
  Here the resampling unit matters. We resample at the level of statement
  pairs, keeping each "X has mass ..." line glued to its "X is a ..." label,
  because the model learns the feature-to-label association from adjacency.
  Resampling individual lines would shuffle labels away from their descriptions
  and destroy the very signal the probe reads.

- **Optimization.** Hold the seed-0 corpus fixed and vary the init seed. This
  is the component the old sweep already had, recomputed here at matched B so
  it sits on the same footing as the other two.

We hold the init seed fixed for the two epistemic ensembles so that their
spread reflects the data alone, following the unstructured design. Parametric
MC and the bootstrap estimate the same epistemic quantity by different means,
so their agreement is a cross-check rather than a redundancy: the parametric
draw resamples the whole world including P10's own features, while the
bootstrap resamples observations for a fixed P10. If the two give similar
spreads, the strain reading is robust to both how we resample and what we hold
fixed.

The headline cells, all `dwarf` mode at `p10_edge=1.0` and the d32l4 model,
are the canonical April-28 settings: `curriculum, overlap (r0/6)` as the cell
where strain should appear, `curriculum, rote-control (r6/6)` as the negative
where it should not, and `mixed, overlap` as a baseline where canon is held and
the tension index sits near zero. A method that discriminates robust from
fragile should return wide, ambiguous intervals on the curriculum cells and
tight ones on the mixed cell.

## Results

Three patterns come out of the B=20 ensembles, and two of them are not what the
single-run sweep led us to expect.

**The one run you did is a sample of size one.** Train the overlap cell once
with the seed-0 corpus and the seed-0 initialization, and P10 comes back at
P(planet)=0.87, P(dwarf)=0.02: canon preserved, no strain. That is the baseline
row below. But it is not where the mass is. Across twenty fresh draws of the
same world, P10 lands on dwarf (swap) in fifteen, on genuine tension in four,
and on canon in just one. The single run we happened to do sits in a tail that
one draw in twenty would reproduce.

| cell | one run (seed 0) | P(planet), parametric mean [2.5, 97.5] | P(dwarf) mean [2.5, 97.5] | modal class (/20) | TENSION frac (param / boot) |
|------|------------------|----------------------------------------|---------------------------|-------------------|------------------------------|
| overlap (curriculum) | canon, 0.87 / 0.02 | 0.15 [0.00, 0.78] | 0.70 [0.12, 0.93] | swap (15) | 0.20 / 0.15 |
| rote-control (curriculum) | canon, 0.99 / 0.00 | 0.40 [0.00, 0.99] | 0.37 [0.00, 0.87] | tension (9) | 0.45 / 0.35 |
| canon (mixed) | canon, 1.00 / 0.00 | 0.99 [0.97, 1.00] | 0.01 [0.00, 0.02] | canon (20) | 0.00 / 0.00 |

**The reading is fragile to both knobs, not just the corpus.** The smoke test at
B=2 suggested the corpus draw would dominate and the initialization seed would
be a sliver. At B=20 that is false: holding the corpus fixed and varying only
the init seed swings P10's P(planet) across nearly the whole interval too (the
overlap optimization ensemble has mean 0.18 with a 2.5-97.5 span of 0.00 to
0.89). The decomposition of the tension index makes the split explicit.

| cell | Var optimization (init) | Var epistemic (bootstrap) | Var epistemic (parametric) |
|------|--------------------------|---------------------------|----------------------------|
| overlap (curriculum) | 0.038 | 0.054 | 0.069 |
| rote-control (curriculum) | 0.120 | 0.016 | 0.056 |
| canon (mixed) | 0.0002 | 0.000 | 0.0001 |

So the old sweep, which varied only the init seed, was not measuring noise: it
was measuring a real and large source of variance. What it missed is the
epistemic term on top, and, more quietly, the fact that its mean was
conditional on a single corpus realization.

**The tension-vs-rote dissociation does not survive resampling.** This is the
result that matters most for the toy's purpose, and it cuts against the May 6
prediction. The plan was that the overlap cell would show tension and the
rote-control cell, with E* parked in a disjoint feature corner, would not. The
ensembles say close to the opposite: P10 classifies as tension in 20% of
overlap draws but 45% of rote-control draws under parametric MC, and 15% versus
35% under the bootstrap. The clean negative control is not clean. Whatever
separates genuine strain from rote at these settings, a single pair of sweep
cells did not capture it.

**The method discriminates.** It would be easy to worry that the wide intervals
are an artifact of the ensemble itself, that bootstrap-and-retrain just
manufactures spread. The mixed/canon cell rules that out: across all three
ensembles, sixty out of sixty draws classify as canon, P(planet) sits at 0.99
with a span no wider than 0.95 to 1.00, and every variance term is within
rounding distance of zero. When the signal is unambiguous the method says so.
The wide curriculum intervals are therefore the finding, not the noise floor.

**The two epistemic estimators agree.** Parametric MC resamples the whole world,
P10's own features included; the bootstrap resamples observations for a fixed
P10. They land close, with tension fractions of 0.20 versus 0.15 on overlap and
0.45 versus 0.35 on rote-control, both well above the optimization-only fraction
on rote-control. The cross-check holds, which is the small reassurance that the
spread is a property of the setup and not of one resampling scheme.

The figure (`fig-ensemble-2026-05-22.png`) shows all three: the dual-label
forest plot with its draw intervals, the variance decomposition, and the
classification votes that never resolve to a single colour on the curriculum
cells.

## Let's take stock

What did we buy with the B times the compute? Three things: the first
methodological, the last two about the toy itself.

The methodological point is the one we ported in, and it held. A single trained
model gives a point estimate with no epistemic uncertainty, and the only way to
know whether a category-strain reading is robust is to retrain and look at the
spread. The toy made this vivid in a way the survey task could not, because here
a single run did not just have a wide error bar, it landed on the wrong modal
answer. The overlap baseline reported canon; the ensemble reported swap. Had we
run the sweep once and written it up, which is in effect what the April 28 and
May 7 studies did, we would have reported whichever face P10 happened to show.

The first claim about the toy is that, at this scale and this prototype edge,
P10's category membership is bistable rather than blended. The strain does not
show up as one model holding both labels at once. It shows up as the model
collapsing to dwarf on most runs, to planet on a few, and to a genuine split on
a few more. The "both labels high" state we built the dual-label probe to catch
is real but rare within any single model: the conflict lives in the variance
across models, not in the mean of one. That is a sharper and more honest picture
than "P10 is under tension," and it is only visible once you retrain.

The second claim is the uncomfortable one. The tension-vs-rote dissociation, the
thing this toy was built to demonstrate, does not survive resampling at these
settings. The rote-control cell was supposed to be the clean negative; under the
ensemble it shows more tension than the overlap cell, not less. We should be
careful about what this does and does not mean. It does not mean the distinction
is unreal, since the conceptual case for it and the main pluto project stand on
their own. It means the present toy cells, at d32l4 and edge 1.0 under the
curriculum schedule, are too underpowered and too near a decision boundary to
exhibit it robustly. The single-run sweep made the dissociation look clean
because it sampled each cell once.

To be fair to the older work, the decomposition also shows that its error bars
were not meaningless. For the rote-control cell the optimization term, the one
the old sweep varied, carries most of the variance. The old bars were incomplete
rather than wrong: they missed the epistemic term, and they were computed around
a mean that was itself conditional on one corpus draw.

This maps onto the main project without modification. Pluto's premise-support
probe reads independent probabilities at a held-out slot, exactly as the
dual-label probe does here, and it too is currently a single-model measurement.
The move ports directly: resample, retrain, decompose, and report whether the
support is robust or an artifact of one run. The toy's lesson for the larger
study is that this is not optional bookkeeping, since on the one case we can
check, the single run was not merely uncertain but unrepresentative.

There is a host of details left to flesh out. B=20 is enough to see the
instability but coarse for the tails of the intervals. We held the prototype
edge and the model size fixed, and the synthetic world is parametric, so the
"epistemic" uncertainty here is Monte Carlo over a generative process rather
than sampling from a population. The most natural next step is the one the
May 6 plan already pointed at: sweep the prototype edge under the ensemble, and
ask whether a robustly-positive TENSION cell exists anywhere in the (edge,
schedule, size) space, or whether category strain at the prototype edge is
always, at small scale, a thing that lives in the variance. We take up exactly
that question in the postscript below.

## Postscript: the edge sweep (2026-05-23)

We ran the sweep the previous section called for: the overlap and rote-control
cells across prototype edges {0.5, 1.0, 1.5, 2.0, 3.0}, parametric MC at B=20,
with P10 probed against each draw's own world. The question was whether a
robustly-positive TENSION cell hides at some edge we had not tried. It does not,
and the way it fails is the informative part.

Across the whole range the rote-control cell classifies as tension about as
often as the overlap cell, and below edge 3.0 more often: tension fractions of
0.30 to 0.55 for rote-control against 0.20 to 0.30 for overlap. The two curves
cross only at the extreme planet end, edge 3.0, where overlap reaches 0.50 and
rote-control falls to 0.40, and even there both sit near a coin flip with
intervals that span almost the whole unit interval. The dissociation we were
looking for needs the overlap cell high and the rote-control cell low at the
same edge. No edge delivers that.

The dual-label curves say why. In the overlap cell P(dwarf) sits above
P(planet) at every edge, since E* share P10's distinctive distant-orbit feature
and pull it across. In the rote-control cell, where E* sit in a disjoint feature
corner and have no feature reason to touch P10, P10 still carries between 0.2
and 0.4 of dwarf mass. That is the tell: at this capacity the model cannot keep
a frequent label off P10. What the probe scores as "tension" in the rote-control
is not feature-driven conflict at all, but label-frequency leakage, the rote
phenomenon we built the control to exclude. The instrument cannot, at d32l4,
separate the two things it was meant to separate.

To be fair, there is a faint signal in the right direction. At edge 3.0 the
overlap cell finally leans more conflicted than the rote-control, which recovers
a canon majority. So the dissociation is not absent in principle; it is buried
under run-to-run instability and label leakage at the scale a laptop affords. We
take this as the boundary of the toy, honestly drawn: it is a clean
demonstration of the *method*, and a cautionary tale about reading a category
off one run, but it is not, at 50K parameters, a clean demonstration of the
tension-versus-rote distinction itself. That demonstration, if it exists, needs
more capacity or a sharper probe, and it is exactly the sort of claim the main
project should now make only with the ensemble in hand.

(Data: `results/edge_scan_summary_2026-05-22.json`; figure:
`results/fig-edge-sweep-2026-05-22.png`.)

## Postscript: capacity, and what the schedules actually track (2026-05-23)

Two further questions, both left open above, and in both cases the answer sharpens
the picture rather than softening it.

**Does more capacity help?** We re-ran the headline cells and the edge sweep at
the 300K model (d64l6, ~306K parameters), four times the size of the d32l4 model
above. It does not rescue the dissociation; it makes the failure more confident.
The overlap cell now swaps P10 to dwarf robustly, P(planet) near zero with a
near-unanimous bootstrap, at every prototype edge, including edge 3.0 where P10's
mass is as planetary as it gets. The rote-control stays a wide, multimodal mess,
and the canon cell stays solid. So capacity buys decisiveness, not discernment:
the larger model commits harder to one answer without committing any harder to
the features, and a tighter, more repeatable verdict is easy to mistake for a
truer one. Robustness is necessary for a finding. It is not the same as being
right.

**Which does each schedule track, the label or the features?** This is worth
pinning down, because the May 7 writeup put it too generously, calling curriculum
a feature-conditional regime. Set P10's probabilities on the two label words next
to what a shallow classifier infers from its features alone, across the
corner-mix:

| | overlap (0/6) | 2/6 | 4/6 | rote-control (6/6) |
|---|---|---|---|---|
| features imply (P_planet / P_dwarf) | 0.13 / 0.73 | 0.31 / 0.45 | 0.49 / 0.17 | 0.67 / 0.00 |
| mixed says | 0.99 / 0.01 | 0.99 / 0.01 | 0.99 / 0.00 | 1.00 / 0.00 |
| curriculum (50K) says | 0.28 / 0.68 | 0.28 / 0.65 | 0.24 / 0.70 | 0.38 / 0.56 |
| curriculum (300K) says | 0.10 / 0.89 | 0.12 / 0.87 | 0.11 / 0.87 | 0.42 / 0.50 |

Only the features row moves. Mixed sits on planet everywhere, the canon label,
even at full overlap where the features say dwarf: it holds the label and ignores
the evidence. Curriculum sits on dwarf everywhere, even at the rote-control where
the features put zero mass on dwarf: it tracks the frequency of the word in
phase 2, not what P10's features imply. The earlier reading, that curriculum
approximately tracks the feature baseline, mistook a shared direction at low
corner-mix for genuine feature-following, and the rote-control column is where
the two come apart. So neither schedule does the rational thing. One repeats the
old label, the other the frequent new word, and the feature evidence, which ought
to settle P10's kind, is a bystander in both.

(Data: `results/ensemble_summary_200k_2026-05-23.json`,
`results/edge_scan_summary_200k_2026-05-23.json`; figures:
`results/fig-ensemble-200k-2026-05-23.png`,
`results/fig-edge-sweep-200k-2026-05-23.png`.)
