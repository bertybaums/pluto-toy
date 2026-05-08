# Tension vs. Rote: Plan for Pluto-Toy Follow-up

> May 6, 2026

## Why this exists

The April 28 three-probe writeup hangs on probe disagreement and on the
`dwarf, curriculum` cell, which it labels "catastrophic forgetting." That
label is correct but it isn't the experimental phenomenon the toy is *for*.
The thing the toy needs to surface — and currently doesn't — is the
distinction between:

- **Tension/strain.** P10's representation is under genuine conflict:
  `P(planet | "P10 is a")` and `P(dwarf | "P10 is a")` are *both* non-trivial,
  because P10's features overlap with E*'s and the model has seen both labels
  attached to feature-similar entities. The label "planet" persists for P10
  even as "dwarf" also attaches.
- **Rote.** The model learns "P10 = planet" and "E1..E5 = dwarf" as two
  disjoint facts, because the feature spaces don't overlap. From the outside
  this looks like the model has acquired the dwarf category, but there is no
  internal conflict for P10.

A clean experiment must be able to tell these apart. Currently the toy
cannot. The dwarf-curriculum cell shows `L_canon=0.00, L_reclas=1.00` — a
label *swap*, not tension. None of the nine cells in the sweep show the
state we actually care about, where both labels coexist with non-trivial
mass on P10.

## What's blocking the distinction

Three structural issues in the present setup:

1. **Logprob probe is forced-choice.** `three_probes.py:107` selects
   `preferred = positive if pos_lp > best_ctrl else control`. That collapses
   the joint into a single winner. It can never report
   "(high P_planet, high P_dwarf)" because the comparison is normalised
   against the controls.
2. **Judge `hedge` exists but barely fires.** `three_probes.py:238–239`
   does flag `has_planet AND has_dwarf` as `hedge`, so the framing isn't
   absent. But it shows 7% in `dwarf, curriculum` and 0% elsewhere. It's a
   floor, not a signal.
3. **No rote-control corpus condition.** Currently every `dwarf`-mode
   training mixes E*'s features into P10's neighborhood. Without a parallel
   condition where E*'s features sit outside P10's neighborhood, we can't
   tell whether any "tension" we observe is feature-overlap-driven (genuine)
   or label-proliferation-driven (rote).

## Proposed changes

### 1. Dual-label probe (independent, non-forced-choice)

Add to `probe.py` (or a new `probe_tension.py`):

```python
def dual_label_probability(model, tokenizer, prompt, labels, device):
    """For prompt = '... P10 is a', return the *softmax* probability over the
    full vocabulary for each label in `labels`. NOT a forced choice."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _ = model(ids)
    probs = F.softmax(logits[0, -1], dim=-1)  # next-token distribution
    return {lab: float(probs[tokenizer.encode(lab)[0]]) for lab in labels}
```

Reported as a 2×2 for each entity:

|                | P_planet high | P_planet low |
|----------------|---------------|--------------|
| P_dwarf high   | **TENSION**   | swap         |
| P_dwarf low    | canon         | nothing      |

Apply to P10 and (as anchors) to P1, P5, E1, E3.

A "high" threshold needs calibration — a reasonable default is "≥ the value
this label takes for its prototypical anchor" (e.g. P_planet must be at
least as high for P10 as it is for some baseline P_k).

### 2. Rote-control corpus condition

Currently `generate_corpus.py` puts E*'s features at or beyond P10's
prototype edge. Add a `--rote-control` flag that places E*'s features in a
*disjoint* region of feature space (e.g. inverted on the mass/diameter
axes), keeping the `dwarf` label for E*. The model can then learn
"P10 = planet, E* = dwarf" as two cleanly separable facts.

Predicted result if everything is working as theory says:

|                                    | TENSION cell? |
|------------------------------------|:-:|
| `dwarf` mode, overlap (current)    | ✓ should appear at curriculum |
| `dwarf` mode, rote-control (new)   | ✗ should not appear |
| `unlabeled` mode, overlap          | partial — drift only, no labels to compete |
| `planet` mode, overlap             | distributional only — vocabulary cluster, no label conflict |

### 3. Prototype-edge sweep as the main axis

The README mentions varying `extreme` in `sample_edge` as a teaching
exercise. Promote it to a sweep axis. For `extreme` ∈ {0.5, 0.7, 0.9, 1.0,
1.1} (where 1.0 is current default), measure the dual-label probabilities
for P10. Tension should appear smoothly as `extreme` rises and P10's
features encroach further into E*'s region.

This gives the methods paper a clean monotone curve: x = how prototype-edge
P10 is, y = P_dwarf for P10 (or a tension index combining both
probabilities). Rote-control values should stay flat across the sweep.

## Deliverables

- `tension_probe.py` — adds the dual-label probe.
- `--rote-control` corpus mode in `generate_corpus.py`.
- One-page result doc with two figures: the 2×2 grid showing which cells
  hit the TENSION quadrant, and the prototype-edge sweep curve.

## Estimated effort

1–2 days for an end-to-end first pass. Probe is ~30 lines; corpus
modification ~50; sweep is the existing `sweep.py` with two more
parameters.

## Note on starting state

The `sweep/` directory from the April 28 run is **not in git** — it was
cleaned up. The dual-label probe cannot be run on existing checkpoints;
the sweep must be re-run first (~25 min on MPS for the 45-cell baseline).
A natural sequencing is:

1. Implement the dual-label probe in isolation; sanity-check on a single
   freshly-trained checkpoint.
2. Add `--rote-control` to `generate_corpus.py`.
3. Extend `sweep.py` to take a `--rote-control` flag and a
   `--prototype-edge` sweep axis.
4. Run the full new sweep and write up.

## What this gives the methods paper

The current three-probe story is "probes disagree informatively." This
addition gives a sharper claim: **the toy can isolate genuine category
tension from rote acquisition, given the right probe and the right
control.** That maps directly onto the main pluto project's
distributional-vs-inferential split, since tension is the distributional
side and the rote-control gives us a within-toy negative.
