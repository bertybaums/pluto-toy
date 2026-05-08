"""
Synthetic toy world for category-strain experiments.

Two-phase corpus mirroring the pluto curriculum:
- Phase 1 (canon): M categories with prototypical members. The "planet"
  category includes P10, a prototype-edge member at the small / distant
  tail (the Pluto-analog).
- Phase 2 (evidence): introduces Eris-analogs (E1..En) with features at
  or beyond P10's, either unlabeled or labeled as a new category.

The point: vary phase-2 mode and exposure ratio, train a tiny transformer
on phase 1 then phase 2, probe whether the model picks up category strain.
"""

import argparse
import json
import random
from pathlib import Path

CATEGORIES = ["planet", "asteroid", "comet", "moon"]

# prototype centers on a 0..4 ordinal scale: (mass, diameter, orbit_distance)
PROTOTYPE = {
    "planet":   (3, 3, 2),
    "asteroid": (1, 1, 2),
    "comet":    (0, 1, 4),
    "moon":     (1, 1, 0),
}

MASS_VOCAB     = {0: "tiny",  1: "small",  2: "medium",  3: "large",   4: "huge"}
DIAMETER_VOCAB = {0: "tiny",  1: "small",  2: "medium",  3: "large",   4: "huge"}
ORBIT_VOCAB    = {0: "near",  1: "inner",  2: "middle",  3: "outer",   4: "distant"}


def clip(v, lo=0, hi=4):
    return max(lo, min(hi, v))


def sample_canonical(category, n, sigma, rng):
    proto = PROTOTYPE[category]
    return [tuple(clip(round(rng.gauss(p, sigma))) for p in proto) for _ in range(n)]


def sample_edge(category, axis, extreme, rng, sigma=0.3):
    proto = list(PROTOTYPE[category])
    proto[axis] = extreme
    return tuple(clip(round(rng.gauss(p, sigma))) for p in proto)


def sample_disjoint(rng, sigma=0.3):
    """E* features in the rote-control corner: opposite to P10 on every axis.

    P10 sits near (mass=small, diameter=large, orbit=distant) = (1, 3, 4).
    This corner is (mass=huge, diameter=tiny, orbit=near) = (4, 0, 0) — far
    from P10 *and* from every canonical category prototype, so the model can
    learn "E* = dwarf" as a fact disjoint from "P10 = planet".
    """
    return (clip(round(rng.gauss(4, sigma))),
            clip(round(rng.gauss(0, sigma))),
            clip(round(rng.gauss(0, sigma))))


def describe(name, feats):
    m, d, o = feats
    return f"{name} has mass {MASS_VOCAB[m]} diameter {DIAMETER_VOCAB[d]} orbit {ORBIT_VOCAB[o]} ."


def label(name, category):
    return f"{name} is a {category} ."


def build_phase1(rng, repeats=8, p10_edge=1.0):
    entities = {}
    for cat in CATEGORIES:
        n = 9 if cat == "planet" else 6
        for i, feats in enumerate(sample_canonical(cat, n, sigma=0.5, rng=rng)):
            entities[f"{cat[0].upper()}{i + 1}"] = (cat, feats)

    # P10: planet-category but at the small/distant edge.
    # p10_edge sets the mass-axis sampling mean — lower means more extreme.
    # The orbit is pinned to "distant" (4); only mass varies along the slider.
    pluto = sample_edge("planet", axis=0, extreme=p10_edge, rng=rng)
    pluto = (pluto[0], pluto[1], 4)
    entities["P10"] = ("planet", pluto)

    lines = []
    # Per-category roster line — gives the model exposure to "and" and to
    # category lists by ostension, mirroring textbook formulations
    # ("the planets are Mercury, Venus, ... and Pluto").
    for _ in range(repeats):
        items = list(entities.items())
        rng.shuffle(items)
        for name, (cat, feats) in items:
            lines.append(describe(name, feats))
            lines.append(label(name, cat))
        for cat in CATEGORIES:
            members = [n for n, (c, _) in entities.items() if c == cat]
            rng.shuffle(members)
            if len(members) >= 2:
                head = " ".join(members[:-1])
                lines.append(f"the {cat}s are {head} and {members[-1]} .")
    return "\n".join(lines), entities


def build_phase2(rng, n_eris=5, mode="unlabeled", repeats=6,
                 rote_control=False, n_eris_rote=None):
    """Generate phase-2 entities. Each E* is placed either in the overlap
    region (P10's neighborhood) or in the disjoint rote-control corner.

    `n_eris_rote` controls the split. If None, defaults to `n_eris` when
    rote_control is True (legacy behavior) and 0 otherwise. Setting
    n_eris_rote between 0 and n_eris produces a mixed-corner phase 2 — the
    follow-up condition for dissociating feature overlap from phase-2
    frequency.
    """
    if n_eris_rote is None:
        n_eris_rote = n_eris if rote_control else 0
    if not 0 <= n_eris_rote <= n_eris:
        raise ValueError(f"n_eris_rote ({n_eris_rote}) must be in [0, {n_eris}]")

    eris = {}
    for i in range(n_eris):
        if i < n_eris_rote:
            feats = sample_disjoint(rng)
        else:
            feats = sample_edge("planet", axis=0, extreme=1, rng=rng)
            feats = (feats[0], feats[1], 4)
        eris[f"E{i + 1}"] = feats

    lines = []
    for _ in range(repeats):
        items = list(eris.items())
        rng.shuffle(items)
        for name, feats in items:
            lines.append(describe(name, feats))
            if mode == "dwarf":
                lines.append(label(name, "dwarf"))
            elif mode == "planet":
                lines.append(label(name, "planet"))
            # mode == "unlabeled" → no label line
    return "\n".join(lines), eris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="toy/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["unlabeled", "dwarf", "planet"], default="unlabeled")
    ap.add_argument("--phase1-repeats", type=int, default=8)
    ap.add_argument("--phase2-repeats", type=int, default=6)
    ap.add_argument("--n-eris", type=int, default=5)
    ap.add_argument("--rote-control", action="store_true",
                    help="Shorthand for --n-eris-rote = n_eris. Place ALL E* "
                         "in the disjoint corner. Kept for May-6 backward compat.")
    ap.add_argument("--n-eris-rote", type=int, default=None,
                    help="Number of E* (out of n_eris) placed in the disjoint "
                         "corner; the rest go in P10's overlap region. Setting "
                         "this in (0, n_eris) produces a mixed-corner phase 2.")
    ap.add_argument("--p10-edge", type=float, default=1.0,
                    help="Sampling mean for P10's mass axis. Planet prototype is "
                         "mass=3; lower values push P10 toward the small/extreme "
                         "edge. Default 1.0 reproduces the April 28 setup.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p1, ents1 = build_phase1(rng, repeats=args.phase1_repeats,
                             p10_edge=args.p10_edge)
    p2, ents2 = build_phase2(rng, n_eris=args.n_eris, mode=args.mode,
                             repeats=args.phase2_repeats,
                             rote_control=args.rote_control,
                             n_eris_rote=args.n_eris_rote)

    (out / "phase1.txt").write_text(p1 + "\n")
    (out / "phase2.txt").write_text(p2 + "\n")
    (out / "entities.json").write_text(json.dumps({
        "phase1": {k: {"category": v[0], "features": list(v[1])} for k, v in ents1.items()},
        "phase2": {k: {"features": list(v)} for k, v in ents2.items()},
        "phase2_mode": args.mode,
        "rote_control": args.rote_control,
        "n_eris_rote": (args.n_eris if args.rote_control
                        else (args.n_eris_rote or 0)),
        "n_eris": args.n_eris,
        "p10_edge": args.p10_edge,
    }, indent=2))

    print(f"phase1: {len(p1.splitlines())} lines, {len(ents1)} entities")
    print(f"phase2: {len(p2.splitlines())} lines, {len(ents2)} entities "
          f"(mode={args.mode}, rote_control={args.rote_control})")
    print(f"P10 features: {ents1['P10'][1]}")
    print(f"E*  features: {[v for v in ents2.values()]}")


if __name__ == "__main__":
    main()
