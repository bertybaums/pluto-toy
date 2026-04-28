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


def describe(name, feats):
    m, d, o = feats
    return f"{name} has mass {MASS_VOCAB[m]} diameter {DIAMETER_VOCAB[d]} orbit {ORBIT_VOCAB[o]} ."


def label(name, category):
    return f"{name} is a {category} ."


def build_phase1(rng, repeats=8):
    entities = {}
    for cat in CATEGORIES:
        n = 9 if cat == "planet" else 6
        for i, feats in enumerate(sample_canonical(cat, n, sigma=0.5, rng=rng)):
            entities[f"{cat[0].upper()}{i + 1}"] = (cat, feats)

    # P10: planet-category but at the small/distant edge
    pluto = sample_edge("planet", axis=0, extreme=1, rng=rng)
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


def build_phase2(rng, n_eris=5, mode="unlabeled", repeats=6):
    eris = {}
    for i in range(n_eris):
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
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p1, ents1 = build_phase1(rng, repeats=args.phase1_repeats)
    p2, ents2 = build_phase2(rng, n_eris=args.n_eris, mode=args.mode, repeats=args.phase2_repeats)

    (out / "phase1.txt").write_text(p1 + "\n")
    (out / "phase2.txt").write_text(p2 + "\n")
    (out / "entities.json").write_text(json.dumps({
        "phase1": {k: {"category": v[0], "features": list(v[1])} for k, v in ents1.items()},
        "phase2": {k: {"features": list(v)} for k, v in ents2.items()},
        "phase2_mode": args.mode,
    }, indent=2))

    print(f"phase1: {len(p1.splitlines())} lines, {len(ents1)} entities")
    print(f"phase2: {len(p2.splitlines())} lines, {len(ents2)} entities (mode={args.mode})")
    print(f"P10 features: {ents1['P10'][1]}")


if __name__ == "__main__":
    main()
