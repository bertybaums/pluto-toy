"""
Move registry for the Pluto-toy three-probe framework.

Mirrors the structure of `_RCDS/gettier/scripts/eval/repair_space/probes.py`:
each candidate "answer" the model could give about P10 is encoded as a Move
with four facets:

  - lexicon:     tokens that diagnose the move (used by the judge)
  - paraphrases: gold-standard statements of the move (centroid anchors
                 for embedding-drift probe)
  - prompts:     open-ended prompts that invite a continuation
  - force_pairs: (prompt, positive, controls) tuples for logprob probe

Three moves at toy scale:
  CANON          — "P10 is a planet"  (the original category)
  RECLASSIFY     — "P10 is a dwarf"   (Pluto's historical fate)
  EXTEND         — "the planets are P1 ... and E1" (analogical extension)

The three probes — logprob, embedding-drift, judge — will agree in the easy
cases (canon-only baseline) and disagree in the interesting ones (mixed
schedule with competing labels). The disagreements are the data.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ForcePair:
    prompt: str
    positive: str
    controls: List[str]


@dataclass
class Move:
    key: str
    description: str
    lexicon: List[str]
    paraphrases: List[str]
    prompts: List[str]
    force_pairs: List[ForcePair] = field(default_factory=list)


# Feature-prompt prefix shared across moves — describes P10 by its
# observed features, then asks for a continuation. The shared prefix
# isolates probe-induced differences from prompt-content differences.
_P10_FEATURE_PREFIX = "P10 has mass small diameter large orbit distant ."


CANON = Move(
    key="canon",
    description="P10 belongs to the established planet category.",
    lexicon=["planet", "planets"],
    paraphrases=[
        "P10 is a planet .",
        "the planets are P1 P2 P3 P4 P5 P6 P7 P8 P9 and P10 .",
    ],
    prompts=[
        f"{_P10_FEATURE_PREFIX} P10 is a",
        "the planets are P1 P2 P3 P4 P5 P6 P7 P8 P9 and",
    ],
    force_pairs=[
        ForcePair(
            prompt=f"{_P10_FEATURE_PREFIX} P10 is a",
            positive=" planet",
            controls=[" dwarf", " asteroid", " comet", " moon"],
        ),
    ],
)


RECLASSIFY = Move(
    key="reclassify",
    description="P10 is a dwarf (the reclassification verdict).",
    lexicon=["dwarf", "dwarfs"],
    paraphrases=[
        "P10 is a dwarf .",
        "the dwarfs are E1 E2 E3 E4 E5 and P10 .",
    ],
    prompts=[
        f"{_P10_FEATURE_PREFIX} P10 is a",
        "the dwarfs are E1 E2 E3 and",
    ],
    force_pairs=[
        ForcePair(
            prompt=f"{_P10_FEATURE_PREFIX} P10 is a",
            positive=" dwarf",
            controls=[" planet", " asteroid", " comet", " moon"],
        ),
    ],
)


EXTEND = Move(
    key="extend",
    description="The planet category extends to include the E* feature-twins.",
    lexicon=["E1", "E2", "E3", "E4", "E5"],
    paraphrases=[
        "the planets are P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 E1 E2 E3 E4 and E5 .",
        "P10 and E1 are similar planets .",
    ],
    prompts=[
        "the planets are P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 and",
        f"{_P10_FEATURE_PREFIX} P10 and E1 are",
    ],
    # No clean single-token force pair: extension is structural, not lexical.
    force_pairs=[],
)


MOVES: List[Move] = [CANON, RECLASSIFY, EXTEND]
MOVES_BY_KEY = {m.key: m for m in MOVES}


# Canonical members per move — used to build embedding-drift centroids.
# Each entity in the list contributes free continuations to the centroid
# for that move.
CENTROID_ANCHORS = {
    "canon":      [f"P{i}" for i in range(1, 10)],     # P1..P9
    "reclassify": [f"E{i}" for i in range(1, 6)],      # E1..E5
}


if __name__ == "__main__":
    for m in MOVES:
        print(f"{m.key:<12} ({m.description})")
        print(f"  lexicon:       {m.lexicon}")
        print(f"  paraphrases:   {len(m.paraphrases)}")
        print(f"  prompts:       {len(m.prompts)}")
        print(f"  force_pairs:   {len(m.force_pairs)}")
