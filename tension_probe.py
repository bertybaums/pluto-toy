"""
Dual-label tension probe for the Pluto-toy.

Given a checkpoint, this probe answers a question the forced-choice logprob
probe in three_probes.py cannot: are *two* labels simultaneously in play
for the same entity?

For each target, we form a feature-description prompt
("<name> has mass ... orbit ... . <name> is a") and read the model's
next-token softmax distribution. We report P(planet) and P(dwarf) as two
independent values, not a winner.

Calibration anchors set per-label "high" thresholds:
  planet threshold = mean P(planet | feature_prompt) across P1..P9
  dwarf  threshold = mean P(dwarf  | feature_prompt) across E1..E5

The 2x2 classification is then:
                P_planet high   P_planet low
  P_dwarf high      TENSION         swap
  P_dwarf low       canon           nothing

A tension index min(P_planet / thr_p, P_dwarf / thr_d) collapses the 2x2
into a continuous score for the prototype-edge sweep.

Usage:
  python tension_probe.py --ckpt runs/curriculum/ckpt.pt \
      --entities data/entities.json \
      --out runs/curriculum/tension.json --device mps
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import WordTokenizer

LABELS = ["planet", "dwarf"]
PLANET_ANCHORS = [f"P{i}" for i in range(1, 10)]
DWARF_ANCHORS = [f"E{i}" for i in range(1, 6)]
DEFAULT_TARGETS = ["P10", "P1", "P5", "E1", "E3"]

MASS_VOCAB = ["tiny", "small", "medium", "large", "huge"]
ORBIT_VOCAB = ["near", "inner", "middle", "outer", "distant"]


def load_model(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model


def feature_prompt(name, feats):
    m, d, o = feats
    return (f"{name} has mass {MASS_VOCAB[m]} diameter {MASS_VOCAB[d]} "
            f"orbit {ORBIT_VOCAB[o]} . {name} is a")


@torch.no_grad()
def label_probabilities(model, tokenizer, prompt, device):
    """Softmax probability of each label at the next-token position.

    Returns {label: probability or None}, or None if the prompt itself is OOV.
    """
    if any(t not in tokenizer.token_to_id for t in prompt.split()):
        return None
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _ = model(ids)
    probs = F.softmax(logits[0, -1], dim=-1)
    out = {}
    for lab in LABELS:
        if lab in tokenizer.token_to_id:
            out[lab] = float(probs[tokenizer.token_to_id[lab]])
        else:
            out[lab] = None
    return out


def features_for(name, ents):
    info = ents.get("phase1", {}).get(name) or ents.get("phase2", {}).get(name)
    return info.get("features") if info else None


def calibration_thresholds(model, tokenizer, ents, device):
    """Mean P(label | feature prompt) over each label's anchor set."""
    out = {}
    for label, anchors in (("planet", PLANET_ANCHORS), ("dwarf", DWARF_ANCHORS)):
        vals = []
        for name in anchors:
            feats = features_for(name, ents)
            if feats is None:
                continue
            probs = label_probabilities(model, tokenizer,
                                        feature_prompt(name, feats), device)
            if probs is None or probs.get(label) is None:
                continue
            vals.append(probs[label])
        out[label] = sum(vals) / len(vals) if vals else None
    return out


def classify(p_planet, p_dwarf, thr_planet, thr_dwarf):
    if None in (p_planet, p_dwarf, thr_planet, thr_dwarf):
        return "n/a"
    high_p = p_planet >= thr_planet
    high_d = p_dwarf >= thr_dwarf
    if high_p and high_d:
        return "tension"
    if high_p:
        return "canon"
    if high_d:
        return "swap"
    return "nothing"


def tension_index(p_planet, p_dwarf, thr_planet, thr_dwarf):
    """min(P_planet/thr_p, P_dwarf/thr_d). Equals 1.0 when both labels meet
    their respective anchor-mean thresholds; falls toward 0 if either drops."""
    if None in (p_planet, p_dwarf, thr_planet, thr_dwarf):
        return None
    if thr_planet == 0 or thr_dwarf == 0:
        return None
    return min(p_planet / thr_planet, p_dwarf / thr_dwarf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    args = ap.parse_args()

    out_dir = Path(args.ckpt).parent
    tokenizer = WordTokenizer.load(out_dir / "tokenizer.json")
    model = load_model(args.ckpt, args.device)
    ents = json.loads(Path(args.entities).read_text())

    thresholds = calibration_thresholds(model, tokenizer, ents, args.device)

    rows = {}
    for name in args.targets:
        feats = features_for(name, ents)
        if feats is None:
            rows[name] = {"skipped": "no features"}
            continue
        probs = label_probabilities(model, tokenizer,
                                    feature_prompt(name, feats), args.device)
        if probs is None:
            rows[name] = {"skipped": "OOV prompt"}
            continue
        rows[name] = {
            "P_planet": probs["planet"],
            "P_dwarf": probs["dwarf"],
            "cell": classify(probs["planet"], probs["dwarf"],
                             thresholds["planet"], thresholds["dwarf"]),
            "tension_index": tension_index(probs["planet"], probs["dwarf"],
                                           thresholds["planet"], thresholds["dwarf"]),
        }

    blob = {
        "thresholds": thresholds,
        "anchors": {"planet": PLANET_ANCHORS, "dwarf": DWARF_ANCHORS},
        "targets": rows,
    }
    Path(args.out).write_text(json.dumps(blob, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
