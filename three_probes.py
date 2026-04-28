"""
Three probes on the Pluto-toy moves, mirroring the Gettier repair-space
methodology. Each probe answers a different question about the same
checkpoint:

  Probe L (logprob)   — verdict preference. Per-token logprob of
                        positive vs control completions on a force pair.
                        Asks: "given the prompt, which verdict is the model
                        most willing to write?"

  Probe D (drift)     — vocabulary clustering. Sample K free continuations
                        from a feature prompt, mean-pool the model's hidden
                        states, compare cosine similarity to centroids built
                        from continuations of canonical instances of each move.
                        Asks: "what neighborhood does the continuation live in?"

  Probe J (judge)     — articulation. Same K continuations classified by a
                        rule-based judge that scans for the move's lexicon
                        and structural signatures.
                        Asks: "did the model actually make the move?"

The three probes can disagree. Disagreements are diagnostic: logprob without
drift means the model has the verdict but no surrounding vocabulary;
drift without judge means vocabulary clustering without articulated content;
judge without logprob means the move is producible but not preferred.

Usage:
  python three_probes.py --ckpt runs/mixed/ckpt.pt \
      --entities data/entities.json \
      --out runs/mixed/three_probes.json --device mps
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import WordTokenizer
from moves import MOVES, MOVES_BY_KEY, CENTROID_ANCHORS

K_SAMPLES = 16   # continuations per prompt
MAX_NEW = 12     # tokens per continuation
TEMPERATURE = 0.9


# ─── model loading ───────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model


# ─── Probe L: logprob (verdict preference) ───────────────────────────────────

@torch.no_grad()
def per_token_logprob(model, tokenizer, prompt, completion, device):
    """Sum of log-probabilities the model assigns to each token of `completion`
    given `prompt`. Higher is better. Returns None if any token is OOV
    (e.g. asking for 'dwarf' in a vocab that never saw the dwarf label)."""
    try:
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + completion)
    except KeyError:
        return None
    completion_ids = full_ids[len(prompt_ids):]
    if not completion_ids:
        return float("nan")
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(ids)
    log_probs = F.log_softmax(logits[0], dim=-1)
    total = 0.0
    for i, tok_id in enumerate(completion_ids):
        pos = len(prompt_ids) - 1 + i
        total += float(log_probs[pos, tok_id])
    return total


def probe_logprob(model, tokenizer, device):
    """For each move's force pairs, log-prob of positive vs controls.

    Skips force pairs whose positive completion is OOV (e.g. RECLASSIFY in
    unlabeled mode, where "dwarf" was never in the corpus).
    """
    results = {}
    for move in MOVES:
        rows = []
        for fp in move.force_pairs:
            pos_lp = per_token_logprob(model, tokenizer, fp.prompt, fp.positive, device)
            if pos_lp is None:
                rows.append({"prompt": fp.prompt, "positive": fp.positive.strip(),
                             "skipped": "positive OOV"})
                continue
            ctrl_lps = {c: per_token_logprob(model, tokenizer, fp.prompt, c, device)
                        for c in fp.controls}
            valid_ctrls = {k: v for k, v in ctrl_lps.items() if v is not None}
            best_ctrl = max(valid_ctrls, key=valid_ctrls.get) if valid_ctrls else None
            if best_ctrl is None:
                preferred = "positive"
            else:
                preferred = "positive" if pos_lp > valid_ctrls[best_ctrl] else f"control:{best_ctrl.strip()}"
            rows.append({
                "prompt": fp.prompt,
                "positive": fp.positive.strip(),
                "positive_logprob": pos_lp,
                "control_logprobs": {c.strip(): lp for c, lp in ctrl_lps.items()},
                "preferred": preferred,
            })
        results[move.key] = rows
    return results


# ─── Probe D: embedding drift (vocabulary clustering) ────────────────────────

@torch.no_grad()
def sample_continuations(model, tokenizer, prompt, k, device):
    """Generate k free continuations from prompt. Returns list of decoded strings."""
    out = []
    oov = [t for t in prompt.split() if t not in tokenizer.token_to_id]
    if oov:
        return out
    ids = torch.tensor([tokenizer.encode(prompt)] * k, dtype=torch.long, device=device)
    gen = model.generate(ids, max_new_tokens=MAX_NEW, temperature=TEMPERATURE)
    for row in gen.tolist():
        out.append(tokenizer.decode(row))
    return out


@torch.no_grad()
def mean_pool_state(model, tokenizer, text, device):
    """Mean of last-layer hidden states across the full text."""
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    h = model.hidden(ids)              # (1, T, n_embd)
    return h[0].mean(dim=0).detach().cpu().tolist()


def cos_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def feature_prompt_for(name, ents):
    # Use the entity's own feature description as the prompt prefix.
    feats = ents.get("phase1", {}).get(name, ents.get("phase2", {}).get(name))
    if not feats:
        return None
    m, d, o = feats["features"]
    mass = ["tiny", "small", "medium", "large", "huge"][m]
    diam = ["tiny", "small", "medium", "large", "huge"][d]
    orbit = ["near", "inner", "middle", "outer", "distant"][o]
    return f"{name} has mass {mass} diameter {diam} orbit {orbit} . {name} is a"


def probe_drift(model, tokenizer, ents, device):
    """Build move centroids from canonical anchors; project P10 onto them.

    Centroid construction:
      canon-centroid      = mean(continuations from P1..P9 feature-prompts)
      reclassify-centroid = mean(continuations from E1..E5 feature-prompts)

    Then sample K continuations of P10's feature-prompt and project each onto
    the (canon, reclassify) axis. Report the mean projection and the fraction
    of samples that land closer to each centroid.
    """
    centroids = {}
    raw_samples = {}
    for move_key, anchor_names in CENTROID_ANCHORS.items():
        states = []
        for name in anchor_names:
            prompt = feature_prompt_for(name, ents)
            if prompt is None:
                continue
            samples = sample_continuations(model, tokenizer, prompt, K_SAMPLES, device)
            for s in samples:
                states.append(mean_pool_state(model, tokenizer, s, device))
        if states:
            d = len(states[0])
            centroid = [sum(s[i] for s in states) / len(states) for i in range(d)]
            centroids[move_key] = centroid

    target_prompt = feature_prompt_for("P10", ents)
    target_samples = sample_continuations(model, tokenizer, target_prompt, K_SAMPLES, device)
    raw_samples["P10"] = target_samples

    sims_per_sample = []
    for s in target_samples:
        emb = mean_pool_state(model, tokenizer, s, device)
        sims_per_sample.append({k: cos_sim(emb, c) for k, c in centroids.items()})

    if sims_per_sample:
        nearest_counts = Counter(max(s, key=s.get) for s in sims_per_sample)
        mean_sim = {k: sum(s[k] for s in sims_per_sample) / len(sims_per_sample)
                    for k in centroids}
    else:
        nearest_counts, mean_sim = {}, {}

    return {
        "centroid_keys": list(centroids.keys()),
        "n_target_samples": len(target_samples),
        "mean_similarity": mean_sim,
        "nearest_centroid_counts": dict(nearest_counts),
        "samples": target_samples,
    }


# ─── Probe J: judge (rule-based articulation) ────────────────────────────────

ERIS_NAMES = {f"E{i}" for i in range(1, 10)}


def classify_continuation(text, target="P10"):
    """Classify a single continuation into one of:
      canon, reclassify, extend, hedge (both planet+dwarf), off_topic.
    """
    tokens = text.split()
    # Strip the prompt by finding the position right after "is a" — judge looks
    # at what comes after the verdict slot.
    if "is" in tokens and "a" in tokens:
        try:
            idx = next(i for i, t in enumerate(tokens) if t == "a"
                       and i > 0 and tokens[i - 1] == "is")
            tail = tokens[idx + 1:]
        except StopIteration:
            tail = tokens
    else:
        tail = tokens
    has_planet = any(t in ("planet", "planets") for t in tail)
    has_dwarf = any(t in ("dwarf", "dwarfs") for t in tail)
    has_eris = any(t in ERIS_NAMES for t in tail)
    if has_planet and has_dwarf:
        return "hedge"
    if has_planet:
        return "canon"
    if has_dwarf:
        return "reclassify"
    if has_eris:
        return "extend"
    return "off_topic"


def probe_judge(model, tokenizer, ents, device):
    """Generate K continuations from P10's feature-prompt and classify each."""
    target_prompt = feature_prompt_for("P10", ents)
    samples = sample_continuations(model, tokenizer, target_prompt, K_SAMPLES, device)
    labels = [classify_continuation(s) for s in samples]
    counts = dict(Counter(labels))
    return {
        "n_samples": len(samples),
        "label_counts": counts,
        "label_fractions": {k: v / len(samples) for k, v in counts.items()} if samples else {},
        "samples": list(zip(samples, labels)),
    }


# ─── driver ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.ckpt).parent
    tokenizer = WordTokenizer.load(out_dir / "tokenizer.json")
    model = load_model(args.ckpt, args.device)
    ents = json.loads(Path(args.entities).read_text())

    results = {
        "L_logprob": probe_logprob(model, tokenizer, args.device),
        "D_drift":   probe_drift(model, tokenizer, ents, args.device),
        "J_judge":   probe_judge(model, tokenizer, ents, args.device),
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
