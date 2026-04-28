"""
Probes for category-strain detection.

Three probes run on the same checkpoint:
  A. Direct labeling — P(category | "<name> is a") for canon planets,
     P10 (the prototype-edge case), and E1..En (evidence-only entities).
     Strain signature: P(planet | "P10 is a") drops or splits after phase 2.
  B. Embedding geometry — cosine distance between name-token embeddings.
     Strain signature: dist(P10, E1) shrinks; dist(P10, canon planet) grows.
  C. Generation — free continuations from prompts that juxtapose conflict
     pairs ("P10 and E1 are ..."). Look for hedging or resolution moves.

Run after training. Compare the three probe outputs across schedules
(canon-only vs curriculum vs mixed) to see whether evidence exposure
shifts the model's representation of P10.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import WordTokenizer

CATS = ["planet", "asteroid", "comet", "moon", "dwarf"]


def load_model(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model


@torch.no_grad()
def next_token_probs(model, tokenizer, prompt, device):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _ = model(ids)
    probs = F.softmax(logits[0, -1], dim=-1)
    return {tokenizer.id_to_token[i]: float(probs[i]) for i in range(probs.numel())}


def token_embedding(model, tokenizer, token):
    return model.tok_emb.weight[tokenizer.token_to_id[token]].detach().cpu().tolist()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=12, device="cpu"):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new, temperature=0.8)
    return tokenizer.decode(out[0].tolist())


def cos_dist(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb + 1e-9)


def probe_direct_labeling(model, tokenizer, names, device):
    rows = {}
    for name in names:
        if name not in tokenizer.token_to_id:
            continue
        probs = next_token_probs(model, tokenizer, f"{name} is a", device)
        rows[name] = {c: probs.get(c, 0.0) for c in CATS}
    return rows


def probe_embedding_geometry(model, tokenizer, names):
    embs = {n: token_embedding(model, tokenizer, n)
            for n in names if n in tokenizer.token_to_id}
    keys = sorted(embs.keys())
    return {
        f"{a}|{b}": cos_dist(embs[a], embs[b])
        for i, a in enumerate(keys) for b in keys[i + 1:]
    }


def probe_generation(model, tokenizer, prompts, device):
    out = {}
    for p in prompts:
        oov = [t for t in p.split() if t not in tokenizer.token_to_id]
        if oov:
            out[p] = f"<skipped: out-of-vocab tokens {oov}>"
        else:
            out[p] = generate(model, tokenizer, p, device=device)
    return out


MASS_VOCAB = ["tiny", "small", "medium", "large", "huge"]
ORBIT_VOCAB = ["near", "inner", "middle", "outer", "distant"]


def feature_prompt(name, feats):
    """Repeat the name after the feature description so we can read out the
    contextualized representation of the entity *given* its features.
    """
    m, d, o = feats
    return (f"{name} has mass {MASS_VOCAB[m]} diameter {MASS_VOCAB[d]} "
            f"orbit {ORBIT_VOCAB[o]} . {name}")


@torch.no_grad()
def contextual_state(model, tokenizer, prompt, device):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    h = model.hidden(ids)
    return h[0, -1].detach().cpu().tolist()


def probe_contextual_geometry(model, tokenizer, ents, device):
    """Cosine distance between *contextualized* hidden states for each entity,
    given a feature-description prompt. This is where category strain would
    show up: a P10 prompted with its features should drift toward E1's
    region of representation space if the model has integrated the strain.
    """
    states = {}
    for name, info in ents["phase1"].items():
        prompt = feature_prompt(name, info["features"])
        if all(t in tokenizer.token_to_id for t in prompt.split()):
            states[name] = contextual_state(model, tokenizer, prompt, device)
    for name, info in ents.get("phase2", {}).items():
        prompt = feature_prompt(name, info["features"])
        if all(t in tokenizer.token_to_id for t in prompt.split()):
            states[name] = contextual_state(model, tokenizer, prompt, device)
    keys = sorted(states.keys())
    return {
        f"{a}|{b}": cos_dist(states[a], states[b])
        for i, a in enumerate(keys) for b in keys[i + 1:]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--entities", required=True, help="entities.json from generate_corpus.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.ckpt).parent
    tokenizer = WordTokenizer.load(out_dir / "tokenizer.json")
    model = load_model(args.ckpt, args.device)

    ents = json.loads(Path(args.entities).read_text())
    canon_planets = [n for n, v in ents["phase1"].items()
                     if v["category"] == "planet" and n != "P10"]
    canon_others = [n for n, v in ents["phase1"].items() if v["category"] != "planet"]
    eris_names = list(ents.get("phase2", {}).keys())

    targets = ["P10"] + canon_planets[:3] + canon_others[:3] + eris_names
    prompts = [
        "P10 is a",
        "E1 is a",
        "P10 and E1 are",
        "P10 and P1 are",
    ]

    results = {
        "A_direct_labeling": probe_direct_labeling(model, tokenizer, targets, args.device),
        "B_embedding_geometry": probe_embedding_geometry(model, tokenizer, targets),
        "C_generation": probe_generation(model, tokenizer, prompts, args.device),
        "D_contextual_geometry": probe_contextual_geometry(model, tokenizer, ents, args.device),
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
