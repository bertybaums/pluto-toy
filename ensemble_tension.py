"""
Bootstrap-and-retrain uncertainty for the dual-label tension probe.

Ported from the /unstructured project's epistemic-uncertainty move: a single
trained model gives only a point estimate of P10's tension (its softmax spread
is aleatoric, not epistemic). To call a category-strain reading *robust*
rather than a one-run artifact, retrain an ensemble and measure the spread.

pluto-toy's existing sweep already varies the init seed with the corpus held
fixed (sweep.py generates the corpus once with --seed 0; train.py varies
torch.manual_seed). That captures OPTIMIZATION noise only. The missing axis is
the corpus draw. This script adds it and decomposes the total:

    Var_total = Var_epistemic (vary corpus, fix init)
              + Var_optimization (fix corpus, vary init)

Three ensembles per cell:
  parametric   - regenerate the corpus from a fresh generate_corpus.py --seed b
                 (redraws features + ordering); probe each member against its
                 OWN world. A Monte-Carlo over the generative process.
  bootstrap    - resample the realized seed-0 corpus with replacement at the
                 statement-pair level (so 'X has ...' stays glued to its
                 'X is a ...'); fixed init; probe against the FIXED seed-0
                 entities, mirroring /unstructured's fixed reference set.
  optimization - fixed seed-0 corpus, vary the train init seed. The component
                 the old sweep already had.

parametric and bootstrap estimate the same epistemic quantity by different
means; their agreement is the cross-check (cf. /unstructured validating its
bootstrap SD against the analytic SE).

Usage:
  python ensemble_tension.py --device mps --B 20 \
      --cells overlap rote_control
"""
import argparse
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from aggregate_tension import absolute_cell

# Headline cells, all dwarf mode at edge=1.0, d32l4 (the April-28 settings).
CELLS = {
    "overlap":      {"mode": "dwarf", "schedule": "curriculum", "n_eris_rote": 0,
                     "label": "overlap (curriculum, r0/6)"},
    "rote_control": {"mode": "dwarf", "schedule": "curriculum", "n_eris_rote": 6,
                     "label": "rote-control (curriculum, r6/6)"},
    "canon":        {"mode": "dwarf", "schedule": "mixed", "n_eris_rote": 0,
                     "label": "canon baseline (mixed, r0/6)"},
}
METRICS = ("tension_index", "P_planet", "P_dwarf")


def run(cmd, log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)} (see {log_path})")


# --- statement-pair bootstrap -------------------------------------------------

def _kind(line):
    t = line.split()
    if len(t) >= 2 and t[1] == "has":
        return "describe", t[0]
    if len(t) >= 3 and t[1] == "is" and t[2] == "a":
        return "label", t[0]
    if t[:1] == ["the"] and "are" in t:
        return "roster", None
    return "other", None


def parse_records(text):
    """Split corpus text into resampling units. A '<name> has ...' line glued
    to the immediately-following '<name> is a ...' label is one record; roster
    lines and unlabeled describes are their own records. Keeping the pair
    intact preserves the feature->label adjacency the model learns from."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    recs, i = [], 0
    while i < len(lines):
        kind, name = _kind(lines[i])
        if kind == "describe" and i + 1 < len(lines):
            k2, n2 = _kind(lines[i + 1])
            if k2 == "label" and n2 == name:
                recs.append([lines[i], lines[i + 1]])
                i += 2
                continue
        recs.append([lines[i]])
        i += 1
    return recs


def bootstrap_corpus(src, dst, rng):
    """Resample each phase with replacement at the record level, then rewrite."""
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("phase1.txt", "phase2.txt"):
        recs = parse_records((src / fname).read_text())
        chosen = [recs[rng.randrange(len(recs))] for _ in range(len(recs))] if recs else []
        rng.shuffle(chosen)
        lines = [ln for rec in chosen for ln in rec]
        (dst / fname).write_text("\n".join(lines) + "\n")
    shutil.copy(src / "entities.json", dst / "entities.json")


# --- train + probe one member -------------------------------------------------

def gen_corpus(data_dir, spec, seed, edge, args):
    run([sys.executable, "generate_corpus.py", "--out-dir", str(data_dir),
         "--mode", spec["mode"], "--seed", str(seed),
         "--p10-edge", str(edge), "--n-eris", str(args.n_eris),
         "--n-eris-rote", str(spec["n_eris_rote"])],
        log_path=data_dir / "_gen.log")


def train_and_probe(data_dir, run_dir, *, schedule, init_seed, entities, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "train.py", "--data-dir", str(data_dir),
         "--out-dir", str(run_dir), "--schedule", schedule,
         "--device", args.device, "--seed", str(init_seed),
         "--phase1-steps", str(args.phase1_steps),
         "--phase2-steps", str(args.phase2_steps),
         "--n-embd", str(args.n_embd), "--n-layer", str(args.n_layer),
         "--n-head", str(args.n_head)],
        log_path=run_dir / "_train.log")
    run([sys.executable, "tension_probe.py", "--ckpt", str(run_dir / "ckpt.pt"),
         "--entities", str(entities), "--out", str(run_dir / "tension.json"),
         "--device", args.device],
        log_path=run_dir / "_tension.log")
    p10 = json.loads((run_dir / "tension.json").read_text())["targets"].get("P10", {})
    pp, pd = p10.get("P_planet"), p10.get("P_dwarf")
    return {"P_planet": pp, "P_dwarf": pd, "tension_index": p10.get("tension_index"),
            "cell": p10.get("cell"), "abs_cell": absolute_cell(pp, pd)}


# --- aggregation --------------------------------------------------------------

def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "sd": None, "ci_lo": None, "ci_hi": None, "n": 0}
    return {"mean": statistics.fmean(vals),
            "sd": statistics.pstdev(vals) if n >= 2 else 0.0,
            "ci_lo": percentile(vals, 2.5), "ci_hi": percentile(vals, 97.5), "n": n}


def summarize_ensemble(draws):
    out = {"n_draws": len(draws), "draws": draws}
    for m in METRICS:
        out[m] = summarize([d[m] for d in draws])
    for key, frac, hit in (("cell", "tension_frac", "tension"),
                           ("abs_cell", "tension_abs_frac", "tension_abs")):
        c = Counter(d[key] for d in draws if d.get(key) is not None)
        out[key + "_votes"] = dict(c)
        out[frac] = c.get(hit, 0) / (sum(c.values()) or 1)
    return out


def decompose(ens):
    def var(name, m):
        s = ens.get(name, {}).get(m, {}).get("sd")
        return s ** 2 if s is not None else None
    out = {}
    for m in METRICS:
        vb, vo, vp = var("bootstrap", m), var("optimization", m), var("parametric", m)
        vt = vb + vo if (vb is not None and vo is not None) else None
        out[m] = {"var_epistemic_boot": vb, "var_optimization": vo,
                  "var_total": vt, "sd_total": math.sqrt(vt) if vt is not None else None,
                  "var_epistemic_parametric": vp}
    return out


# --- driver -------------------------------------------------------------------

def run_cell(name, spec, args, out_root, edge):
    cdir = out_root / name
    base_data = cdir / "base_data"
    gen_corpus(base_data, spec, args.base_seed, edge, args)
    base_ents = base_data / "entities.json"
    baseline = train_and_probe(base_data, cdir / "baseline", schedule=spec["schedule"],
                               init_seed=args.init_seed, entities=base_ents, args=args)

    ens = {}
    for ename in args.ensembles:
        draws = []
        for b in range(args.B):
            t0 = time.time()
            if ename == "parametric":
                d = cdir / f"param_b{b}_data"
                gen_corpus(d, spec, args.param_seed_offset + b, edge, args)
                r = train_and_probe(d, cdir / f"param_b{b}", schedule=spec["schedule"],
                                    init_seed=args.init_seed, entities=d / "entities.json", args=args)
            elif ename == "bootstrap":
                d = cdir / f"boot_b{b}_data"
                bootstrap_corpus(base_data, d, random.Random(args.boot_seed_offset + b))
                r = train_and_probe(d, cdir / f"boot_b{b}", schedule=spec["schedule"],
                                    init_seed=args.init_seed, entities=base_ents, args=args)
            else:  # optimization: fixed corpus, vary init
                r = train_and_probe(base_data, cdir / f"opt_b{b}", schedule=spec["schedule"],
                                    init_seed=b, entities=base_ents, args=args)
            draws.append(r)
            print(f"  [{name}/{ename} {b + 1}/{args.B}] "
                  f"Pp={r['P_planet']} Pd={r['P_dwarf']} T={r['tension_index']} "
                  f"{r['abs_cell']} ({time.time() - t0:.1f}s)", flush=True)
        ens[ename] = summarize_ensemble(draws)

    result = {"spec": spec, "label": spec["label"], "p10_edge": edge, "B": args.B,
              "init_seed": args.init_seed, "base_seed": args.base_seed,
              "baseline": baseline, "ensembles": ens, "decomposition": decompose(ens)}
    (cdir / "ensemble.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["overlap", "rote_control"],
                    choices=list(CELLS))
    ap.add_argument("--ensembles", nargs="+",
                    default=["parametric", "bootstrap", "optimization"],
                    choices=["parametric", "bootstrap", "optimization"])
    ap.add_argument("--B", type=int, default=20, help="ensemble size per axis")
    ap.add_argument("--init-seed", type=int, default=0,
                    help="fixed init seed for the epistemic ensembles")
    ap.add_argument("--base-seed", type=int, default=0,
                    help="corpus seed for the realized (bootstrap/optimization) corpus")
    ap.add_argument("--param-seed-offset", type=int, default=1000)
    ap.add_argument("--boot-seed-offset", type=int, default=20000)
    ap.add_argument("--n-eris", type=int, default=6)
    ap.add_argument("--p10-edge", type=float, default=1.0)
    ap.add_argument("--p10-edges", type=float, nargs="+", default=None,
                    help="Sweep these prototype edges (overrides --p10-edge). "
                         "Result keys become <cell>__e<edge>.")
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--phase1-steps", type=int, default=1500)
    ap.add_argument("--phase2-steps", type=int, default=300)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-root", default="ensemble")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    edges = args.p10_edges or [args.p10_edge]
    started = time.time()
    summary = {}
    for edge in edges:
        for cell_name in args.cells:
            name = cell_name if len(edges) == 1 else f"{cell_name}__e{edge:.2f}"
            summary[name] = run_cell(name, CELLS[cell_name], args, out_root, edge)
            (out_root / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone - {len(edges)} edges x {len(args.cells)} cells x "
          f"{len(args.ensembles)} ensembles x B={args.B}, "
          f"{(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
