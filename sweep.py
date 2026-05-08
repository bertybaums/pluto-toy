"""
Run the sweep across mode × schedule × seed × rote-mix × p10-edge × model size.

Default grid (April-28-compatible):
  modes        = unlabeled, dwarf, planet
  schedules    = canon-only, curriculum, mixed
  seeds        = 0..4
  rote-controls= [0]            (overlap only; pass --rote-controls 0 1 to add)
  p10-edges    = [1.0]
  n-eris-rotes = unset          (uses --rote-controls; set to override the binary)
  model        = n_embd=32, n_layer=4 (~50K params; the May-6 default)

Pass `--n-eris-rotes 0 2 4 6 --n-eris 6` to do a corner-mix sweep
(May-7 follow-up step #1). Pass `--n-embd 64 --n-layer 6` to do the
~200K-param size (May-7 follow-up step #2).

Each cell:
  1. regenerate the corpus (one per mode × rote-mix × edge × n_eris)
  2. train (per cell, with the chosen model size)
  3. run probe.py (legacy three-probe set)
  4. run tension_probe.py (dual-label tension probe)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODES = ["unlabeled", "dwarf", "planet"]
SCHEDULES = ["canon-only", "curriculum", "mixed"]


def run(cmd, log_path=None):
    print(f"  $ {' '.join(cmd)}", flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)} (see {log_path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--schedules", nargs="+", default=SCHEDULES)
    ap.add_argument("--rote-controls", type=int, nargs="+", default=[0],
                    help="0 = overlap, 1 = full rote-control. Use this OR "
                         "--n-eris-rotes (mutually exclusive in spirit).")
    ap.add_argument("--n-eris-rotes", type=int, nargs="+", default=None,
                    help="Number of E* (out of n_eris) placed in the disjoint "
                         "corner. Overrides --rote-controls when set. "
                         "Example: --n-eris-rotes 0 2 4 6 --n-eris 6.")
    ap.add_argument("--n-eris", type=int, default=5)
    ap.add_argument("--p10-edges", type=float, nargs="+", default=[1.0])
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--phase1-steps", type=int, default=1500)
    ap.add_argument("--phase2-steps", type=int, default=300)
    ap.add_argument("--out-root", default="sweep")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve the corner-mix axis. --n-eris-rotes wins when set; otherwise
    # use --rote-controls (binary) and convert to {0, n_eris}.
    if args.n_eris_rotes is not None:
        rote_axis = list(args.n_eris_rotes)
    else:
        rote_axis = [args.n_eris if rc else 0 for rc in args.rote_controls]

    manifest = []
    started = time.time()
    n_total = (len(args.modes) * len(args.seeds) * len(args.schedules)
               * len(rote_axis) * len(args.p10_edges))
    n_done = 0

    for mode in args.modes:
        for n_rote in rote_axis:
            for edge in args.p10_edges:
                corpus_tag = f"{mode}_r{n_rote}of{args.n_eris}_e{edge:.2f}"
                data_dir = out_root / f"data_{corpus_tag}"
                run([sys.executable, "generate_corpus.py",
                     "--out-dir", str(data_dir),
                     "--mode", mode,
                     "--seed", "0",
                     "--p10-edge", str(edge),
                     "--n-eris", str(args.n_eris),
                     "--n-eris-rote", str(n_rote)],
                    log_path=data_dir / "_gen.log")

                for seed in args.seeds:
                    for sched in args.schedules:
                        run_name = f"{corpus_tag}__{sched}__s{seed}"
                        run_dir = out_root / run_name
                        run_dir.mkdir(parents=True, exist_ok=True)

                        t0 = time.time()
                        run([sys.executable, "train.py",
                             "--data-dir", str(data_dir),
                             "--out-dir", str(run_dir),
                             "--schedule", sched,
                             "--device", args.device,
                             "--seed", str(seed),
                             "--phase1-steps", str(args.phase1_steps),
                             "--phase2-steps", str(args.phase2_steps),
                             "--n-embd", str(args.n_embd),
                             "--n-layer", str(args.n_layer),
                             "--n-head", str(args.n_head)],
                            log_path=run_dir / "_train.log")
                        run([sys.executable, "probe.py",
                             "--ckpt", str(run_dir / "ckpt.pt"),
                             "--entities", str(data_dir / "entities.json"),
                             "--out", str(run_dir / "probe.json"),
                             "--device", args.device],
                            log_path=run_dir / "_probe.log")
                        run([sys.executable, "tension_probe.py",
                             "--ckpt", str(run_dir / "ckpt.pt"),
                             "--entities", str(data_dir / "entities.json"),
                             "--out", str(run_dir / "tension.json"),
                             "--device", args.device],
                            log_path=run_dir / "_tension.log")

                        manifest.append({
                            "mode": mode, "schedule": sched, "seed": seed,
                            "n_eris": args.n_eris, "n_eris_rote": n_rote,
                            "rote_fraction": n_rote / args.n_eris,
                            "rote_control": (n_rote == args.n_eris),
                            "p10_edge": edge,
                            "n_embd": args.n_embd, "n_layer": args.n_layer,
                            "run_dir": str(run_dir),
                            "probe": str(run_dir / "probe.json"),
                            "tension": str(run_dir / "tension.json"),
                            "wall_seconds": time.time() - t0,
                        })
                        n_done += 1
                        elapsed = time.time() - started
                        eta = elapsed / n_done * (n_total - n_done)
                        print(f"  [{n_done}/{n_total}]  {run_name}  "
                              f"({manifest[-1]['wall_seconds']:.1f}s)  "
                              f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                              flush=True)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone — {n_done} runs, {(time.time() - started)/60:.1f} min total")


if __name__ == "__main__":
    main()
