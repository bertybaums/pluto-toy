"""
Run the mode × schedule × seed sweep and aggregate results.

Grid:
  modes      = unlabeled, dwarf, planet
  schedules  = canon-only, curriculum, mixed
  seeds      = 0..4 (model init only; world layout is fixed at corpus-seed=0)

Default total: 3 modes × 3 schedules × 5 seeds = 45 runs.
Each run: regenerate corpus (once per mode), train, probe.
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
    ap.add_argument("--device", default="mps")
    ap.add_argument("--phase1-steps", type=int, default=1500)
    ap.add_argument("--phase2-steps", type=int, default=300)
    ap.add_argument("--out-root", default="sweep")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    started = time.time()
    n_total = len(args.modes) * len(args.seeds) * len(args.schedules)
    n_done = 0

    for mode in args.modes:
        # One corpus per mode, fixed world seed = 0
        data_dir = out_root / f"data_{mode}"
        run([sys.executable, "generate_corpus.py",
             "--out-dir", str(data_dir), "--mode", mode, "--seed", "0"],
            log_path=data_dir / "_gen.log")

        for seed in args.seeds:
            for sched in args.schedules:
                run_name = f"{mode}__{sched}__s{seed}"
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
                     "--phase2-steps", str(args.phase2_steps)],
                    log_path=run_dir / "_train.log")
                run([sys.executable, "probe.py",
                     "--ckpt", str(run_dir / "ckpt.pt"),
                     "--entities", str(data_dir / "entities.json"),
                     "--out", str(run_dir / "probe.json"),
                     "--device", args.device],
                    log_path=run_dir / "_probe.log")

                manifest.append({
                    "mode": mode, "schedule": sched, "seed": seed,
                    "run_dir": str(run_dir),
                    "probe": str(run_dir / "probe.json"),
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
