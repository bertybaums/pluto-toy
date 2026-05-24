"""
Edge-sweep figure from ensemble_tension.py run with --p10-edges.

  (1) robust TENSION fraction vs prototype edge, one line per cell. The
      tension-vs-rote dissociation survives iff the overlap line stays high
      where the rote-control line stays low.
  (2,3) P10's P(planet) and P(dwarf), ensemble mean with a 2.5-97.5 band, vs
      edge, one panel per cell. This is the dual-label edge curve the May 6
      plan envisioned, now with bootstrap-and-retrain error bars rather than
      single runs.

Reads an ensemble_summary.json keyed <cell>__e<edge> (parametric ensemble).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENS = "parametric"
CELL_COLOR = {"overlap": "#1f77b4", "rote_control": "#d62728"}


def by_cell(summary):
    cells = defaultdict(list)
    for k, v in summary.items():
        cells[k.split("__e")[0]].append((v["p10_edge"], v))
    for c in cells:
        cells[c].sort(key=lambda t: t[0])
    return cells


def frac_panel(ax, cells):
    for cell, color in CELL_COLOR.items():
        if cell not in cells:
            continue
        rows = cells[cell]
        xs = [e for e, _ in rows]
        ys = [v["ensembles"][ENS]["tension_abs_frac"] for _, v in rows]
        ax.plot(xs, ys, "o-", color=color, label=cell)
    ax.set_xlabel("p10_edge (lower = more dwarf-like)")
    ax.set_ylabel("fraction of draws classified TENSION")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("(1) robust TENSION fraction vs edge")


def dual_panel(ax, rows, title):
    xs = np.array([e for e, _ in rows], dtype=float)
    for metric, color, lab in (("P_planet", "#1f77b4", "P(planet)"),
                               ("P_dwarf", "#ff7f0e", "P(dwarf)")):
        stat = [v["ensembles"][ENS][metric] for _, v in rows]
        m = np.array([s["mean"] for s in stat], dtype=float)
        lo = np.array([s["ci_lo"] if s["ci_lo"] is not None else s["mean"] for s in stat], dtype=float)
        hi = np.array([s["ci_hi"] if s["ci_hi"] is not None else s["mean"] for s in stat], dtype=float)
        ax.plot(xs, m, "o-", color=color, label=lab)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18)
    ax.axhline(0.10, color="grey", ls=":", alpha=0.5, label="abs floor 0.10")
    ax.set_xlabel("p10_edge")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    cells = by_cell(summary)
    out = Path(args.out) if args.out else Path(args.summary).parent / "fig_edge_sweep.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    frac_panel(axes[0], cells)
    if "overlap" in cells:
        dual_panel(axes[1], cells["overlap"], "(2) overlap: P10 dual-label vs edge")
    if "rote_control" in cells:
        dual_panel(axes[2], cells["rote_control"], "(3) rote-control: P10 dual-label vs edge")
    B = next(iter(summary.values())).get("B")
    fig.suptitle(f"Prototype-edge sweep under bootstrap-and-retrain "
                 f"(parametric MC, B={B})", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
