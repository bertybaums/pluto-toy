"""
Figures from the tension-probe sweep:

  (1) cell-classification grid — one panel per rote-mix endpoint
      (overlap = 0/n, rote-control = n/n) at a fixed model size and edge.

  (2) prototype-edge curve — P10 dual-label probabilities vs p10_edge.

  (3) corner-mix curve — P10 dual-label probabilities vs n_eris_rote
      (the new May-7 axis). One panel per (schedule, model size).

Reads tension_summary.json from aggregate_tension.py (one or more dirs).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CELL_COLORS = {
    "tension_abs": "#d62728",
    "canon_abs":   "#2ca02c",
    "swap_abs":    "#ff7f0e",
    "nothing_abs": "#7f7f7f",
    None:          "#ffffff",
    "n/a":         "#ffffff",
}


def filter_summary(summary, **eq):
    """Return entries whose top-level fields match eq (e.g. n_embd=32)."""
    out = {}
    for k, v in summary.items():
        if all(v.get(field) == val for field, val in eq.items()):
            out[k] = v
    return out


def figure_grid(summary, *, edge=1.0, n_embd=32, n_layer=4,
                n_eris=5, out_path="grid.png"):
    """3x3 mode x schedule grid at the two corner-mix endpoints.

    Filters: only entries matching edge, n_embd, n_layer, n_eris.
    """
    modes = ["unlabeled", "dwarf", "planet"]
    scheds = ["canon-only", "curriculum", "mixed"]
    rote_endpoints = [(0, "overlap"), (n_eris, "rote-control")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sub = filter_summary(summary, p10_edge=edge, n_embd=n_embd,
                         n_layer=n_layer, n_eris=n_eris)
    for ax, (n_rote, panel_label) in zip(axes, rote_endpoints):
        ax.set_title(panel_label)
        ax.set_xticks(range(len(scheds))); ax.set_xticklabels(scheds, rotation=20)
        ax.set_yticks(range(len(modes)));  ax.set_yticklabels(modes)
        ax.set_xlabel("schedule"); ax.set_ylabel("mode")
        for j, mode in enumerate(modes):
            for i, sched in enumerate(scheds):
                cell_data = next(
                    (v for v in sub.values()
                     if v["mode"] == mode and v["schedule"] == sched
                     and v["n_eris_rote"] == n_rote),
                    None)
                if cell_data is None:
                    continue
                cell = cell_data.get("abs_cell_mode")
                p_p = cell_data["P_planet"]["mean"]
                p_d = cell_data["P_dwarf"]["mean"]
                ax.add_patch(plt.Rectangle((i - 0.45, j - 0.45), 0.9, 0.9,
                                           facecolor=CELL_COLORS.get(cell, "#ffffff"),
                                           edgecolor="black"))
                p_p_s = "--" if p_p is None else f"{p_p:.2f}"
                p_d_s = "--" if p_d is None else f"{p_d:.2f}"
                ax.text(i, j + 0.07, (cell or "n/a"), ha="center", va="center",
                        fontsize=10, fontweight="bold")
                ax.text(i, j - 0.18, f"P={p_p_s}\nD={p_d_s}",
                        ha="center", va="center", fontsize=8)
        ax.set_xlim(-0.5, len(scheds) - 0.5)
        ax.set_ylim(-0.5, len(modes) - 0.5)
        ax.set_aspect("equal")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black",
                                    label=lab.replace("_abs", ""))
                      for lab, c in CELL_COLORS.items() if lab in
                      ("tension_abs", "canon_abs", "swap_abs", "nothing_abs")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"P10 cell classification  (n_embd={n_embd}, n_layer={n_layer}, "
                 f"p10_edge={edge:.1f}, abs floor 0.10)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def figure_edge_curve(summary, *, mode="dwarf",
                      schedules=("mixed", "curriculum"),
                      n_embd=32, n_layer=4, out_path="edge_curve.png"):
    fig, axes = plt.subplots(1, len(schedules),
                             figsize=(5 * len(schedules), 4.2),
                             sharey=True)
    if len(schedules) == 1:
        axes = [axes]
    sub = filter_summary(summary, mode=mode, n_embd=n_embd, n_layer=n_layer)
    for ax, sched in zip(axes, schedules):
        for n_rote, marker, color, label in [
                (0, "o", "#1f77b4", "overlap"),
                (None, "s", "#ff7f0e", "rote-control")]:
            xs, p_p, p_d = [], [], []
            for v in sub.values():
                if v["schedule"] != sched:
                    continue
                if n_rote == 0 and v["n_eris_rote"] != 0:
                    continue
                if n_rote is None and v["n_eris_rote"] != v["n_eris"]:
                    continue
                xs.append(v["p10_edge"])
                p_p.append(v["P_planet"]["mean"])
                p_d.append(v["P_dwarf"]["mean"])
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ax.plot(xs, np.array(p_p)[order], marker + "-",
                    color=color, label=f"{label}: P(planet)", alpha=0.9)
            ax.plot(xs, np.array(p_d)[order], marker + "--",
                    color=color, label=f"{label}: P(dwarf)", alpha=0.6)
        ax.set_title(f"schedule = {sched}")
        ax.set_xlabel("p10_edge")
        ax.set_ylabel("probability")
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"P10 dual-label probabilities vs prototype edge "
                 f"({mode} mode, n_embd={n_embd}, n_layer={n_layer})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def figure_corner_mix(summary, *, mode="dwarf",
                      schedules=("mixed", "curriculum"),
                      model_sizes=((32, 4), (64, 6)),
                      out_path="corner_mix.png"):
    """Headline figure for May-7 follow-up.

    For each (schedule, model size), plot P_planet, P_dwarf, and
    tension_index on P10 as a function of n_eris_rote (the corner-mix
    knob). Each row is a model size; each column is a schedule.
    """
    n_rows = len(model_sizes)
    n_cols = len(schedules)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.6 * n_rows),
                             sharey=True, squeeze=False)
    for r, (n_embd, n_layer) in enumerate(model_sizes):
        sub = filter_summary(summary, mode=mode, n_embd=n_embd, n_layer=n_layer)
        for c, sched in enumerate(schedules):
            ax = axes[r][c]
            xs, p_p, p_p_se, p_d, p_d_se, t_idx = [], [], [], [], [], []
            for v in sub.values():
                if v["schedule"] != sched:
                    continue
                xs.append(v["n_eris_rote"])
                pp = v["P_planet"]
                pd = v["P_dwarf"]
                ti = v["tension_index"]
                p_p.append(pp["mean"]); p_p_se.append(pp["std"] / max(pp["n"], 1) ** 0.5)
                p_d.append(pd["mean"]); p_d_se.append(pd["std"] / max(pd["n"], 1) ** 0.5)
                t_idx.append(ti["mean"] if ti["mean"] is not None else 0.0)
            if not xs:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ax.errorbar(xs, np.array(p_p)[order], yerr=np.array(p_p_se)[order],
                        fmt="o-", color="#1f77b4", label="P(planet)", capsize=3)
            ax.errorbar(xs, np.array(p_d)[order], yerr=np.array(p_d_se)[order],
                        fmt="s-", color="#ff7f0e", label="P(dwarf)", capsize=3)
            ax.plot(xs, np.array(t_idx)[order], "x:", color="#2ca02c",
                    label="tension index", alpha=0.6)
            ax.axhline(0.10, color="grey", linestyle=":", alpha=0.4,
                       label="abs floor 0.10")
            ax.set_title(f"{sched}  ·  n_embd={n_embd}, n_layer={n_layer}")
            ax.set_xlabel("n_eris_rote (E* in disjoint corner)")
            if c == 0:
                ax.set_ylabel("probability / index")
            ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"P10 dual-label probabilities vs corner-mix  ({mode} mode)",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="tension_summary.json from aggregate_tension.py")
    ap.add_argument("--figure", choices=["grid", "edge", "corner_mix", "all"],
                    default="all")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mode", default="dwarf")
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-eris", type=int, default=5)
    ap.add_argument("--edge", type=float, default=1.0)
    ap.add_argument("--model-sizes", nargs="+", default=["32,4", "64,6"],
                    help="Comma-separated n_embd,n_layer pairs for the "
                         "corner-mix figure.")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    out_dir = Path(args.out_dir or Path(args.summary).parent)

    if args.figure in ("grid", "all"):
        figure_grid(summary, edge=args.edge, n_embd=args.n_embd,
                    n_layer=args.n_layer, n_eris=args.n_eris,
                    out_path=out_dir / "fig_grid.png")
    if args.figure in ("edge", "all"):
        figure_edge_curve(summary, mode=args.mode, n_embd=args.n_embd,
                          n_layer=args.n_layer,
                          out_path=out_dir / "fig_edge_curve.png")
    if args.figure in ("corner_mix", "all"):
        sizes = [tuple(int(x) for x in s.split(",")) for s in args.model_sizes]
        figure_corner_mix(summary, mode=args.mode, model_sizes=sizes,
                          out_path=out_dir / "fig_corner_mix.png")


if __name__ == "__main__":
    main()
