"""
Figures from ensemble_tension.py's ensemble_summary.json.

Three panels tell the robust-vs-artifact story:

  (1) forest plot - P10's P(planet) and P(dwarf), mean + 2.5-97.5 percentile
      interval, for each (cell, ensemble). Wide bars = not robust.
  (2) variance decomposition - for the tension index, the optimization
      component (vary init, fixed corpus; what the old sweep measured)
      stacked under the epistemic component (vary corpus, fixed init; the
      added axis), with the parametric-MC estimate beside it as a cross-check.
  (3) vote composition - the abs-cell classification fractions per
      (cell, ensemble). A robust TENSION reading would be one solid colour.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENS = ["parametric", "bootstrap", "optimization"]
ENS_LABEL = {"parametric": "parametric MC", "bootstrap": "bootstrap",
             "optimization": "optimization\n(init only)"}
ABS_COLORS = {"tension_abs": "#d62728", "canon_abs": "#2ca02c",
              "swap_abs": "#ff7f0e", "nothing_abs": "#7f7f7f", "n/a": "#dddddd"}


def xerr(stat):
    """(mean, [lo_err, hi_err]) for an errorbar, or None if unmeasured."""
    m = stat.get("mean")
    if m is None:
        return None
    lo = stat.get("ci_lo", m) or m
    hi = stat.get("ci_hi", m) or m
    return m, [[max(0.0, m - lo)], [max(0.0, hi - m)]]


def forest(ax, summary, cells):
    rows, labels = [], []
    for ci, cell in enumerate(cells):
        ens = summary[cell]["ensembles"]
        for e in ENS:
            if e in ens:
                rows.append((cell, e, ens[e]))
    y = list(range(len(rows)))[::-1]
    for yi, (cell, e, v) in zip(y, rows):
        for metric, color, off, mk, lab in (
                ("P_planet", "#1f77b4", 0.16, "o", "P(planet)"),
                ("P_dwarf", "#ff7f0e", -0.16, "s", "P(dwarf)")):
            r = xerr(v[metric])
            if r is None:
                continue
            m, e2 = r
            ax.errorbar(m, yi + off, xerr=e2, fmt=mk, color=color, capsize=3,
                        ms=6, label=lab if (yi == y[0]) else None)
        labels.append(f"{cell}\n{ENS_LABEL[e].splitlines()[0]}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{cell} / {e[:4]}" for cell, e, _ in rows], fontsize=8)
    ax.axvline(0.10, color="grey", ls=":", alpha=0.6, label="abs floor 0.10")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("probability for P10")
    ax.set_title("(1) P10 dual-label probabilities\nmean + 2.5-97.5% interval")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)


def decomp_bars(ax, summary, cells, metric="tension_index"):
    x = np.arange(len(cells))
    w = 0.36
    v_opt, v_boot, v_par = [], [], []
    for cell in cells:
        d = summary[cell]["decomposition"][metric]
        v_opt.append(d.get("var_optimization") or 0.0)
        v_boot.append(d.get("var_epistemic_boot") or 0.0)
        v_par.append(d.get("var_epistemic_parametric") or 0.0)
    v_opt, v_boot, v_par = map(np.array, (v_opt, v_boot, v_par))
    ax.bar(x - w / 2, v_opt, w, color="#7f7f7f", label="optimization (init)")
    ax.bar(x - w / 2, v_boot, w, bottom=v_opt, color="#d62728",
           label="epistemic (corpus, boot)")
    ax.bar(x + w / 2, v_par, w, color="#9467bd", label="epistemic (parametric)")
    for xi, vo, vb in zip(x, v_opt, v_boot):
        if vo > 0 and vo + vb > 0.005:
            ax.annotate(f"boot/opt\n={vb / vo:.1f}x", (xi - w / 2, vo + vb),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=8)
    ax.set_ylabel(f"variance of {metric}")
    ax.set_title("(2) variance decomposition\n(old sweep saw only the grey)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)


def vote_bars(ax, summary, cells):
    cols, fracs = [], []
    for cell in cells:
        ens = summary[cell]["ensembles"]
        for e in ENS:
            if e not in ens:
                continue
            votes = ens[e]["abs_cell_votes"]
            tot = sum(votes.values()) or 1
            cols.append(f"{cell}\n{e[:4]}")
            fracs.append({k: votes.get(k, 0) / tot for k in ABS_COLORS})
    x = np.arange(len(cols))
    bottom = np.zeros(len(cols))
    for cls, color in ABS_COLORS.items():
        h = np.array([f[cls] for f in fracs])
        if h.sum() == 0:
            continue
        ax.bar(x, h, bottom=bottom, color=color, label=cls.replace("_abs", ""))
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of ensemble")
    ax.set_title("(3) P10 classification vote\n(robust = one solid colour)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="ensemble_summary.json from ensemble_tension.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="tension_index",
                    help="metric for the variance-decomposition panel")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    cells = list(summary)
    out = Path(args.out) if args.out else Path(args.summary).parent / "fig_ensemble.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6),
                             gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    forest(axes[0], summary, cells)
    decomp_bars(axes[1], summary, cells, metric=args.metric)
    vote_bars(axes[2], summary, cells)
    B = summary[cells[0]].get("B")
    fig.suptitle(f"Bootstrap-and-retrain uncertainty on P10's tension "
                 f"(B={B} per ensemble)", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
