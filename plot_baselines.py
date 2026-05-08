"""
Baseline-vs-transformer comparison figure.

For each corner-mix value:
  - run all four baselines on the corpus → P(planet), P(dwarf) for P10
  - look up the corresponding transformer mean from the tension summary
  - plot as one line per classifier on each panel

Two panels: P(planet) on the left, P(dwarf) on the right.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baseline_classifier import evaluate_baselines

BASELINE_NAMES = ["logreg", "gnb", "knn3", "bayes"]
BASELINE_LABELS = {
    "logreg": "logistic regression",
    "gnb":    "Gaussian naive Bayes",
    "knn3":   "k-NN (k=3)",
    "bayes":  "Bayes posterior (by hand)",
}
BASELINE_COLORS = {
    "logreg": "#1f77b4",
    "gnb":    "#2ca02c",
    "knn3":   "#9467bd",
    "bayes":  "#8c564b",
}
TRANSFORMER_COLORS = {
    "small_curriculum": "#d62728",
    "large_curriculum": "#ff7f0e",
}


def find_corpus_dirs(sweep_dir, n_eris):
    """Return list of (n_rote, entities_path) for all n_rote present."""
    dirs = []
    for path in sorted(Path(sweep_dir).glob(f"data_dwarf_r*of{n_eris}_*")):
        # data_dwarf_r{N}of{n_eris}_e1.00
        n_rote = int(path.name.split("_r")[1].split("of")[0])
        ents = path / "entities.json"
        if ents.exists():
            dirs.append((n_rote, str(ents)))
    return dirs


def transformer_curve(summary, *, n_embd, n_layer, schedule="curriculum"):
    """Pull P(planet), P(dwarf) means as a function of n_eris_rote."""
    xs, p_p, p_p_se, p_d, p_d_se = [], [], [], [], []
    for v in summary.values():
        if v.get("schedule") != schedule: continue
        if v.get("n_embd") != n_embd or v.get("n_layer") != n_layer: continue
        xs.append(v["n_eris_rote"])
        pp = v["P_planet"]; pd = v["P_dwarf"]
        p_p.append(pp["mean"]); p_p_se.append(pp["std"] / max(pp["n"], 1) ** 0.5)
        p_d.append(pd["mean"]); p_d_se.append(pd["std"] / max(pd["n"], 1) ** 0.5)
    order = np.argsort(xs)
    return (np.array(xs)[order],
            np.array(p_p)[order], np.array(p_p_se)[order],
            np.array(p_d)[order], np.array(p_d_se)[order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="tension_summary.json (combined small+large from aggregate_tension)")
    ap.add_argument("--sweep-dir", default="sweep_small",
                    help="Sweep dir whose data_* corpora to read for the baselines.")
    ap.add_argument("--n-eris", type=int, default=6)
    ap.add_argument("--out", default="results/fig_baseline_vs_transformer_2026-05-07.png")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    corpora = find_corpus_dirs(args.sweep_dir, args.n_eris)
    if not corpora:
        raise SystemExit(f"no corpora found under {args.sweep_dir}")

    # Run all baselines for each corner-mix.
    baseline_results = {}
    for n_rote, ents_path in corpora:
        baseline_results[n_rote] = evaluate_baselines(ents_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, label in zip(axes, ("planet", "dwarf")):
        # baselines
        for name in BASELINE_NAMES:
            xs, ys = [], []
            for n_rote in sorted(baseline_results):
                v = baseline_results[n_rote][name].get(label)
                if v is None:
                    continue
                xs.append(n_rote); ys.append(v)
            ax.plot(xs, ys, "o-", color=BASELINE_COLORS[name],
                    label=BASELINE_LABELS[name], alpha=0.85, linewidth=1.5)
        # transformer curves
        for size_label, (n_embd, n_layer) in [("small (50K)", (32, 4)),
                                              ("large (200K)", (64, 6))]:
            xs, p_p, p_p_se, p_d, p_d_se = transformer_curve(
                summary, n_embd=n_embd, n_layer=n_layer)
            y = p_p if label == "planet" else p_d
            yerr = p_p_se if label == "planet" else p_d_se
            color = TRANSFORMER_COLORS[
                "small_curriculum" if n_embd == 32 else "large_curriculum"]
            ax.errorbar(xs, y, yerr=yerr, fmt="s--", color=color, capsize=3,
                        label=f"transformer {size_label} curriculum",
                        alpha=0.9, linewidth=2)

        ax.set_title(f"P({label} | P10's features)")
        ax.set_xlabel("n_eris_rote (E* in disjoint corner; out of 6)")
        ax.set_ylabel("probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(sorted(baseline_results.keys()))
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)
    fig.suptitle("Baseline classifiers vs transformer "
                 "(curriculum schedule, dwarf mode, p10_edge=1.0)", y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Also dump the raw baseline values as a table.
    print("\n=== baseline P(planet | P10) by corner-mix ===")
    header = "n_rote  " + "  ".join(f"{n:>14}" for n in BASELINE_NAMES)
    print(header)
    for n_rote in sorted(baseline_results):
        row = f"{n_rote}/6   "
        for name in BASELINE_NAMES:
            v = baseline_results[n_rote][name].get("planet")
            row += f"  {v:14.3f}" if v is not None else f"  {'--':>14}"
        print(row)
    print("\n=== baseline P(dwarf | P10) by corner-mix ===")
    print(header)
    for n_rote in sorted(baseline_results):
        row = f"{n_rote}/6   "
        for name in BASELINE_NAMES:
            v = baseline_results[n_rote][name].get("dwarf")
            row += f"  {v:14.3f}" if v is not None else f"  {'--':>14}"
        print(row)


if __name__ == "__main__":
    main()
