"""
Aggregate tension.json files across the sweep into a comparison table.

For each (mode, schedule, rote_control, p10_edge) cell, averaged over seeds:

  P_planet(P10)       — mean next-token probability of "planet" given P10's
                        feature prompt
  P_dwarf(P10)        — same for "dwarf"
  tension_index(P10)  — mean of min(P_planet/thr_p, P_dwarf/thr_d), the
                        sweep's headline scalar
  cell_mode           — most common 2x2 cell across seeds
                        (tension / canon / swap / nothing)
  thr_planet, thr_dwarf — mean per-label calibration thresholds (anchor means)

The TENSION cell is the one where both labels meet their anchor-mean
thresholds — what the rote-control should *not* produce, and what an
overlap run with low p10_edge *should*.
"""
import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


ABS_TENSION_FLOOR = 0.10  # both labels must hold at least 10% mass


def cell_key(entry):
    return (entry["mode"], entry["schedule"],
            entry.get("n_eris_rote",
                      entry.get("n_eris", 5) if entry.get("rote_control") else 0),
            entry.get("n_eris", 5),
            entry.get("p10_edge", 1.0),
            entry.get("n_embd", 32), entry.get("n_layer", 4))


def absolute_cell(p_p, p_d, floor=ABS_TENSION_FLOOR):
    """A cell label that doesn't depend on anchor thresholds.

    Useful when anchors themselves are degraded (e.g. catastrophic forgetting
    drags thr_planet down to ~0, so the threshold-based cell label is
    misleading). Here both labels must clear an absolute floor for TENSION.
    """
    if p_p is None or p_d is None:
        return "n/a"
    if p_p >= floor and p_d >= floor:
        return "tension_abs"
    if p_p >= floor:
        return "canon_abs"
    if p_d >= floor:
        return "swap_abs"
    return "nothing_abs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", nargs="+", default=["sweep"],
                    help="One or more sweep directories whose manifests to "
                         "merge. Output is written to the first directory.")
    args = ap.parse_args()
    roots = [Path(d) for d in args.sweep_dir]
    manifest = []
    for root in roots:
        manifest.extend(json.loads((root / "manifest.json").read_text()))

    bins = defaultdict(list)
    for entry in manifest:
        path = Path(entry.get("tension") or entry["run_dir"]) / "tension.json"
        if not path.exists():
            path = Path(entry["run_dir"]) / "tension.json"
        if not path.exists():
            continue
        t = json.loads(path.read_text())
        p10 = t["targets"].get("P10", {})
        bins[cell_key(entry)].append({
            "P_planet": p10.get("P_planet"),
            "P_dwarf": p10.get("P_dwarf"),
            "tension_index": p10.get("tension_index"),
            "cell": p10.get("cell"),
            "thr_planet": t["thresholds"].get("planet"),
            "thr_dwarf": t["thresholds"].get("dwarf"),
        })

    summary = {}
    for key, rows in bins.items():
        agg = {}
        for k in ("P_planet", "P_dwarf", "tension_index", "thr_planet", "thr_dwarf"):
            agg[k] = {"mean": safe_mean([r[k] for r in rows]),
                      "std":  safe_std([r[k] for r in rows]),
                      "n":    len([r for r in rows if r[k] is not None])}
        cells = [r["cell"] for r in rows if r["cell"]]
        agg["cell_mode"] = Counter(cells).most_common(1)[0][0] if cells else None
        agg["cell_counts"] = dict(Counter(cells))
        abs_cells = [absolute_cell(r["P_planet"], r["P_dwarf"]) for r in rows]
        agg["abs_cell_mode"] = Counter(abs_cells).most_common(1)[0][0] if abs_cells else None
        agg["abs_cell_counts"] = dict(Counter(abs_cells))
        mode, sched, n_rote, n_eris, edge, n_embd, n_layer = key
        k = (f"{mode}__{sched}__r{n_rote}of{n_eris}__e{edge:.2f}"
             f"__d{n_embd}l{n_layer}")
        agg["mode"] = mode
        agg["schedule"] = sched
        agg["n_eris_rote"] = n_rote
        agg["n_eris"] = n_eris
        agg["rote_fraction"] = n_rote / n_eris if n_eris else 0
        agg["p10_edge"] = edge
        agg["n_embd"] = n_embd
        agg["n_layer"] = n_layer
        summary[k] = agg

    (roots[0] / "tension_summary.json").write_text(json.dumps(summary, indent=2))
    root = roots[0]

    headers = ["P(planet)", "P(dwarf)", "T_idx", "cell", "abs_cell"]
    cols = ["mode", "schedule", "rote/n", "edge", "model"] + headers
    widths = [11, 12, 8, 6, 8, 11, 11, 9, 10, 12]
    lines = [" ".join(c.ljust(w) for c, w in zip(cols, widths))]
    lines.append("-" * len(lines[0]))
    for key in sorted(summary.keys()):
        a = summary[key]

        def fcell(k, fmt="{:.3f}"):
            v = a[k]["mean"]
            return "--" if v is None else fmt.format(v)

        rote = f"{a['n_eris_rote']}/{a['n_eris']}"
        model = f"d{a['n_embd']}l{a['n_layer']}"
        edge_s = f"{a['p10_edge']:.1f}"
        vals = [a["mode"], a["schedule"], rote, edge_s, model,
                fcell("P_planet"), fcell("P_dwarf"), fcell("tension_index"),
                a["cell_mode"] or "--", a["abs_cell_mode"] or "--"]
        lines.append(" ".join(v.ljust(w) for v, w in zip(vals, widths)))

    (root / "tension_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
