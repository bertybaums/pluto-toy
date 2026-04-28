"""
Aggregate probe results from a sweep into summary tables.

Reads sweep/manifest.json and per-run probe.json files, produces:
  - sweep/summary.json       — per-condition means and stds
  - sweep/summary_table.txt  — human-readable table

Key metrics:
  P_planet_P10   : mean P("planet" | "P10 is a")            — direct labeling
  P_dwarf_P10    : mean P("dwarf"  | "P10 is a")            — direct labeling
  d_P10_E1       : contextual cosine distance P10 ↔ E1      — feature-twin
  d_P10_P1       : contextual cosine distance P10 ↔ P1      — vs canon planet
  d_E1_E2        : contextual cosine distance E1 ↔ E2       — within-Eris baseline
  d_P1_P2        : contextual cosine distance P1 ↔ P2       — within-canon baseline
"""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def get_pair(d, a, b):
    return d.get(f"{a}|{b}", d.get(f"{b}|{a}"))


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None, 0)
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return (m, math.sqrt(v), len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="sweep")
    args = ap.parse_args()

    root = Path(args.sweep_dir)
    manifest = json.loads((root / "manifest.json").read_text())

    bins = defaultdict(list)
    for entry in manifest:
        d = json.loads(Path(entry["probe"]).read_text())
        key = (entry["mode"], entry["schedule"])
        row = {
            "P_planet_P10": d["A_direct_labeling"].get("P10", {}).get("planet"),
            "P_dwarf_P10":  d["A_direct_labeling"].get("P10", {}).get("dwarf"),
            "P_planet_E1": d["A_direct_labeling"].get("E1", {}).get("planet"),
            "P_dwarf_E1":  d["A_direct_labeling"].get("E1", {}).get("dwarf"),
            "d_P10_E1":  get_pair(d["D_contextual_geometry"], "P10", "E1"),
            "d_P10_P1":  get_pair(d["D_contextual_geometry"], "P10", "P1"),
            "d_E1_E2":   get_pair(d["D_contextual_geometry"], "E1", "E2"),
            "d_P1_P2":   get_pair(d["D_contextual_geometry"], "P1", "P2"),
        }
        bins[key].append(row)

    summary = {}
    for (mode, sched), rows in bins.items():
        keys = rows[0].keys()
        agg = {}
        for k in keys:
            m, s, n = stats([r[k] for r in rows])
            agg[k] = {"mean": m, "std": s, "n": n}
        summary[f"{mode}__{sched}"] = agg

    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    metrics = ["P_planet_P10", "P_dwarf_P10",
               "d_P10_E1", "d_P10_P1", "d_E1_E2", "d_P1_P2"]
    lines = []
    header = f"{'mode':<10} {'schedule':<12} " + " ".join(f"{m:<14}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))
    for mode in ["unlabeled", "dwarf", "planet"]:
        for sched in ["canon-only", "curriculum", "mixed"]:
            key = f"{mode}__{sched}"
            if key not in summary:
                continue
            agg = summary[key]
            row = f"{mode:<10} {sched:<12} "
            for m in metrics:
                v = agg[m]
                if v["mean"] is None:
                    row += f"{'-':<14} "
                else:
                    row += f"{v['mean']:.3f}±{v['std']:.3f}  "
            lines.append(row)

    table = "\n".join(lines)
    (root / "summary_table.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
