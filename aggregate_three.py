"""
Aggregate three_probes.json files across the sweep into a comparison table.

For each (mode, schedule) cell, averaged over seeds:

  L_canon_pref       — fraction of seeds where logprob prefers "planet" for P10
  L_reclassify_pref  — fraction where logprob prefers "dwarf" (None if OOV)

  D_frac_canon       — mean fraction of P10 samples nearest the canon centroid
  D_frac_reclassify  — mean fraction nearest the reclassify centroid
  D_sim_canon        — mean cosine similarity to canon centroid
  D_sim_reclassify   — mean cosine similarity to reclassify centroid

  J_canon            — mean fraction of P10 continuations the judge classifies as canon
  J_reclassify       — mean fraction classified as reclassify
  J_extend           — mean fraction classified as extend (mentions E*)
  J_hedge            — mean fraction classified as hedge (both planet and dwarf)
  J_off_topic        — mean fraction off-topic

Disagreement is the data: rows where L and D disagree, or D and J disagree,
are where the three-probe framework earns its keep.
"""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="sweep")
    args = ap.parse_args()
    root = Path(args.sweep_dir)
    manifest = json.loads((root / "manifest.json").read_text())

    bins = defaultdict(list)
    for entry in manifest:
        tp = json.loads((Path(entry["run_dir"]) / "three_probes.json").read_text())
        # ── L: which verdict does logprob prefer per move?
        canon_rows = tp["L_logprob"].get("canon", [])
        recls_rows = tp["L_logprob"].get("reclassify", [])
        l_canon_pref = (canon_rows[0].get("preferred") == "positive"
                        if canon_rows and "preferred" in canon_rows[0] else None)
        l_recls_pref = (recls_rows[0].get("preferred") == "positive"
                        if recls_rows and "preferred" in recls_rows[0] else None)
        # ── D: drift fractions and similarities
        d = tp["D_drift"]
        n = max(d["n_target_samples"], 1)
        nearest = d.get("nearest_centroid_counts", {})
        sim = d.get("mean_similarity", {})
        # ── J: judge label fractions
        j = tp["J_judge"]["label_fractions"]
        bins[(entry["mode"], entry["schedule"])].append({
            "L_canon_pref": 1.0 if l_canon_pref else (0.0 if l_canon_pref is False else None),
            "L_reclassify_pref": 1.0 if l_recls_pref else (0.0 if l_recls_pref is False else None),
            "D_frac_canon":      nearest.get("canon", 0) / n,
            "D_frac_reclassify": nearest.get("reclassify", 0) / n,
            "D_sim_canon":      sim.get("canon"),
            "D_sim_reclassify": sim.get("reclassify"),
            "J_canon":      j.get("canon", 0.0),
            "J_reclassify": j.get("reclassify", 0.0),
            "J_extend":     j.get("extend", 0.0),
            "J_hedge":      j.get("hedge", 0.0),
            "J_off_topic":  j.get("off_topic", 0.0),
        })

    summary = {}
    for key, rows in bins.items():
        agg = {}
        for k in rows[0]:
            agg[k] = {"mean": safe_mean([r[k] for r in rows]),
                      "std": safe_std([r[k] for r in rows]),
                      "n": len([r[k] for r in rows if r[k] is not None])}
        summary[f"{key[0]}__{key[1]}"] = agg

    (root / "three_probes_summary.json").write_text(json.dumps(summary, indent=2))

    headers = ["L_canon", "L_reclas", "D_canon%", "D_reclas%", "D_sim_C", "D_sim_R",
               "J_canon", "J_reclas", "J_extend", "J_hedge"]
    lines = [f"{'mode':<10} {'schedule':<12} " + " ".join(f"{h:<9}" for h in headers)]
    lines.append("-" * len(lines[0]))
    for mode in ["unlabeled", "dwarf", "planet"]:
        for sched in ["canon-only", "curriculum", "mixed"]:
            key = f"{mode}__{sched}"
            if key not in summary:
                continue
            a = summary[key]
            def cell(k, fmt="{:.2f}"):
                v = a[k]["mean"]
                return ("--" if v is None else fmt.format(v)).ljust(9)
            row = (f"{mode:<10} {sched:<12} "
                   f"{cell('L_canon_pref')}"
                   f"{cell('L_reclassify_pref')}"
                   f"{cell('D_frac_canon')}"
                   f"{cell('D_frac_reclassify')}"
                   f"{cell('D_sim_canon', '{:.3f}')}"
                   f"{cell('D_sim_reclassify', '{:.3f}')}"
                   f"{cell('J_canon')}"
                   f"{cell('J_reclassify')}"
                   f"{cell('J_extend')}"
                   f"{cell('J_hedge')}")
            lines.append(row)

    (root / "three_probes_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
