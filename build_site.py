"""
Build docs/index.html — a self-contained interactive walkthrough of the
methods and results so far. Embeds the sweep data + baseline values into
a single HTML page that renders Plotly plots client-side.

Reads:
  sweep_small/tension_summary.json     (combined small + large sweep summary)
  sweep_small/data_dwarf_r{N}of6_*/entities.json  (entity feature tables per corner-mix)

Writes:
  docs/index.html
"""
import json
from pathlib import Path

from baseline_classifier import evaluate_baselines

REPO = Path(__file__).parent
SUMMARY = REPO / "sweep_small" / "tension_summary.json"
SWEEP_DIR = REPO / "sweep_small"
DOCS = REPO / "docs"


def transformer_curves(summary):
    out = {}
    for n_embd, n_layer in ((32, 4), (64, 6)):
        for sched in ("curriculum", "mixed"):
            key = f"d{n_embd}l{n_layer}_{sched}"
            rows = []
            for v in summary.values():
                if v.get("n_embd") != n_embd or v.get("n_layer") != n_layer:
                    continue
                if v.get("schedule") != sched:
                    continue
                pp, pd = v["P_planet"], v["P_dwarf"]
                n_pp = max(pp.get("n", 1), 1)
                n_pd = max(pd.get("n", 1), 1)
                rows.append({
                    "n_eris_rote": v["n_eris_rote"],
                    "P_planet": pp["mean"],
                    "P_planet_se": pp["std"] / n_pp ** 0.5,
                    "P_dwarf": pd["mean"],
                    "P_dwarf_se": pd["std"] / n_pd ** 0.5,
                    "abs_cell_counts": v.get("abs_cell_counts", {}),
                })
            rows.sort(key=lambda r: r["n_eris_rote"])
            out[key] = rows
    return out


def baseline_curves():
    out = {}
    for n_rote in (0, 2, 4, 6):
        ents = SWEEP_DIR / f"data_dwarf_r{n_rote}of6_e1.00" / "entities.json"
        if not ents.exists():
            continue
        res = evaluate_baselines(str(ents))
        for name in ("logreg", "gnb", "knn3", "bayes"):
            row = {
                "n_eris_rote": n_rote,
                "P_planet": res[name].get("planet"),
                "P_dwarf": res[name].get("dwarf"),
            }
            out.setdefault(name, []).append(row)
    for v in out.values():
        v.sort(key=lambda r: r["n_eris_rote"])
    return out


def entity_tables():
    """For each corner-mix, extract every entity's name, category, features."""
    out = {}
    for n_rote in (0, 2, 4, 6):
        path = SWEEP_DIR / f"data_dwarf_r{n_rote}of6_e1.00" / "entities.json"
        if not path.exists():
            continue
        ents = json.loads(path.read_text())
        rows = []
        for name, info in ents["phase1"].items():
            rows.append({
                "name": name,
                "category": info["category"],
                "features": info["features"],
                "phase": 1,
            })
        for name, info in ents.get("phase2", {}).items():
            rows.append({
                "name": name,
                "category": ents.get("phase2_mode", "unlabeled"),
                "features": info["features"],
                "phase": 2,
            })
        out[n_rote] = rows
    return out


def main():
    summary = json.loads(SUMMARY.read_text())
    data = {
        "transformer": transformer_curves(summary),
        "baselines": baseline_curves(),
        "entities": entity_tables(),
        "axes": {
            "mass":     ["tiny", "small", "medium", "large", "huge"],
            "diameter": ["tiny", "small", "medium", "large", "huge"],
            "orbit":    ["near", "inner", "middle", "outer", "distant"],
        },
    }
    template = (REPO / "docs" / "template.html").read_text()
    html = template.replace("/* __DATA__ */", json.dumps(data))
    (DOCS / "index.html").write_text(html)
    print(f"wrote {DOCS / 'index.html'} "
          f"({len(html):,} bytes, transformer={len(data['transformer'])} curves, "
          f"baselines={len(data['baselines'])} classifiers, "
          f"entities={len(data['entities'])} corpora)")


if __name__ == "__main__":
    main()
