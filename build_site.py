"""
Build docs/index.html — a self-contained interactive walkthrough.

Everything is sourced from files that are IN git, so the page rebuilds from a
fresh checkout. (The previous version read the gitignored sweep_small/, which
meant the page could not be rebuilt without re-running the May 7 sweep.)

Reads from results/:
  tension_summary_2026-05-07.json        - corner-mix sweep curves (50K + 200K)
  ensemble_summary_2026-05-22.json       - 50K bootstrap-and-retrain draws + decomposition
  ensemble_summary_200k_2026-05-23.json  - 200K draws + decomposition
  edge_scan_summary_2026-05-22.json      - 50K prototype-edge sweep
  edge_scan_summary_200k_2026-05-23.json - 200K prototype-edge sweep

Entity tables and shallow-classifier baselines are regenerated on the fly from
generate_corpus.py (corpus generation only, no training).

Writes: docs/index.html
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from baseline_classifier import evaluate_baselines

REPO = Path(__file__).parent
RESULTS = REPO / "results"
DOCS = REPO / "docs"

# size label -> (ensemble summary, edge-scan summary)
SIZES = {
    "50K":  ("ensemble_summary_2026-05-22.json", "edge_scan_summary_2026-05-22.json"),
    "200K": ("ensemble_summary_200k_2026-05-23.json", "edge_scan_summary_200k_2026-05-23.json"),
}
CELLS = ["overlap", "rote_control", "canon"]
AXES = {
    "mass":     ["tiny", "small", "medium", "large", "huge"],
    "diameter": ["tiny", "small", "medium", "large", "huge"],
    "orbit":    ["near", "inner", "middle", "outer", "distant"],
}


def gen_corpora(scratch):
    """Regenerate the four corner-mix corpora (corpus only, no training)."""
    paths = {}
    for n_rote in (0, 2, 4, 6):
        d = scratch / f"r{n_rote}"
        subprocess.run([sys.executable, "generate_corpus.py", "--out-dir", str(d),
                        "--mode", "dwarf", "--seed", "0", "--p10-edge", "1.0",
                        "--n-eris", "6", "--n-eris-rote", str(n_rote)],
                       check=True, capture_output=True)
        paths[n_rote] = d / "entities.json"
    return paths


def entity_tables(ent_paths):
    out = {}
    for n_rote, path in ent_paths.items():
        ents = json.loads(path.read_text())
        rows = [{"name": n, "category": i["category"], "features": i["features"], "phase": 1}
                for n, i in ents["phase1"].items()]
        rows += [{"name": n, "category": ents.get("phase2_mode", "unlabeled"),
                  "features": i["features"], "phase": 2}
                 for n, i in ents.get("phase2", {}).items()]
        out[n_rote] = rows
    return out


def baseline_curves(ent_paths):
    out = {}
    for n_rote, path in ent_paths.items():
        res = evaluate_baselines(str(path))
        for name in ("logreg", "gnb", "knn3", "bayes"):
            out.setdefault(name, []).append({
                "n_eris_rote": n_rote,
                "P_planet": res[name].get("planet"),
                "P_dwarf": res[name].get("dwarf")})
    for v in out.values():
        v.sort(key=lambda r: r["n_eris_rote"])
    return out


def transformer_curves(summary):
    out = {}
    for n_embd, n_layer in ((32, 4), (64, 6)):
        for sched in ("curriculum", "mixed"):
            rows = []
            for v in summary.values():
                if (v.get("n_embd") != n_embd or v.get("n_layer") != n_layer
                        or v.get("schedule") != sched):
                    continue
                pp, pd = v["P_planet"], v["P_dwarf"]
                rows.append({
                    "n_eris_rote": v["n_eris_rote"],
                    "P_planet": pp["mean"], "P_planet_se": pp["std"] / max(pp.get("n", 1), 1) ** 0.5,
                    "P_dwarf": pd["mean"], "P_dwarf_se": pd["std"] / max(pd.get("n", 1), 1) ** 0.5})
            rows.sort(key=lambda r: r["n_eris_rote"])
            out[f"d{n_embd}l{n_layer}_{sched}"] = rows
    return out


def _pt(d):
    return {"P_planet": d["P_planet"], "P_dwarf": d["P_dwarf"], "cell": d["abs_cell"]}


def ensemble_blocks():
    """Per-draw points + decomposition, per size, for the draw-a-run and
    variance panels."""
    out = {}
    for size, (ens_file, _) in SIZES.items():
        s = json.loads((RESULTS / ens_file).read_text())
        out[size] = {}
        for cell in CELLS:
            if cell not in s:
                continue
            c = s[cell]
            block = {"label": c["label"], "baseline": _pt(c["baseline"]),
                     "decomposition": c["decomposition"], "ensembles": {}}
            for ename, v in c["ensembles"].items():
                block["ensembles"][ename] = {
                    "draws": [_pt(d) for d in v["draws"]],
                    "tension_frac": v["tension_abs_frac"],
                    "P_planet": v["P_planet"], "P_dwarf": v["P_dwarf"]}
            out[size][cell] = block
    return out


def edge_blocks():
    """Parametric tension-fraction and dual-label curves vs prototype edge."""
    out = {}
    for size, (_, edge_file) in SIZES.items():
        s = json.loads((RESULTS / edge_file).read_text())
        cells = {}
        for k, v in s.items():
            cell = k.split("__e")[0]
            en = v["ensembles"]["parametric"]
            cells.setdefault(cell, []).append({
                "edge": v["p10_edge"], "tension_frac": en["tension_abs_frac"],
                "P_planet": en["P_planet"], "P_dwarf": en["P_dwarf"]})
        for c in cells:
            cells[c].sort(key=lambda r: r["edge"])
        out[size] = cells
    return out


def main():
    summary = json.loads((RESULTS / "tension_summary_2026-05-07.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        ent_paths = gen_corpora(Path(tmp))
        data = {
            "axes": AXES,
            "entities": entity_tables(ent_paths),
            "baselines": baseline_curves(ent_paths),
            "transformer": transformer_curves(summary),
            "ensemble": ensemble_blocks(),
            "edge": edge_blocks(),
        }
    template = (DOCS / "template.html").read_text()
    html = template.replace("/* __DATA__ */", json.dumps(data))
    (DOCS / "index.html").write_text(html)
    n_draws = len(data["ensemble"]["50K"]["overlap"]["ensembles"]["parametric"]["draws"])
    print(f"wrote {DOCS / 'index.html'} ({len(html):,} bytes); "
          f"ensemble sizes={list(data['ensemble'])}, edge sizes={list(data['edge'])}, "
          f"draws/cell={n_draws}")


if __name__ == "__main__":
    main()
