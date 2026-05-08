"""
Traditional statistical baselines on the same labeled feature data the
transformer trains on. The point is to ask: given just the (features,
label) pairs in the corpus — without any sequence modeling, schedule
effects, or fine-tuning dynamics — what does a shallow classifier think
P10 is?

Four baselines, all reading the same training set:

  - logreg : multinomial logistic regression on (mass, diameter, orbit).
  - gnb    : Gaussian naive Bayes (class-conditional Gaussians per axis).
  - knn3   : 3-nearest-neighbors in feature space.
  - bayes  : explicit Bayes-rule computation with class-conditional
             Gaussian likelihoods, written by hand for transparency. This
             matches gnb numerically but exposes the calculation.

Training set: union of phase1 entities (with their categories) and
phase2 entities (with the phase2_mode label, e.g. "dwarf"). Entity P10
is in phase1 with label "planet" and is therefore in the training set
— same information the transformer sees in phase 1 of training.

Usage:
  python baseline_classifier.py --entities sweep_small/data_dwarf_r0of6_e1.00/entities.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

LABELS_OF_INTEREST = ("planet", "dwarf")


def load_training_set(entities_path):
    """Return X (n, 3), y (n,), p10_features (3,), and the label list."""
    ents = json.loads(Path(entities_path).read_text())
    phase2_mode = ents.get("phase2_mode", "unlabeled")
    X, y = [], []
    p10_feats = None
    for name, info in ents["phase1"].items():
        X.append(info["features"]); y.append(info["category"])
        if name == "P10":
            p10_feats = info["features"]
    if phase2_mode in ("dwarf", "planet"):
        for name, info in ents.get("phase2", {}).items():
            X.append(info["features"]); y.append(phase2_mode)
    return np.array(X, dtype=float), np.array(y), np.array(p10_feats, dtype=float)


def predict_proba_dict(model, x_query, classes):
    """Return {label: probability} from a fitted sklearn classifier."""
    proba = model.predict_proba(x_query.reshape(1, -1))[0]
    return {cls: float(p) for cls, p in zip(classes, proba)}


def hand_bayes_posterior(X, y, x_query, var_floor=0.05):
    """Class-conditional Gaussian likelihoods with hand-computed posterior.

    For each label c, fit per-axis Gaussian (μ, σ²) using training rows with
    label c. Apply naive (per-axis independent) likelihood and uniform prior.
    Floor the variance to avoid degenerate point-mass estimates when a
    single class has all-equal feature values.
    """
    labels = sorted(set(y))
    posteriors = {}
    log_lik = {}
    for c in labels:
        mask = (y == c)
        if mask.sum() == 0:
            continue
        Xc = X[mask]
        mu = Xc.mean(axis=0)
        var = Xc.var(axis=0, ddof=1) if mask.sum() > 1 else np.full(X.shape[1], var_floor)
        var = np.maximum(var, var_floor)
        # log p(x | c) = -0.5 * sum( log(2 pi var_i) + (x_i - mu_i)^2 / var_i )
        ll = -0.5 * np.sum(np.log(2 * math.pi * var)
                           + (x_query - mu) ** 2 / var)
        log_lik[c] = float(ll)
    # Uniform prior: posterior ∝ likelihood
    if not log_lik:
        return {}
    m = max(log_lik.values())
    unnorm = {c: math.exp(ll - m) for c, ll in log_lik.items()}
    Z = sum(unnorm.values())
    return {c: v / Z for c, v in unnorm.items()}


def evaluate_baselines(entities_path):
    """Run all four baselines on a single corpus. Return dict of per-baseline
    {label: prob} for the LABELS_OF_INTEREST."""
    X, y, p10 = load_training_set(entities_path)

    out = {}

    lr = LogisticRegression(max_iter=2000).fit(X, y)
    out["logreg"] = predict_proba_dict(lr, p10, list(lr.classes_))

    gnb = GaussianNB().fit(X, y)
    out["gnb"] = predict_proba_dict(gnb, p10, list(gnb.classes_))

    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    out["knn3"] = predict_proba_dict(knn, p10, list(knn.classes_))

    out["bayes"] = hand_bayes_posterior(X, y, p10)

    # Restrict to labels of interest; missing entries (e.g. "dwarf" not in
    # training set in unlabeled mode) become None for downstream consistency.
    summary = {}
    for name, dist in out.items():
        summary[name] = {lab: dist.get(lab) for lab in LABELS_OF_INTEREST}
    summary["_meta"] = {
        "p10_features": p10.tolist(),
        "n_train": int(len(y)),
        "label_counts": {c: int((y == c).sum()) for c in sorted(set(y))},
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summary = evaluate_baselines(args.entities)
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
