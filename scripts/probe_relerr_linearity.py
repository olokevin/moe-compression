#!/usr/bin/env python
"""Is per-layer ``rel_err`` a linear predictor of final HellaSwag accuracy?

The output-error ladder (``probe_output_error.py``) predicted two measured points
to within 0.1pt using the *layer-averaged* rel_err, which is why it is the repo's
screening instrument. But averaging over four layers hides the question the
allocation problem actually needs answered:

    if I make layer L a bit worse and layer L' a bit better, at constant cost,
    does the model's accuracy stay put?

That question only has a clean answer if accuracy is (locally) an **additive,
linear** function of the per-layer errors. This script tests exactly that: for
each captured layer independently, regress that layer's rel_err against the
measured end-to-end HellaSwag acc_norm over every selector/budget point for
which both numbers exist, and report slope / intercept / R².

Two readings matter:
  * **High R² per layer** → rel_err is a valid per-layer objective, so minimizing
    total rel_err under a budget is the right allocation program.
  * **Similar slopes across layers** → a *uniform* per-layer error target is
    optimal, and the interesting variation is in each layer's rel_err(p, rho)
    *cost curve*, not in its sensitivity.

Reads only cached JSON + the measured-accuracy table below; no GPU, no model.
"""

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Measured HellaSwag 0-shot acc_norm for (ladder row name, rho). Every entry is a
# real eval from this repo's docs; the ladder rows are keyed by the same names
# probe_output_error.py emits. Sources:
#   oracle_mag rho=0.5/0.25/0.125          docs/exps/.../q3_30b_dynamic_active.md
#   oracle_mag_noW rho=0.125 (Q1)          77.11
#   oracle_up rho=0.125 (Q2)               71.30
#   probe_q4_k1.0 rho=0.125                76.95
#   probe_q3_k0.5 rho=0.125                76.37  (A100-3, 2026-08-12)
#   probe_q3_k0.25 rho=0.125               74.56
MEASURED = {
    ("oracle_mag", 0.5): 78.54,
    ("oracle_mag", 0.25): 78.28,
    ("oracle_mag", 0.125): 76.84,
    ("oracle_mag_noW", 0.125): 77.11,
    ("oracle_up", 0.125): 71.30,
    ("probe_q4_k1.0", 0.125): 76.95,
    ("probe_q3_k0.5", 0.125): 76.37,
    ("probe_q3_k0.25", 0.125): 74.56,
}

# Held-out check of the fitted slope: reuse-probe rows measured *after* the fit,
# whose accuracy was pre-registered from it. (rel_err values are the 4-layer
# reuse-probe screen, docs/results/idea_pilot/input_alloc.json.)
HELDOUT = [
    # label,                rel_err, predicted, measured
    ("reuse p=.25 rho=.10", 0.4434, 74.2, 74.06),
    ("  + router alloc",    0.4183, 74.9, 74.64),
]

DENSE = 78.56


def fit(x, y):
    """OLS ``y = a*x + b``; returns (slope, intercept, r2, n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2, len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default=os.path.join(
        _REPO, "docs/results/idea_pilot/output_error_ladder.json"))
    ap.add_argument("--include-dense", action="store_true",
                    help="add the (rel_err=0, acc=dense) anchor point")
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/idea_pilot/relerr_linearity.json"))
    args = ap.parse_args()

    d = json.load(open(args.ladder))
    ratios = d["ratios"]
    rows = d["rows"]
    layers = sorted({r["layer"] for r in rows})

    # points[layer] = [(rel_err, acc, label), ...]
    points = {l: [] for l in layers}
    pooled = {}          # label -> {layer: rel_err, "acc": acc}
    for r in rows:
        for rho in ratios:
            key = (r["name"], rho)
            if key not in MEASURED:
                continue
            e = r.get(f"rel_err@rho{rho}")
            if e is None:
                continue
            label = f"{r['name']}@{rho}"
            points[r["layer"]].append((e, MEASURED[key], label))
            pooled.setdefault(label, {"acc": MEASURED[key]})[r["layer"]] = e

    if args.include_dense:
        for l in layers:
            points[l].append((0.0, DENSE, "dense"))
        pooled["dense"] = {"acc": DENSE, **{l: 0.0 for l in layers}}

    out = {"layers": {}, "measured_points": len(pooled)}
    print(f"[linearity] {len(pooled)} measured (selector,budget) points, "
          f"{len(layers)} layers\n")
    print("all points pooled (both families):")
    print(f"{'layer':>6} {'n':>3} {'slope (pt/unit)':>16} {'intercept':>10} {'R2':>7}")
    slopes = []
    for l in layers:
        pts = sorted(points[l])
        f = fit([p[0] for p in pts], [p[1] for p in pts])
        if f is None:
            continue
        a, b, r2, n = f
        slopes.append(a)
        out["layers"][str(l)] = {"slope": a, "intercept": b, "r2": r2, "n": n,
                                 "points": [{"rel_err": p[0], "acc": p[1],
                                             "label": p[2]} for p in pts]}
        print(f"{l:>6} {n:>3} {a:>16.2f} {b:>10.2f} {r2:>7.4f}")

    # Per-layer, restricted to the fixed-budget (mis-selection) family -- the
    # regime every scorer comparison and every layer-vs-layer trade lives in.
    print("\nper layer, fixed-budget family only (rho=0.125, varying selector):")
    print(f"{'layer':>6} {'n':>3} {'slope (pt/unit)':>16} {'intercept':>10} {'R2':>7}")
    fslopes = []
    for l in layers:
        pts = sorted(p for p in points[l] if p[2].endswith("@0.125"))
        f = fit([p[0] for p in pts], [p[1] for p in pts])
        if f is None:
            continue
        a, b, r2, n = f
        fslopes.append(a)
        out["layers"].setdefault(str(l), {})["fixed_budget"] = {
            "slope": a, "intercept": b, "r2": r2, "n": n,
            "points": [{"rel_err": p[0], "acc": p[1], "label": p[2]} for p in pts],
        }
        print(f"{l:>6} {n:>3} {a:>16.2f} {b:>10.2f} {r2:>7.4f}")
    if fslopes:
        fs = np.array(fslopes)
        out["fixed_budget_slope_spread"] = {
            "min": float(fs.min()), "max": float(fs.max()),
            "mean": float(fs.mean()), "std": float(fs.std()),
            "cv": float(fs.std() / abs(fs.mean()))}
        print(f"[fixed-budget slopes] mean {fs.mean():.2f}, std {fs.std():.2f}, "
              f"CV {fs.std()/abs(fs.mean()):.3f}")

    # layer-averaged rel_err (the instrument as currently used) for reference
    avg_pts = []
    for label, rec in pooled.items():
        es = [rec[l] for l in layers if l in rec]
        if len(es) == len(layers):
            avg_pts.append((float(np.mean(es)), rec["acc"], label))
    f = fit([p[0] for p in avg_pts], [p[1] for p in avg_pts])
    if f:
        a, b, r2, n = f
        out["layer_averaged"] = {"slope": a, "intercept": b, "r2": r2, "n": n,
                                 "points": [{"rel_err": p[0], "acc": p[1],
                                             "label": p[2]} for p in
                                            sorted(avg_pts)]}
        print(f"\n{'avg':>6} {n:>3} {a:>16.2f} {b:>10.2f} {r2:>7.4f}   "
              f"<- layer-averaged (the instrument in use)")

    # ---- stratified fits -------------------------------------------------
    # rel_err is produced two different ways and the doc already noted the slopes
    # differ: moving along the *oracle's own budget ladder* (honestly fewer
    # channels) versus *mis-selecting* at a fixed budget. Pooling them is what
    # costs the fit its R2, and only the fixed-budget family is the right ruler
    # for comparing scorers -- or for trading error between layers at fixed cost.
    FAMILIES = {
        "budget_ladder": [f"oracle_mag@{r}" for r in (0.5, 0.25, 0.125)],
        "fixed_rho0.125": [k for k in pooled
                           if k.endswith("@0.125") and k != "dense"],
    }
    out["families"] = {}
    print("\n[stratified] the two ways rel_err arises are NOT the same ruler:")
    for fam, labels in FAMILIES.items():
        sub = [(float(np.mean([pooled[l][x] for x in layers if x in pooled[l]])),
                pooled[l]["acc"], l) for l in labels if l in pooled]
        f = fit([p[0] for p in sub], [p[1] for p in sub])
        if not f:
            continue
        a, b, r2, n = f
        out["families"][fam] = {"slope": a, "intercept": b, "r2": r2, "n": n,
                                "points": [{"rel_err": p[0], "acc": p[1],
                                            "label": p[2]} for p in sorted(sub)]}
        print(f"  {fam:16s} n={n} slope={a:8.2f} pt/unit  R2={r2:.4f}")
    if {"budget_ladder", "fixed_rho0.125"} <= set(out["families"]):
        sa = out["families"]["budget_ladder"]["slope"]
        sb = out["families"]["fixed_rho0.125"]["slope"]
        print(f"  => mis-selection at fixed budget costs {abs(sb/sa):.1f}x more "
              f"accuracy per unit rel_err than an honestly smaller budget.")
        print("  => use the fixed-budget slope when screening scorers.")

    if slopes:
        s = np.array(slopes)
        out["slope_spread"] = {"min": float(s.min()), "max": float(s.max()),
                               "mean": float(s.mean()), "std": float(s.std()),
                               "cv": float(s.std() / abs(s.mean()))}
        print(f"\n[slopes] mean {s.mean():.2f}, std {s.std():.2f}, "
              f"CV {s.std()/abs(s.mean()):.3f}, range [{s.min():.2f}, {s.max():.2f}]")
    if fslopes:
        fs = np.array(fslopes)
        a_d, _ = np.polyfit([int(l) for l in layers][:len(fs)], fs, 1)
        print(f"[reading] fixed-budget slopes fall with depth ({a_d:+.3f} pt/unit "
              f"per layer): early layers are\n          ~"
              f"{abs(fs.min()/fs.max()):.1f}x more accuracy-sensitive than late "
              f"ones. So a UNIFORM per-layer\n          rel_err target is NOT "
              f"optimal, and minimizing the *unweighted* sum of\n          rel_err "
              f"across layers is the wrong program -- it starves the sensitive\n"
              f"          early layers. Measured: the unweighted schedule lost "
              f"0.16pt.\n          Use layer_slope_weights() in "
              f"scripts/probe_layer_surface.py.")

    # ---- held-out validation of the fixed-budget slope --------------------
    fam = out.get("families", {}).get("fixed_rho0.125")
    if fam:
        print("\n[held-out] rows measured AFTER the fit, predictions pre-registered:")
        errs = []
        for label, e, pred, meas in HELDOUT:
            fit_pred = fam["slope"] * e + fam["intercept"]
            errs.append(abs(meas - pred))
            print(f"  {label:22s} rel_err={e:.4f}  pre-registered {pred:.2f}  "
                  f"measured {meas:.2f}  (|err| {abs(meas-pred):.2f}pt; "
                  f"raw fit says {fit_pred:.2f})")
        out["heldout"] = [{"label": l, "rel_err": e, "predicted": p, "measured": m}
                          for l, e, p, m in HELDOUT]
        out["heldout_mae"] = float(np.mean(errs))
        print(f"  MAE {np.mean(errs):.2f}pt over {len(errs)} held-out points "
              f"(stderr on each measurement is ~0.43pt)")
        print("  => the ladder is validated for SELECTOR changes. It is NOT valid "
              "for\n     cross-layer budget moves: the per-layer schedule was "
              "predicted 74.9,\n     measured 73.90 (1.0pt miss) -- see "
              "efficient_scorer.md 'The schedule failure'.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
