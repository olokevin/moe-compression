#!/usr/bin/env python
"""Phase-0 exit document: aggregate P1–P6 + the PPL ladder into one verdict.

Writes ``summary.md`` with the plan's exit criteria evaluated explicitly, plus two
figures: the P1 rank curve (the paper's motivation figure) and the calibration curve
that converts mass-recall into ΔPPL (§0.3), fitted from the controlled-degradation rows
of the ladder rather than assumed.

Runs anywhere (CPU, no torch needed beyond json/matplotlib).
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fit_line(xs, ys):
    """Least-squares slope/intercept plus R²."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"slope": a, "intercept": b, "r2": (1 - ss_res / ss_tot) if ss_tot else 1.0}


def p1_figure(p1, out_png, ratio="0.125"):
    if not p1:
        return None
    fig, axes = plt.subplots(1, len(p1["layers"]), figsize=(6 * len(p1["layers"]), 4.4),
                             squeeze=False)
    summary = {}
    for ax, (layer, rows) in zip(axes[0], sorted(p1["layers"].items())):
        series = {}
        for r in rows:
            if str(r.get("ratio")) != ratio:
                continue
            key = r["mode"].replace(f"_r{r['rank']}", "_r")
            series.setdefault(key, []).append((r["rank"], r["mass_recall"]))
        for key, pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=key)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rank r")
        ax.set_ylabel("mass-recall @ B")
        ax.set_title(f"layer {layer} (rho={ratio})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        summary[layer] = {k: max(v, key=lambda t: t[1]) for k, v in series.items()}
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return summary


def calibration_figure(ladder, out_png):
    if not ladder:
        return None
    pts = {}
    dense = next((r["ppl"] for r in ladder["rows"] if r["spec"]["kind"] == "dense"), None)
    # The mask's ΔPPL has two parts: what the *budget* costs (the oracle row at that rho)
    # and what *mis-selection* costs on top. Only the second is a property of the
    # selector, so the curve is fitted on the excess over the oracle at the same rho.
    oracle = {r["spec"]["rho"]: r["ppl"] for r in ladder["rows"]
              if r["spec"]["kind"] == "oracle"}
    for r in ladder["rows"]:
        if r["spec"]["kind"] != "degrade" or dense is None:
            continue
        rho = r["spec"]["rho"]
        if rho not in oracle:
            continue
        pts.setdefault(rho, []).append(
            (r["mass_recall"], 100.0 * (r["ppl"] - oracle[rho]) / oracle[rho]))
    for rho in list(pts):
        pts[rho].append((1.0, 0.0))            # the oracle itself, by construction
    fits = {}
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for rho, ps in sorted(pts.items()):
        ps.sort(reverse=True)
        xs = [1 - p[0] for p in ps]
        ys = [p[1] for p in ps]
        ax.plot([p[0] for p in ps], ys, marker="o", label=f"rho={rho}")
        f = fit_line(xs, ys)
        if f:
            f["dppl_at_mass"] = {f"{m:.2f}": f["slope"] * (1 - m) + f["intercept"]
                                 for m in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6)}
            for tgt in (1.0, 2.0, 5.0):
                need = 1 - (tgt - f["intercept"]) / f["slope"] if f["slope"] else None
                f[f"mass_recall_for_dppl_{tgt}"] = need
            fits[str(rho)] = f
    ax.axhline(1.0, ls="--", c="k", lw=0.8)
    ax.set_xlabel("importance-mass recall of the applied mask")
    ax.set_ylabel("excess PPL over the oracle at the same budget (%)")
    ax.set_title("calibration: mis-selection cost vs mass-recall (wikitext2)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default=os.path.join(_REPO, "docs/results/channel_router"))
    ap.add_argument("--ratio", default="0.125")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    p0 = os.path.join(args.res_dir, "phase0")
    out_md = args.out or os.path.join(p0, "summary.md")
    os.makedirs(p0, exist_ok=True)

    p1 = load(os.path.join(p0, "p1_logistic_rank.json"))
    p2 = load(os.path.join(p0, "p2_gate_sufficiency.json"))
    p3 = load(os.path.join(p0, "p3_input_screening.json"))
    p4 = load(os.path.join(p0, "p4_static_dynamic.json"))
    p5 = load(os.path.join(p0, "p5_tiles_sinkhorn.json"))
    p6 = load(os.path.join(p0, "p6_temporal.json"))
    ladder = load(os.path.join(args.res_dir, "ppl_ladder.json"))

    p1_best = p1_figure(p1, os.path.join(p0, "p1_rank.png"), ratio=args.ratio)
    fits = calibration_figure(ladder, os.path.join(p0, "calibration.png"))

    L = []
    A = L.append
    A("# Channel router — Phase 0 summary\n")
    A("Generated by `scripts/channel_router/phase0_summary.py`. "
      "Model Qwen3-30B-A3B-Thinking-2507, calibration c4, 1.05M tokens/layer, "
      "layers 22 (mid) and 46 (deep).\n")

    # ---- §0.2 gate
    A("\n## §0.2 gate — oracle-mask ΔPPL\n")
    if ladder:
        dense = next((r["ppl"] for r in ladder["rows"] if r["spec"]["kind"] == "dense"), None)
        A(f"wikitext2, seqlen {ladder['seqlen']}, "
          f"{ladder['rows'][0]['windows']} windows, all 48 MoE layers masked. "
          f"Dense PPL **{dense:.4f}**.\n")
        A("\n| rho | B (of 6144) | PPL | ΔPPL % |")
        A("| --- | --- | --- | --- |")
        for r in ladder["rows"]:
            if r["spec"]["kind"] != "oracle":
                continue
            rho = r["spec"]["rho"]
            A(f"| {rho} | {int(round(rho * 6144))} | {r['ppl']:.4f} | "
              f"{r['d_ppl_pct']:+.3f}% |")
        ok = [r for r in ladder["rows"]
              if r["spec"]["kind"] == "oracle" and r["d_ppl_pct"] is not None
              and r["d_ppl_pct"] < 1.0]
        if ok:
            best = min(ok, key=lambda r: r["spec"]["rho"])
            A(f"\n**Smallest budget with ΔPPL < 1%: rho = {best['spec']['rho']} "
              f"(k = {int(round(best['spec']['rho'] * 6144))}), "
              f"ΔPPL = {best['d_ppl_pct']:+.3f}%.**")
    else:
        A("_ladder results missing_")

    # ---- §0.3 calibration
    A("\n## §0.3 calibration — mass-recall to ΔPPL\n")
    if fits:
        A("Controlled degradation of the oracle mask (drop x% of the top-B, backfill with "
          "the next ranks), so the cost of mis-selection is measured as a function of a "
          "*measured* mass-recall. The y axis is the **excess over the oracle at the same "
          "budget**, i.e. the part of ΔPPL that is the selector's fault rather than the "
          "budget's.\n")
        A("\n| rho | slope (excess PPL% per unit mass lost) | R² | mass-recall needed "
          "for +1% over oracle | for +2% |")
        A("| --- | --- | --- | --- | --- |")
        for rho, f in sorted(fits.items()):
            A(f"| {rho} | {f['slope']:.2f} | {f['r2']:.3f} | "
              f"{f['mass_recall_for_dppl_1.0']:.4f} | {f['mass_recall_for_dppl_2.0']:.4f} |")
        A("\n![calibration](calibration.png)")
    else:
        A("_no degradation rows in the ladder_")

    # ---- P1
    A("\n## P1 — intrinsic logistic rank (go/no-go)\n")
    if p1_best:
        A("Best mass-recall over the rank sweep, per curve (held-out tokens; the free-U "
          "curve solves each held-out token's embedding with V frozen):\n")
        A("\n| layer | curve | best rank | best mass-recall |")
        A("| --- | --- | --- | --- |")
        for layer, d in sorted(p1_best.items()):
            for k, (rank, mr) in sorted(d.items(), key=lambda kv: -kv[1][1]):
                A(f"| {layer} | {k} | {rank} | {mr:.4f} |")
        A("\n![p1](p1_rank.png)")
    else:
        A("_p1 results missing_")

    # ---- P2
    A("\n## P2 — gate sufficiency\n")
    if p2:
        A("\n| layer | rho | best target | mass-recall | gate-only (silu, +cn+g) |")
        A("| --- | --- | --- | --- | --- |")
        for layer, rows in sorted(p2["layers"].items()):
            by_ratio = {}
            for r in rows:
                by_ratio.setdefault(r["ratio"], []).append(r)
            for ratio, rs in sorted(by_ratio.items()):
                # bilinear_cn_g IS the oracle (mass-recall 1.0 by construction), so the
                # informative "best" is the best *cheaper* surrogate.
                cand = [r for r in rs if r["target"] != "bilinear_cn_g"]
                best = max(cand, key=lambda r: r["mass_recall"])
                gate = next((r for r in rs if r["target"] == "silu_gate_cn_g"), None)
                A(f"| {layer} | {ratio} | {best['target']} | {best['mass_recall']:.4f} | "
                  + (f"{gate['mass_recall']:.4f} |" if gate else "n/a |"))
        A("\nDecision rule: gate-only ≥ 0.97 ⇒ distil gate scores. See table.")
    # ---- P3
    A("\n## P3 — input screening\n")
    if p3:
        A("\n| layer | m | per-token mass | global(energy) | global(anova_sampled) | "
          "frac of full |")
        A("| --- | --- | --- | --- | --- | --- |")
        for layer, d in sorted(p3["layers"].items()):
            cur = d["curves"].get(args.ratio, {})
            if not cur:
                continue
            full = cur["global_energy"]["2048"]["mass_recall"]
            for m in ("16", "64", "128", "512"):
                A(f"| {layer} | {m} | {cur['per_token'][m]['mass_recall']:.3f} | "
                  f"{cur['global_energy'][m]['mass_recall']:.3f} | "
                  f"{cur['global_anova_sampled'][m]['mass_recall']:.3f} | "
                  f"{cur['global_energy'][m]['mass_recall'] / full:.3f} |")
        A("\nDecision rule: m ≤ 64 *global* coords reaching ≥ 0.90 of the full-input "
          "recall enables the outlier-passthrough branch.")
    # ---- P4
    A("\n## P4 — static / dynamic decomposition\n")
    if p4:
        A("\n| layer | rho | static predictor | recall | mass-recall |")
        A("| --- | --- | --- | --- | --- |")
        for layer, d in sorted(p4["layers"].items()):
            for row in d["static"]:
                if str(row["ratio"]) != args.ratio:
                    continue
                A(f"| {layer} | {row['ratio']} | {row['predictor']} | "
                  f"{row['recall']:.4f} | {row['mass_recall']:.4f} |")
        A("\nHot-set coverage E[|M∩H|]/B (held-out):\n")
        A("\n| layer | " + " | ".join(f"|H|={q}" for q in
                                      ["768", "1536", "3072", "6144", "12288", "24576",
                                       "49152"]) + " |")
        A("| --- " * 8 + "|")
        for layer, d in sorted(p4["layers"].items()):
            cov = d["hot_coverage"].get(args.ratio, {})
            A(f"| {layer} | " + " | ".join(
                f"{cov.get(q, float('nan')):.3f}" for q in
                ["768", "1536", "3072", "6144", "12288", "24576", "49152"]) + " |")
    # ---- P5 / P6
    A("\n## P5 — tile-ability\n")
    if p5:
        A("\n| layer | rho | construction | tiles/expert | touched tiles | "
          "oracle top-10 mass | static top-10 mass |")
        A("| --- | --- | --- | --- | --- | --- | --- |")
        for layer, d in sorted(p5["layers"].items()):
            for rho, ro in sorted(d.items()):
                for key, res in sorted(ro.items()):
                    meth, nt = key.rsplit("_nt", 1)
                    o10 = res["restricted"].get("oracle_n10", {}).get("mass_recall")
                    s10 = res["restricted"].get("static_n10", {}).get("mass_recall")
                    A(f"| {layer} | {rho} | {meth} | {nt} | {res['touched_mean']:.1f} | "
                      + (f"{o10:.4f} | " if o10 else "n/a | ")
                      + (f"{s10:.4f} |" if s10 else "n/a |"))
        A("\nDecision rule: top-10 of 64 tiles covering ≥ 95% of the oracle mass enables "
          "the level-1 tile scorer.")
    else:
        A("_p5 results missing_")
    A("\n## P6 — temporal coherence\n")
    if p6:
        A("\n| layer | rho | adjacent IoU | reuse-mask recall | d=4 IoU | d=16 IoU |")
        A("| --- | --- | --- | --- | --- | --- |")
        for layer, d in sorted(p6["layers"].items()):
            for rho, row in sorted(d.items()):
                A(f"| {layer} | {rho} | {row['1']['iou']:.4f} | "
                  f"{row['prev_mask_mass_recall']:.4f} | "
                  f"{row.get('4', {}).get('iou', float('nan')):.4f} | "
                  f"{row.get('16', {}).get('iou', float('nan')):.4f} |")
        A("\nDecision rule: adjacent IoU ≥ 0.70 ⇒ add the reuse-previous-mask variant.")
    else:
        A("_p6 results missing_")

    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
