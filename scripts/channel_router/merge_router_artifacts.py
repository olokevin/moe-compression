#!/usr/bin/env python
"""Merge per-layer-group router artifacts into one all-layer artifact.

The 48 layers are trained as parallel jobs (one per GPU, a few layers each); the PPL
ladder and the eval protocol want a single file keyed by absolute layer index.
"""

import argparse
import glob
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob of artifacts to merge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    merged = {"meta": {"merged_from": files}, "layers": {}}
    for f in files:
        d = torch.load(f, map_location="cpu")
        for li, ent in d["layers"].items():
            if int(li) in merged["layers"]:
                print(f"[merge] warning: layer {li} appears twice; keeping the later file")
            merged["layers"][int(li)] = ent
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(merged, args.out)
    print(f"[merge] {len(files)} files -> {args.out} with "
          f"{len(merged['layers'])} layers: {sorted(merged['layers'])}")


if __name__ == "__main__":
    main()
