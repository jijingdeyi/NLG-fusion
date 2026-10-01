#!/usr/bin/env python3
"""Copy Stage-1 checkpoint into the hpfm package as weights.pt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=root.parent / "checkpoints" / "scorer_stage1_best.pt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "src" / "hpfm" / "weights.pt",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    payload = {
        "model": ckpt["model"],
        "size": int(ckpt.get("size", 224)),
        "arch": ckpt.get("arch", "resnet18_scalar"),
        "stage": ckpt.get("stage", 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"[wrote] {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
