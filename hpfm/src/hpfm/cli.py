"""CLI: ``hpfm ir.png vi.png fused.png``."""

from __future__ import annotations

import argparse

from hpfm import hpfm


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="HPFM: human-preference IVIF score")
    p.add_argument("ir")
    p.add_argument("vis")
    p.add_argument("fused")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    print(hpfm(args.ir, args.vis, args.fused, device=args.device))


if __name__ == "__main__":
    main()
