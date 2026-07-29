#!/usr/bin/env python3
"""Cheap qualitative control check for LDT-Fusion.

1) Manual VI token sweep → save fused Y + MAFD vs baseline
2) Optional: dump auto router + gates for one pair

Example:
  python scripts/qual_control.py \\
    --ckpt checkpoint/ldt_20260723_1611/ldt_20260723_1611-best-eepoch=015.ckpt \\
    --data /data/ykx/LLVIP/test --n 8 --out result/qual_1611
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

raise SystemExit(
    'qual_control.py targets the removed token/SRC path. '
    'Use dtrc/ + scripts/train_dtrc_stage2.py for the new control line.'
)

from ldt.dataset import IVIFDataset
from ldt.ldt_fusion import LDTFusion
from ldt.run_utils import merge_inference_args_from_checkpoint
from modules.ldt import VI_DEGRADATION_NAMES


def _parse_args():
    p = argparse.ArgumentParser(description='Manual VI control MAFD + gate dump')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--data', type=str, default='/data/ykx/LLVIP/test')
    p.add_argument('--out', type=str, default='result/qual_control')
    p.add_argument('--n', type=int, default=8, help='number of pairs (sorted names)')
    p.add_argument('--names', type=str, default='', help='comma-separated filenames; overrides --n')
    p.add_argument('--baseline_vi', type=str, default='normal')
    p.add_argument(
        '--conds',
        type=str,
        default='low;veil',
        help='VI tokens separated by `;` (IR is always clean)',
    )
    p.add_argument('--no_cuda', action='store_true')
    p = LDTFusion.add_model_specific_args(p)
    return p.parse_args()


def _y_u8(t: torch.Tensor) -> np.ndarray:
    return (t.detach().float().clamp(0, 1).squeeze().cpu().numpy() * 255.0).astype(np.uint8)


def _save_color(vi_bgr: np.ndarray, y: np.ndarray, path: Path) -> None:
    ycrcb = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = y
    cv2.imwrite(str(path), cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR))


@torch.no_grad()
def _forward(model: LDTFusion, ir: torch.Tensor, vi: torch.Tensor, alpha_vi, strength: float):
    model.control_mode = 'manual'
    a_vi = alpha_vi.to(device=vi.device, dtype=vi.dtype)
    if a_vi.dim() == 1:
        a_vi = a_vi.unsqueeze(0)
    a_ir = model._ir_normal_alpha(ir.shape[0], ir.device, dtype=ir.dtype)
    model._manual_alpha_vi = a_vi
    model.control_strength = float(strength)
    return model.fuser(
        torch.cat([ir, vi], dim=1),
        alpha_ir=a_ir,
        alpha_vi=a_vi.expand(vi.shape[0], -1),
        eta_scale=strength,
        return_aux=True,
    )


def main():
    cli = _parse_args()
    cli.device = 'cpu' if cli.no_cuda else 'cuda:0'
    cli.control_mode = 'manual'
    cli.degradation_vi = cli.baseline_vi

    args = merge_inference_args_from_checkpoint(cli.ckpt, cli)
    # merge() drops script-only flags; keep them from CLI.
    for k in ('n', 'names', 'out', 'baseline_vi', 'conds', 'data'):
        setattr(args, k, getattr(cli, k))
    device = args.device
    # PyTorch 2.6+ defaults weights_only=True; ckpt stores argparse.Namespace.
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = LDTFusion(args)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.to(device).eval()

    root = Path(args.data)
    samples = sorted((root / 'vi').iterdir())
    samples = [p.name for p in samples if p.is_file()]
    if args.names.strip():
        names = [n.strip() for n in args.names.split(',') if n.strip()]
    else:
        names = samples[: max(1, args.n)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_vi = model._resolve_manual_alpha(
        args.baseline_vi, VI_DEGRADATION_NAMES, model.fuser.token_bank.num_vi,
    )
    assert base_vi is not None

    conds = []
    for spec in args.conds.split(';'):
        vi_s = spec.strip()
        if not vi_s:
            continue
        a_vi = model._resolve_manual_alpha(vi_s, VI_DEGRADATION_NAMES, model.fuser.token_bank.num_vi)
        conds.append((vi_s, a_vi))

    # MAFD table: mean |Y_cond - Y_base| over pixels, averaged over images (0–255)
    mafd = {vi: [] for vi, _ in conds}
    rows = []

    for name in names:
        ir = IVIFDataset.gray_read(str(root / 'ir' / name)).unsqueeze(0).to(device)
        vi = IVIFDataset.color_read(str(root / 'vi' / name)).unsqueeze(0).to(device)
        vi_bgr = cv2.imread(str(root / 'vi' / name), cv2.IMREAD_COLOR)
        if vi_bgr is None:
            raise FileNotFoundError(root / 'vi' / name)

        stem = Path(name).stem
        img_dir = out / stem
        img_dir.mkdir(parents=True, exist_ok=True)

        base = _forward(model, ir, vi, base_vi, args.control_strength)
        y0 = _y_u8(base['fused'])
        _save_color(vi_bgr, y0, img_dir / f'fused_{args.baseline_vi}.png')

        # auto router + gates (reference)
        model.control_mode = 'auto'
        auto = model.fuser(torch.cat([ir, vi], dim=1), return_aux=True)
        _save_color(vi_bgr, _y_u8(auto['fused']), img_dir / 'fused_auto.png')
        if auto.get('gate_vi') is not None:
            g = auto['gate_vi']
            save_image((1.0 - g[:, :1]).clamp(0, 1) if g.shape[1] > 1 else g.clamp(0, 1),
                       img_dir / 'gate_vi_union.png')
            for k, n in enumerate(VI_DEGRADATION_NAMES):
                if g.shape[1] > k:
                    save_image(g[:, k : k + 1], img_dir / f'gate_vi_{n}.png')
        if auto.get('alpha_vi') is not None:
            with open(img_dir / 'router.txt', 'w', encoding='utf-8') as f:
                f.write('VI: ' + ', '.join(
                    f'{n}={w:.3f}' for n, w in zip(VI_DEGRADATION_NAMES, auto['alpha_vi'][0].tolist())
                ) + '\n')
                f.write('IR: uncontrolled (no SRC/gate)\n')

        y0f = y0.astype(np.float32)
        for vi_s, a_vi in conds:
            out_c = _forward(model, ir, vi, a_vi, args.control_strength)
            y = _y_u8(out_c['fused'])
            _save_color(vi_bgr, y, img_dir / f'fused_{vi_s}.png')
            d = float(np.mean(np.abs(y.astype(np.float32) - y0f)))
            mafd[vi_s].append(d)
            rows.append((name, vi_s, d))

    print(f'ckpt={args.ckpt}')
    print(f'baseline=VI:{args.baseline_vi} (IR fixed clean)  strength={args.control_strength}')
    print(f'n_images={len(names)}  out={out}')
    print('\nMAFD vs baseline (mean |ΔY| in 0–255):')
    print(f'{"condition":<28} {"mean":>8} {"median":>8} {"min":>8} {"max":>8}')
    for key, vals in mafd.items():
        a = np.asarray(vals, dtype=np.float64)
        print(f'{key:<28} {a.mean():8.3f} {np.median(a):8.3f} {a.min():8.3f} {a.max():8.3f}')

    with open(out / 'mafd.csv', 'w', encoding='utf-8') as f:
        f.write('name,condition,mafd\n')
        for name, cond, d in rows:
            f.write(f'{name},{cond},{d:.6f}\n')
    print(f'\nwrote {out / "mafd.csv"}')
    print('判读: mean≪1 → 几乎不可控; ~2–5 弱; >8 肉眼通常可见。再肉眼看各 fused_*.png。')


if __name__ == '__main__':
    main()
