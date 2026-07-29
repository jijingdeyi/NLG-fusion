"""DEPRECATED: router/SRC visualization (modules removed from Fuser).

Use DTRC diagnostics once stage-2+ is trained.
"""
raise SystemExit('visualize_ldt.py is deprecated; Fuser no longer has router/SRC.')

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from ldt.dataset import IVIFDataset
from ldt.ldt_fusion import LDTFusion
from ldt.run_utils import merge_inference_args_from_checkpoint
from modules.ldt import VI_DEGRADATION_NAMES


def main():
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--ir', type=str, required=True, help='infrared image path')
    parser.add_argument('--vi', type=str, required=True, help='visible image path')
    parser.add_argument('--out', type=str, default='result/ldt_vis')
    parser.add_argument('--no_cuda', action='store_true')
    parser = LDTFusion.add_model_specific_args(parser)
    cli = parser.parse_args()
    cli.device = 'cpu' if cli.no_cuda else 'cuda:0'
    cli.control_mode = 'auto'

    args = merge_inference_args_from_checkpoint(cli.ckpt, cli)
    device = args.device
    model = LDTFusion.load_from_checkpoint(
        args.ckpt, args=args, map_location='cpu', strict=True,
    ).to(device).eval()

    # Same Y/gray read path as training / pred.
    ir = IVIFDataset.gray_read(args.ir).unsqueeze(0).to(device)
    vi = IVIFDataset.color_read(args.vi).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.fuser(torch.cat([ir, vi], dim=1), return_aux=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_vi = out.get('alpha_vi')
    if alpha_vi is not None:
        with open(out_dir / 'router_weights.txt', 'w', encoding='utf-8') as f:
            f.write('Infrared: uncontrolled (no SRC / gate)\n')
            f.write('\nVisible token weights:\n')
            for name, w in zip(VI_DEGRADATION_NAMES, alpha_vi[0].cpu().tolist()):
                f.write(f'  {name}: {w:.4f}\n')

    save_image(out['fused'].clamp(0, 1), out_dir / 'fused.png')
    gate_vi = out.get('gate_vi')

    def _union(g):
        if g is None:
            return None
        if g.shape[1] > 1:
            return (1.0 - g[:, :1]).clamp(0, 1)
        return g

    u_vi = _union(gate_vi)
    if u_vi is not None:
        save_image(u_vi, out_dir / 'gate_vi.png')
    if gate_vi is not None and gate_vi.shape[1] > 1:
        for k, name in enumerate(VI_DEGRADATION_NAMES):
            save_image(gate_vi[:, k:k + 1], out_dir / f'gate_vi_{name}.png')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if u_vi is not None:
        axes[0].imshow(u_vi[0, 0].cpu(), cmap='hot')
    axes[0].set_title('VI gate' if u_vi is not None else 'VI gate N/A')
    axes[0].axis('off')
    axes[1].imshow(out['fused'][0, 0].cpu().clamp(0, 1), cmap='gray')
    axes[1].set_title('Fused')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'overview.png', dpi=150)
    print(f'Saved visualizations to {out_dir}')


if __name__ == '__main__':
    main()
