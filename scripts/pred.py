import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldt.dataset import IVIFDataset
from ldt.ldt_fusion import LDTFusion
from ldt.run_utils import merge_inference_args_from_checkpoint


class SaveFigure(Callback):
    def __init__(self, dst: str | Path, original_images: dict):
        super().__init__()
        self.dst = Path(dst)
        self.dst.mkdir(parents=True, exist_ok=True)
        self.original_images = original_images

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        name, fused_i = outputs
        key = name[0]
        original = self.original_images.get(key)
        if original is None:
            raise FileNotFoundError(f'missing original VI for {key}')
        ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
        fused = fused_i.squeeze().detach().cpu().numpy()
        fused = (np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)
        ycrcb[:, :, 0] = fused
        cv2.imwrite(str(self.dst / key), cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR))


def main():
    parser = ArgumentParser(description='Fusion backbone inference.')
    parser.add_argument(
        '--ckpt', type=str, required=True,
        help='Lightning checkpoint (backbone-only or legacy; control heads ignored)',
    )
    parser.add_argument('--data', type=str, default='/data/ykx/MSRS/detection')
    parser.add_argument('--dst', type=str, default='result/pred')
    parser.add_argument('--no_cuda', action='store_true')
    parser = LDTFusion.add_model_specific_args(parser)
    cli = parser.parse_args()
    cli.device = 'cpu' if cli.no_cuda else 'cuda:0'

    args = merge_inference_args_from_checkpoint(cli.ckpt, cli)

    pl.seed_everything(443)
    # Legacy full-LDT ckpts have extra keys; load backbone weights loosely then attach.
    try:
        model = LDTFusion.load_from_checkpoint(
            args.ckpt, args=args, map_location='cpu', strict=False,
        )
    except Exception as exc:
        print(f'[pred] load_from_checkpoint failed ({exc}); building fresh + backbone weights')
        from modules.fuser import load_backbone_from_checkpoint
        model = LDTFusion(args)
        model.fuser = load_backbone_from_checkpoint(args.ckpt, freeze=False)

    dataset = IVIFDataset(args.data, 'pred')
    loader = DataLoader(dataset, collate_fn=IVIFDataset.collate_fn, num_workers=4)

    vi_dir = Path(args.data) / 'vi'
    original_images = {}
    for name in dataset.samples:
        path = vi_dir / name
        original_images[name] = cv2.imread(str(path), cv2.IMREAD_COLOR)

    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[SaveFigure(args.dst, original_images)],
    )
    trainer.predict(model, loader)


if __name__ == '__main__':
    main()
