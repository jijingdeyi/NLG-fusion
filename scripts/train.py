from argparse import ArgumentParser
import logging
import sys
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldt.dataset import IVIFDataset
from ldt.defaults import (
    DEFAULT_CROP_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_VAL_SAMPLES,
)
from ldt.ldt_fusion import LDTFusion
from ldt.run_utils import build_run_name, snapshot_run_config


class LogValImages(Callback):
    """One val panel per epoch: IR | VI | fused (2 darker crops)."""

    def __init__(self, num_samples: int = 2):
        super().__init__()
        self.num_samples = num_samples

    @staticmethod
    def _to_gray01(img: Tensor) -> Tensor:
        x = img.detach().float().cpu()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x[:1].clamp(0, 1).expand(3, -1, -1).contiguous()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0 or outputs is None or trainer.logger is None:
            return
        try:
            ir, vi, fused = outputs['ir'], outputs['vi'], outputs['fused']
            means = vi[:, 0].flatten(1).mean(dim=1)
            idxs = torch.argsort(means)[: min(self.num_samples, ir.shape[0])].tolist()
            tiles = []
            for i in idxs:
                tiles.extend([self._to_gray01(ir[i]), self._to_gray01(vi[i]), self._to_gray01(fused[i])])
            grid = torchvision.utils.make_grid(tiles, nrow=3, padding=2, pad_value=1.0)
            trainer.logger.experiment.add_image('val/ir_vi_fused', grid, trainer.current_epoch)
        except Exception as exc:  # noqa: BLE001
            print(f'[LogValImages] skipped: {exc}')


def main():
    parser = ArgumentParser(description='Train IR–VI fusion backbone (no token/SRC).')
    parser.add_argument('--data', type=str, default='/data/ykx/LLVIP/train')
    parser.add_argument('--ckpt', type=str, default='checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--val_bs', type=int, default=4)
    parser.add_argument('--val_samples', type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument('--epochs', type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument('--seed', type=int, default=443)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser = LDTFusion.add_model_specific_args(parser)
    args = parser.parse_args()
    args.device = 'cpu' if args.no_cuda else 'cuda:0'
    args.no_synth = True

    run_name = build_run_name(args)
    run_ckpt_dir = Path(args.ckpt) / run_name
    config_path = snapshot_run_config(args, run_ckpt_dir)
    print(f'[train] run_name={run_name}')
    print(f'[train] checkpoints -> {run_ckpt_dir.resolve()}')
    print(f'[train] config snapshot -> {config_path.resolve()}')
    print('[train] backbone-only; TB under logs/fusion-backbone/')

    if not args.verbose:
        logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
        logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)

    pl.seed_everything(args.seed)
    model = LDTFusion(args)

    train_loader = DataLoader(
        IVIFDataset(
            args.data, 'train', crop_size=args.crop_size,
            vi_synth=False, val_samples=args.val_samples,
        ),
        batch_size=args.bs, shuffle=True,
        collate_fn=IVIFDataset.collate_fn, num_workers=4, pin_memory=not args.no_cuda,
    )
    val_loader = DataLoader(
        IVIFDataset(
            args.data, 'val', crop_size=args.crop_size,
            vi_synth=False, val_samples=args.val_samples,
        ),
        batch_size=args.val_bs,
        collate_fn=IVIFDataset.collate_fn, num_workers=4, pin_memory=not args.no_cuda,
    )

    logger = TensorBoardLogger(
        save_dir=args.log_dir, name='fusion-backbone', version=run_name,
        default_hp_metric=False,
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_ckpt_dir),
            filename=f'{run_name}-best-e{{epoch:03d}}',
            monitor='val/score',
            mode='max',
            save_top_k=1,
        ),
        EarlyStopping(
            monitor='val/score', mode='max', patience=args.patience, verbose=True,
            check_finite=True,
        ),
        LogValImages(num_samples=2),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        strategy='auto',
        log_every_n_steps=50,
        enable_model_summary=args.verbose,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
