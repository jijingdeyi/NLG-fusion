from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader

from lightning.auto_rf import AutoRF
from lightning.reco import ReCo
from utils.pretty_vars import pretty_vars
from time import time


class LogImageCallback(Callback):
    def __init__(self, logger: WandbLogger, show_grad: bool = False):
        super().__init__()
        self.logger = logger
        self.show_grad = show_grad

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass
        #self.log_gpu_memory()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            assert isinstance(outputs, Tensor)
            size = outputs.shape[1]
            imgs = [torch.clip(img[-1].squeeze(), min=0, max=1) for img in torch.chunk(outputs, chunks=size, dim=1)]
            captions = ['ir', 'vi', 'vi_t', 'vi_m', 'dif', 'f'] if size == 6 else ['ir', 'vi', 'f']
            self.logger.log_image(key='sample', images=imgs, caption=captions)
 
        #self.log_gpu_memory()

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if self.show_grad:
            print(pretty_vars(pl_module.register))

    
    def log_gpu_memory(self):
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            self.logger.log_metrics({'GPU Memory (MB)': memory_used})


def main():
    # args parser
    parser = ArgumentParser()

    # program level args
    # lightning
    parser.add_argument('--ckpt', type=str, default='checkpoint', help='checkpoints save folder')
    parser.add_argument('--show_grad', action='store_true', help='show grad before zero_grad')
    parser.add_argument('--seed', type=int, default=443, help='seed for random number')
    # wandb
    parser.add_argument('--key', type=str, help='wandb auth key')
    # auto rf
    parser.add_argument('--data', type=str, default='../data/tno', help='input data folder')
    parser.add_argument('--deform', type=str, default='none', help='random adjust level')
    # loader
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    # cuda
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda (for cpu and out of memory)')
    parser.add_argument('--text', type=str, default='This is an infrared and visible image fusion task.', help='prompts to help identify the task')
    parser.add_argument('--text_vi', type=str, default='low light degradation', help='prompts to help identify the degradation in visible images')
    parser.add_argument('--text_ir', type=str, default='low contrast and blurred', help='prompts to help identify the degradation in infrared images')
    parser.add_argument('--device', type=str, default=torch.device('cuda'))

    # model specific args
    parser = ReCo.add_model_specific_args(parser)

    # parse
    args = parser.parse_args()

    # fix seed
    pl.seed_everything(args.seed)

    # model
    reco = ReCo(args)


    # dataloader
    train_dataset = AutoRF(root=args.data, mode='train', level=args.deform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs,
        shuffle=True, collate_fn=AutoRF.collate_fn, num_workers=4,
    )
    val_dataset = AutoRF(root=args.data, mode='val', level=args.deform)
    val_loader = DataLoader(
        val_dataset, batch_size=args.bs,
        collate_fn=AutoRF.collate_fn, num_workers=4,
    )

    # logger
    wandb.login(key=args.key)
    logger = WandbLogger(project='ours')

    #total_params = sum(p.numel() for p in reco.parameters())
    #logger.log_metrics({'Total Parameters': total_params})

    

    # callbacks
    callbacks = [
        
        ModelCheckpoint(dirpath=args.ckpt, filename='ours', every_n_train_steps=10),
        LogImageCallback(logger=logger, show_grad=args.show_grad),
        LearningRateMonitor(logging_interval='step')
    ]

    # lightning
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    accelerator, devices, strategy = ('cpu', None, None) if args.no_cuda else ('gpu', -1, strategy)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        max_epochs= 30,
        strategy=strategy,
        log_every_n_steps=5,
    )
    trainer.fit(model=reco, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    main()
