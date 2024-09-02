
import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch

from lightning.auto_rf import AutoRF
from lightning.reco import ReCo
import cv2
import numpy as np
from torch import nn
from time import time

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SaveFigure(Callback):
    def __init__(self, dst: str | Path, original_images: dict):
        super().__init__()
        self.dst = Path(dst)
        self.dst.mkdir(parents=True, exist_ok=True)
        self.original_images = original_images  # Save a reference to the original image

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Record the time when the prediction batch started
        #self.batch_start_time = time()
        pass

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        name, fused_I = outputs
        original_image = self.original_images[name[0]]  # Get the original color image

        # Convert the original image to HSV color space
        original_ycrcb = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)
        
        # Replace the fused I channel into the HSV image
        fused_I_numpy = np.array(fused_I.cpu())

        # Normalization: Scale values ​​to the range 0-1
        normalized_fused_I = (fused_I_numpy - fused_I_numpy.min()) / (fused_I_numpy.max() - fused_I_numpy.min())

        # Convert to 0-255 range
        scaled_fused_I = normalized_fused_I * 255

        # convert to uint8 type
        fused_I_uint8 = scaled_fused_I.astype(np.uint8)
        original_ycrcb[:, :, 0] = fused_I_uint8
        # Convert ycrcb image back to BGR format
        fused_color = cv2.cvtColor(original_ycrcb, cv2.COLOR_YCrCb2BGR)
        # Save the color fused image
        cv2.imwrite(str(self.dst / name[0]), fused_color)

        #time_elapsed = time() - self.batch_start_time
        #print(f"Batch {batch_idx} prediction time: {time_elapsed} seconds")

        # record gpu memory
        #self.log_gpu_memory(trainer)


    def log_gpu_memory(self, trainer):
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            #print(f"Current GPU Memory Used: {memory_used} MB")
            if trainer.logger:
                trainer.logger.log_metrics({'GPU Memory (MB)': memory_used})



def main():
    # args parser
    parser = ArgumentParser()

    # program level args
    # lightning
    parser.add_argument('--ckpt', type=str, default='../weights/default-f.ckpt', help='checkpoint path')
    # auto rf
    parser.add_argument('--data', type=str, default='/data/ykx/llvip_test', help='input data folder')
    parser.add_argument('--deform', type=str, default='none', help='random adjust level')
    # reco
    parser.add_argument('--dst', type=str, default='./result/llvip', help='output save folder')
    # cuda
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda (for cpu and out of memory)')
    parser.add_argument('--text', type=str, default='This is an infrared and visible image fusion task.', help='prompts to help identify the task')
    parser.add_argument('--text_vi', type=str, default='Visible light images contain a lot of texture details, but they are easily obscured by smoke.', help='prompts to help identify the degradation in visible images')
    parser.add_argument('--text_ir', type=str, default='Infrared images lack texture details but contain important salient target information', help='prompts to help identify the degradation in infrared images')
    parser.add_argument('--device', type=str, default=torch.device('cuda'))

    # model specific args
    parser = ReCo.add_model_specific_args(parser)

    # parse
    args = parser.parse_args()

    # fix seed
    pl.seed_everything(443)

    # model
    reco = ReCo(args)
    #print(f"The model has {count_parameters(reco)} trainable parameters")

    # dataloader
    dataset = AutoRF(root=args.data, mode='pred', level=args.deform)
    loader = DataLoader(dataset, collate_fn=AutoRF.collate_fn, num_workers=63)

    original_images = {}
    for img_name in os.listdir(os.path.join(args.data, 'vi')):
        img_path = os.path.join(os.path.join(args.data, 'vi'), img_name)
        original_images[img_name] = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #print(original_images)

    # callbacks
    callbacks = [SaveFigure(dst=args.dst, original_images=original_images)]
    #callbacks = [SaveFigure(dst=args.dst)]


    # lightning
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    accelerator, devices, strategy = ('cpu', None, None) if args.no_cuda else ('gpu', -1, strategy)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, callbacks=callbacks, strategy=strategy)
    trainer.predict(model=reco, dataloaders=loader, ckpt_path=args.ckpt)


if __name__ == '__main__':
    
    torch.set_float32_matmul_precision('medium')
    main()
