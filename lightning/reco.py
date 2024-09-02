from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from kornia.filters import canny
from kornia.losses import ssim_loss
from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.functions.integrate import integrate
from modules.functions.transformer import transformer
from modules.fuser import Fuser
from modules.m_register import MRegister
from modules.u_register import URegister
import clip


class ReCo(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.dim = args.dim

        # init register
        match args.register:
            case 'm':
                print('Init with m-register')
                self.register = MRegister(in_c=4, dim=16, k_sizes=(3, 5, 7), use_bn=False, use_rs=True)
            case 'u':
                print('Init with u-register')
                self.register = URegister(in_c=4, dim=16)
            case _:
                #print('Turn off register')
                self.register = None

        # text preprocess
        self.model_clip, _ = clip.load("ViT-B/32")
        self.model_clip.eval()
        self.text_embed = clip.tokenize(args.text).to(args.device)
        self.text_embed_ir = clip.tokenize(args.text_ir).to(args.device)
        self.text_embed_vi = clip.tokenize(args.text_vi).to(args.device)

        # init fuser
        self.fuser = Fuser(depth=1, dim=self.dim, use_bn=False, model_clip=self.model_clip) # small batch size, no batch norm

        # learning rate
        self.lr = args.lr

        # specify weight
        self.rf_weight = [0.4, 0.6]
        self.r_weight, self.f_weight = args.r_weight, [0.6, 0.3, 0.1] # intensity, structure similarity, texture loss

    def training_step(self, batch, batch_idx):
        # infrared: [b, 1, h, w]
        # visible: [b, 1, h, w]
        x, y = batch['ir'], batch['vi']

        # register (optional): infrared (fixed) & visible (moving) -> y_m (moved)
        _, y_e = canny(y)
        if self.register is not None:
            y_t = batch['vi_t']
            y_m, locs_pred, y_m_e = self.r_forward(moving=y_t, fixed=x)
        else:
            y_m, locs_pred, y_m_e = y, 0, y_e

        # fuser: infrared (infrared) & visible (visible) -> f (fusion)
        f = self.f_forward(ir=x, vi=y_m, text_embed=self.text_embed, text_embed_ir=self.text_embed_ir, text_embed_vi=self.text_embed_vi)

        # register loss (optional):
        if self.register is not None:
            # image loss: y_m_e (edges of moved) -> y (edges of visible)
            img_loss = mse_loss(y_m_e, y_e)
            self.log('reg/img', img_loss)
            # locs loss: locs_pred -> locs_gt
            locs_gt = batch['locs_gt']
            locs_loss = mse_loss(locs_pred, locs_gt)
            self.log('reg/locs', locs_loss)
            # smooth loss: y_m_e (edges of moved) smooth
            dx = torch.abs(y_m_e[:, :, 1:, :] - y_m_e[:, :, :-1, :])
            dy = torch.abs(y_m_e[:, :, :, 1:] - y_m_e[:, :, :, :-1])
            smo_loss = (torch.mean(dx * dx) + torch.mean(dy * dy)) / 2
            self.log('reg/smooth', smo_loss)
            reg_loss = img_loss * self.r_weight[0] + locs_loss * self.r_weight[1] + smo_loss * self.r_weight[2]
            self.log('train/reg', reg_loss)
        else:
            reg_loss = 0


        # fuse loss with iqa (if iqa is disabled, x_w = y_w = 1)
        x_ssim = ssim_loss(f, x, window_size=11, reduction='none')
        y_ssim = ssim_loss(f, y, window_size=11, reduction='none')
        sim_loss = x_ssim * 0.6 + y_ssim * 0.4
        self.log('fus/ssim', sim_loss.mean())

        # intensity loss
        x_in_max = torch.max(x, y)
        int_loss = l1_loss(f, x_in_max)
        self.log('fus/mse', int_loss.mean())

        # texture loss
        grad_loss = gradloss()
        tex_loss = grad_loss(y, x, f)
        self.log('fus/tex', tex_loss.mean())

        # fusion loss
        fus_loss = self.f_weight[0] * int_loss + self.f_weight[1] * sim_loss + self.f_weight[2] * tex_loss
        fus_loss = fus_loss.mean()
        self.log('train/fus', fus_loss)

        # final loss
        fin_loss = self.rf_weight[0] * reg_loss + self.rf_weight[1] * fus_loss
        self.log('train/fin', fin_loss)

        return fin_loss

    def validation_step(self, batch, batch_idx):
        # infrared & visible: [b, 1, h, w]
        x, y = batch['ir'], batch['vi']

        # output
        o = [x, y]

        # register (optional): infrared (fixed) & visible (moving) -> y_m (moved)
        if self.register is not None:
            y_t = batch['vi_t']
            y_m, _, _ = self.r_forward(moving=y_t, fixed=x)
            o += [y_t, y_m, y_m - y_t]
        else:
            y_m = y

        # fuser: ir (infrared) & vi (visible) -> f (fusion)
        f = self.f_forward(ir=x, vi=y_m, text_embed=self.text_embed, text_embed_ir=self.text_embed_ir, text_embed_vi=self.text_embed_vi)
        o += [f]

        # output
        o = torch.cat(o, dim=1)
        return o

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> list[str, Tensor]:
        
        # infrared & visible (moving): [b, 1, h, w]
        x, y_t = batch['ir'], batch['vi']

        # register (optional): infrared (fixed) & visible (moving) -> y_m (moved)
        if self.register is not None:
            y_m, _, _ = self.r_forward(moving=y_t, fixed=x)
        else:
            y_m = y_t

        # fuser: infrared (infrared) & visible (visible) -> f (fusion)
        f = self.f_forward(ir=x, vi=y_m, text_embed=self.text_embed, text_embed_ir=self.text_embed_ir, text_embed_vi=self.text_embed_vi)

        # output
        return batch['name'], f

    def r_forward(self, moving: Tensor, fixed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # moving & fixed: [b, 1, h, w]
        # pred: moving & grid -> moved
        # apply transform
        moving_m, moving_e = canny(moving)
        fixed_m, fixed_e = canny(fixed)
        # predict flow
        flow = self.register(torch.cat([moving, fixed, moving_m, fixed_m], dim=1))
        flow = integrate(n_step=7, flow=flow)
        moved, locs = transformer(moving, flow)
        moved_e, locs = transformer(moving_e, flow)
        return moved, locs, moved_e

    def f_forward(self, ir: Tensor, vi: Tensor, text_embed: Tensor, text_embed_ir: Tensor, text_embed_vi: Tensor) -> Tensor:
        # ir: [b, 1, h, w]
        # vi: [b, 1, h, w]
        # pred: ir (infrared) & vi (visible) -> f (fusion)
        f = self.fuser(torch.cat([ir, vi],  dim=1), text_embed, text_embed_ir, text_embed_vi)
        return f

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], {'scheduler': scheduler, 'monitor': 'train/fin'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ReCo+')
        # reco
        parser.add_argument('--register', type=str, default='x', help='register (m: micro, u: u-net, x: none)')
        parser.add_argument('--dim', type=int, default=32, help='dimension in backbone (default: 16)')
        # optimizer
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
        # weights
        parser.add_argument('--rf_weight', nargs='+', type=float, help='balance in register & fuse')
        parser.add_argument('--r_weight', nargs='+', type=float, help='balance in register: img, locs, smooth')
        parser.add_argument('--f_weight', nargs='+', type=float, help='balance in fuse: ssim, l1')

        return parent_parser
    

class gradloss(nn.Module):
    def __init__(self):
        super(gradloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint, generate_img_grad, reduction='none')
        return loss_grad

# Sobel filter, image edge detection
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)