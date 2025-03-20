# L1 loss (seems to be faster than the built-in L1Loss)
import torch
from torch import nn

from .ssim import SSIM, MS_SSIM


class L1Loss(nn.Module):
    def forward(self, input, target):
        l = torch.abs(input - target).mean()
        print(l)
        return l

# L2 (MSE) loss
class L2Loss(nn.Module):
    def forward(self, input, target):
        return ((input - target) ** 2).mean()

# MAPE (relative L1) loss
class MAPELoss(nn.Module):
    def forward(self, input, target):
        return (torch.abs(input - target) / (torch.abs(target) + 1e-2)).mean()

# SMAPE (symmetric MAPE) loss
class SMAPELoss(nn.Module):
    def forward(self, input, target):
        return (torch.abs(input - target) / (torch.abs(input) + torch.abs(target) + 1e-2)).mean()

# SSIM loss
class SSIMLoss(nn.Module):
    def __init__(self, num_channels=3):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1., channel=num_channels)

    def forward(self, input, target):
        with torch.autocast('cuda', enabled=False):
            return 1. - self.ssim(input.float(), target.float())

# MS-SSIM loss
class MSSSIMLoss(nn.Module):
    def __init__(self, num_channels=3, weights=None):
        super(MSSSIMLoss, self).__init__()
        self.msssim = MS_SSIM(data_range=1., channel=num_channels, weights=weights)

    def forward(self, input, target):
        with torch.autocast('cuda', enabled=False):
            return (1. - self.msssim(input.float(), target.float())).mean()

class PSRNLoss(nn.Module):
    def __init__(self, target_psnr=36):
        super(PSRNLoss, self).__init__()
        self.target_psnr = target_psnr

    def forward(self, input, target):
        if len(input.shape) == 4:
            loss = []
            for i in range(input.shape[0]):
                # 归一化
                # print(target[i].max() - target[i].min())
                # if torch.isnan((target[i].max() - target[i].min())) or (target[i].max() - target[i].min()) == 0 :
                #     print(target[i].max(),target[i].min())
                input_ = (input[i]-target[i].min()) / (target[i].max() - target[i].min() + 1e-8)
                target_ = (target[i]-target[i].min()) / (target[i].max() - target[i].min() + 1e-8)
                mse = torch.mean((input_ - target_) ** 2)
                psnr =  20 * torch.log10(1.0 / mse)
                # print(input_.mean(), target_.mean(), mse,psnr)
                loss.append(1.0 - psnr / self.target_psnr)
                # if psnr < self.target_psnr:
                #     loss.append(1.0 - psnr/self.target_psnr)
                # else:
                #     loss.append(torch.full(psnr.shape, 1e-8, device=psnr.device))
            return torch.stack(loss,dim=0).mean()
        else:
            input_ = (input - target.min()) / (target.max() - target.min())
            target_ = (target - target.min()) / (target.max() - target.min())
            mse = torch.mean((input_ - target_) ** 2)
            psnr = 20 * torch.log10(1.0 / mse).mean()
            return (1.0 - psnr / self.target_psnr).mean()

# Computes gradient for a tensor
def tensor_gradient(input):
  input0 = input[..., :-1, :-1]
  didy   = input[..., 1:,  :-1] - input0
  didx   = input[..., :-1, 1:]  - input0
  return torch.cat((didy, didx), -3)

class GradientLoss(nn.Module):
    def forward(self, input, target):
        l = torch.abs(tensor_gradient(input) - tensor_gradient(target)).mean()
        return l

# Mix loss
class MixLoss(nn.Module):
    def __init__(self, losses, weights):
        super(MixLoss, self).__init__()
        self.losses = nn.Sequential(*losses)
        self.weights = weights

    def forward(self, input, target):
        return sum([l(input, target) * w for l, w in zip(self.losses, self.weights)])

class RMSELoss(nn.Module):
    def __init__(self,eps = 1e-3):
        super(RMSELoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        mse = torch.pow((input - target),2)
        return (mse / (torch.pow(input,2) + self.eps)).mean()