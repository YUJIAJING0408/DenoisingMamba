import cv2
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from models.loss.loss import L1Loss, L2Loss, MAPELoss, SMAPELoss, SSIMLoss, MSSSIMLoss, MixLoss, GradientLoss, PSRNLoss
from .mfp.vip import VimOutPic
import lightning as l
import numpy as np
from utils.metric import calculate_psnr, calculate_rmse, calculate_ssim, calculate_psnr_255,calculate_ssim_255
from utils.preprocessing import *
from .ffcs.ffc import FFC


class DenoisingMambaModel(l.LightningModule):
    def __init__(self, img_size, loss_weights, epochs, lrMilestone, lr, save_img_time, mix_rule=None,if_bidirectional=True,if_abs_pos_embed=False):
        super(DenoisingMambaModel, self).__init__()
        self.lrMilestone = lrMilestone
        self.epochs = epochs
        self.img_size = img_size
        self.scheduler = None
        self.optimizer = None
        self.save_img_time = save_img_time if save_img_time >= 0 else 10
        self.lr = lr
        self.model = DenoisingMamba(
            img_size=self.img_size,
            msfe_mid_channels=32,  # 32
            msfe_out_channels=8,
            m_block_num=3,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            if_bidirectional=if_bidirectional,
            if_abs_pos_embed=if_abs_pos_embed,
            mix_rule=mix_rule,
        )
        self.need_dispose_specular = True
        self.save_hyperparameters()  # 保存超参
        self.loss_weights = loss_weights

        def _init_weight(submodule):
            if isinstance(submodule, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(submodule.weight)

        # 二维的拉普拉斯-高斯（Laplacian of Gaussian, LoG）滤波器
        x = np.linspace(-4, 4, 9)
        y = np.linspace(-4, 4, 9)
        x_grid, y_grid = np.meshgrid(x, y)
        self.LoG_filter = torch.tensor(self._LoG(x_grid, y_grid, 1.5),
                                       dtype=torch.float32,
                                       requires_grad=False).repeat(3, 3, 1, 1)
        self.model.apply(_init_weight)

        # loss_type = "l1_grad"
        #
        # if loss_type == 'l1':
        #     self.loss = L1Loss()
        # elif loss_type == 'l2':
        #     self.loss = L2Loss()
        # elif loss_type == 'mape':
        #     self.loss = MAPELoss()
        # elif loss_type == 'smape':
        #     self.loss = SMAPELoss()
        # elif loss_type == 'ssim':
        #     self.loss = SSIMLoss(num_channels=3)
        # elif loss_type == 'msssim':
        #     self.loss = MSSSIMLoss(num_channels=3)
        # elif loss_type == 'l1_msssim':
        #     # [Zhao et al., 2018, "Loss Functions for Image Restoration with Neural Networks"]
        #     msssim_loss = MSSSIMLoss(num_channels=3)
        #     self.loss = MixLoss([L1Loss(), msssim_loss], [0.16, 0.84])
        # elif loss_type == 'l1_ssim_grad':
        #     self.loss = MixLoss([L1Loss(), SSIMLoss(num_channels=3),GradientLoss()], [0.65, 0.005,0.35])
        # elif loss_type == 'l1_grad':
        #     self.loss = MixLoss([L1Loss(), GradientLoss()], [0.5, 0.5])
        # else:
        #     raise NotImplementedError

        self.PSNRLoss = PSRNLoss(50)
        # self.MSSSIMLoss = MSSSIMLoss(num_channels=3)

    def configure_optimizers(self):
        # 存储学习率衰减的时间节点（epoch）
        milestones = [i * self.lrMilestone - 1 for i in range(1, self.epochs // self.lrMilestone)]
        print(milestones)
        # 定义生成器的优化器和学习率调度器
        self.optimizer = optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.002, betas=(0.9, 0.99))
        # l = lambda epoch: 1 / np.sqrt(epoch) if epoch != 0 else 1
        # # l = lambda epoch: 2.2 / np.sqrt(epoch / ( np.cos(epoch) + 2 )) if epoch >= 3 else 1.5
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[l])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }

    def loss(self, denoised, target):
        """Spatial loss"""
        # loss_s = F.huber_loss(denoised,target, reduction='mean',delta=0.001)
        loss_s = F.l1_loss(denoised, target, reduction='none')
        # loss_s 大小 (7,16,3,128,128) -> (时序长度,批次,通道,长,宽)
        """
        梯度域损失：Gradient-domain loss
        不是直接比较两个图像的像素值差异，而是比较它们梯度（即图像中像素强度的变化率）的差异。梯度域损失通常用于鼓励生成的图像或处理后的图像与参考图像在结构上相似，而不仅仅是在像素级别上相似。
        在深度学习中，使用梯度域损失可以有助于解决一些由于光照变化、颜色差异或纹理差异引起的问题，这些问题在仅使用像素级损失函数时可能难以解决。
        例如，在图像修复、风格迁移或超分辨率等任务中，梯度域损失可以帮助模型更好地捕捉图像的轮廓和结构信息。
        具体来说，梯度域损失通常通过计算输入图像和输出图像（或参考图像）在梯度空间中的差异来实现。这通常涉及到对图像进行梯度运算（如计算一阶或二阶导数），然后比较这些梯度图的差异。
        """
        LoG_filter = self.LoG_filter.to(self.device)
        denoised_LoG = []
        target_LoG = []
        for i in range(target.size(0)):
            denoised_LoG.append(F.conv2d(denoised[i], LoG_filter, padding='same'))
            target_LoG.append(F.conv2d(target[i], LoG_filter, padding='same'))
        denoised_LoG = torch.stack(denoised_LoG)
        target_LoG = torch.stack(target_LoG)
        loss_g = F.l1_loss(denoised_LoG, target_LoG, reduction='none')
        # loss_SMAPE = SMAPELoss()(target_LoG, denoised_LoG)
        total_loss = (self.loss_weights['loss_s'] * loss_s) \
                     + (self.loss_weights['loss_g'] * loss_g)
        loss_PSNR = self.PSNRLoss(denoised,target) # 1-32/36 =0.2 # 1 - 10/36
        # loss_MSSIM = self.MSSSIMLoss(denoised, target)
        # print(loss_MSSIM)
        # loss_SSIM = SSIMLoss().to("cuda")(denoised, target)
        # print(loss_SSIM)
        total_loss = torch.mean(total_loss, dim=[0, 1, 2, 3]) + (0.005 * loss_PSNR) # 0.005 * loss_PSNR

        print(total_loss.mean())
        return total_loss.mean()


    def forward(self, x):
        # print(x.shape)
        x = self.model.forward(x)
        return x

    def training_step(self, batch, batch_idx):
        gt = batch[:, :3, :, :]
        noisy_with_aux_features = batch[:, 3:, :, :]
        # # 噪声高光预处理
        # if self.need_dispose_specular:
        #     noisy_with_aux_features[:, 0:3, :, :] = preprocess_specular_t(noisy_with_aux_features[:, 0:3, :, :])
        denoise = self.forward(noisy_with_aux_features)
        # if self.need_dispose_specular:
        #     # 噪声高光还原
        #     denoise = postprocess_specular_t(denoise)
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and self.trainer.current_epoch > 10:
            sch.step()
        if self.optimizer is not None:
            self.log('lr', self.optimizer.param_groups[0]['lr'])
        return self.loss(denoise, gt)

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def _shared_eval(self, batch, batch_idx, prefix):
        gt_ldr = batch[:, :3, :, :]
        noisy_with_aux_features = batch[:, 3:, :, :]
        # if self.need_dispose_specular:
        #     # 噪声高光预处理
        #     noisy_with_aux_features[:, 0:3, :, :] = preprocess_specular_t(noisy_with_aux_features[:, 0:3, :, :])
        denoise_ldr = self.forward(noisy_with_aux_features)
        # if self.need_dispose_specular:
        #     # 噪声高光还原
        #     denoise_ldr = postprocess_specular_t(denoise_ldr)
        # 转为np
        denoise_ldr_np =denoise_ldr.detach().cpu().numpy()
        gt_ldr_np = gt_ldr.detach().cpu().numpy()
        loss = self.loss(denoise_ldr, gt_ldr)
        self.log(f'{prefix}_loss', loss)
        # LDR->HDR(无损Reinhard)
        denoise_hdr_np = restore_reinhard(denoise_ldr_np)
        gt_hdr_np = restore_reinhard(gt_ldr_np)
        # HDR->LDR(有损Tone map映射)
        denoise_tonemap_np = tonemap(denoise_hdr_np)
        denoise_tonemap_255_np = (denoise_tonemap_np * 255.0).astype(np.uint8)
        gt_tonemap_np = tonemap(gt_hdr_np)
        gt_tonemap_255_np = (gt_tonemap_np * 255.0).astype(np.uint8)
        total_psnr = 0.0
        total_rmse = 0.0
        total_ssim = 0.0
        for i in range(denoise_ldr_np.shape[0]):
            psnr = calculate_psnr_255(denoise_tonemap_255_np[i], gt_tonemap_255_np[i])
            if psnr is not None:
                total_psnr += psnr
            else:
                assert "PSNR求值错误"
            rmse = calculate_rmse(denoise_tonemap_np[i], gt_tonemap_np[i])
            if rmse is not None:
                total_rmse += rmse
            else:
                assert "RMSE求值错误"

            ssim = calculate_ssim_255(denoise_tonemap_255_np[i], gt_tonemap_255_np[i])
            # print("**************")
            # print(f"denoise-255: {denoise_ldr_255_np.max()},{denoise_ldr_255_np.min()}")
            # print(f"gt-255: {gt_ldr_255_np.max()},{gt_ldr_255_np.min()}")
            # print(np.any(np.isnan(denoise_ldr_255_np)))
            # print(np.any(np.isnan(gt_ldr_255_np)))
            # print(ssim)
            if ssim is not None:
                # print(ssim)
                total_ssim += ssim
            else:
                assert "SSIM求值错误"
        total = denoise_hdr_np.shape[0]
        self.log(f'{prefix}_psnr', total_psnr / total)
        self.log(f'{prefix}_rmse', total_rmse / total)
        self.log(f'{prefix}_ssim', total_ssim / total)
        if prefix == 'val' and self.trainer.current_epoch % self.save_img_time == 0:
            # 保存图片
            batch_size = batch.shape[0]
            # 通过tonemap后变换为[0,1.0]or[0,255]
            noisy_img = torchvision.utils.make_grid(tonemap_t( restore_reinhard_t(batch[:, 3:6, :, :])), nrow=batch_size, padding=0,value_range=(0, 1))
            denoise_img = torchvision.utils.make_grid(tonemap_t( restore_reinhard_t(denoise_ldr)), nrow=batch_size, padding=0,value_range=(0, 1))
            gt_img = torchvision.utils.make_grid( tonemap_t(restore_reinhard_t(gt_ldr)), nrow=batch_size, padding=0,value_range=(0, 1))
            # # 多张图片组合输出
            grid_mix = torchvision.utils.make_grid(torch.cat((gt_img, noisy_img, denoise_img), dim=1), nrow=1)
            # 保存到本地文件夹
            # grid_mix_np = grid_mix.detach().permute(1, 2, 0).cpu().numpy()
            # cv2.imwrite(f"/media/yujiajing0408/Data/MD_logs/JPG_logs/{self.trainer.current_epoch}-{self.trainer.global_step}.jpg", (grid_mix_np[:,:,::-1]*255).astype(np.uint8))
            self.logger.experiment.add_image('mix', grid_mix, self.trainer.current_epoch)
            pass

    def _LoG(self, x, y, sigma):
        return -1.0 / (np.pi * sigma ** 4) \
            * (1.0 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) \
            * np.e ** (-(x ** 2 + y ** 2) / (2 * sigma ** 2))


class MB(nn.Module):
    def __init__(self, img_size=120, patch_size=3, stride=3, in_channels=13, mid_channels=256, depth=24, drop_rate=0.1,
                 drop_path_rate=0.1, device='cuda', if_bidirectional=False,if_abs_pos_embed=False):
        super(MB, self).__init__()
        self.mid_channels = mid_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.depth = depth
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.if_bidirectional = if_bidirectional
        self.m = VimOutPic(
            img_size=self.img_size,
            patch_size=self.patch_size,
            stride=self.stride,
            depth=self.depth,
            in_channels=self.in_channels,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            device=device,
            if_bidirectional=self.if_bidirectional,
            if_abs_pos_embed=if_abs_pos_embed
        )
        self.conv1 = conv_block(self.in_channels, self.mid_channels, kernel_size=3, padding=1, padding_mode='reflect',
                                act_type='leakyrelu')
        self.conv2 = conv_block(self.mid_channels, self.in_channels // 2, kernel_size=3, padding=1,
                                padding_mode='reflect',
                                act_type='leakyrelu')
        self.door = conv_block(self.in_channels, self.in_channels // 2, kernel_size=3, padding=1,
                               padding_mode='reflect',
                               act_type='leakyrelu')

    def forward(self, x):
        org_x = x.clone()
        x = self.m(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.door(org_x)
        return x

# 残差多向Mamba编码器（x 和 x.T）
class DenoisingMamba(nn.Module):
    def __init__(self, img_size, msfe_mid_channels, msfe_out_channels, out_channels=3, mix_rule=None, m_block_num=3,
                 if_bidirectional=True,if_abs_pos_embed=False, device='cuda'):
        super(DenoisingMamba, self).__init__()
        self.device = device
        self.if_bidirectional = if_bidirectional
        self.img_size = img_size
        self.m_block_num = m_block_num
        self.mix_rule = {
            'noisy': (0, 3),
            'aux': (3, 10)
        } if mix_rule is None else mix_rule
        self.aux_in_channels = self.mix_rule['aux'][1] - self.mix_rule['aux'][0]
        self.noisy_in_channels = self.mix_rule['noisy'][1] - self.mix_rule['noisy'][0]
        self.msfe_mid_channels = msfe_mid_channels
        self.msfe_out_channels = msfe_out_channels
        self.out_channels = out_channels
        self.msfe_ffc = FFCFeatureExtractor(aux_in_channels=self.aux_in_channels,
                                                   noisy_in_channels=3,
                                                   mid_channels=self.msfe_mid_channels,
                                                   out_channels=self.msfe_out_channels * 2)

        m_blocks = []
        # patch_size = [5,4,3]
        patch_size = [3,4,5] # 控制渐进内核大小3->4->5
        m_depths = [24, 24, 24, 24]
        for i in range(self.m_block_num):
            m_blocks.append(MB(
                img_size=self.img_size,
                patch_size=patch_size[i],
                stride=patch_size[i],
                in_channels=self.msfe_out_channels * 2,
                mid_channels=256,
                depth=m_depths[i],
                drop_rate=0.1,
                drop_path_rate=0.1,
                device=self.device,
                if_bidirectional=self.if_bidirectional,
                if_abs_pos_embed=if_abs_pos_embed
            ))
        self.m_blocks = nn.ModuleList(m_blocks)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.msfe_out_channels, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.PReLU(num_parameters=256, init=0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.PReLU(num_parameters=256, init=0.2),
            nn.Conv2d(256, 3, kernel_size=3, padding=1, padding_mode='zeros'),
        )

    def forward(self, x):
        noisy = x[:, self.mix_rule['noisy'][0]:self.mix_rule['noisy'][1], :, :]
        aux = x[:, self.mix_rule['aux'][0]:self.mix_rule['aux'][1], :, :]
        # print(aux.shape)
        noisy_features, aux_features = self.msfe_ffc(noisy=noisy, aux=aux)
        # 拼接
        for i in range(self.m_block_num):
            bak_features = noisy_features
            mix = torch.cat([noisy_features, aux_features], dim=1)
            mix_t = mix.permute(0, 1, 3, 2)
            noisy_features = self.m_blocks[i](mix)
            noisy_features_t = self.m_blocks[i](mix_t)
            noisy_features = (noisy_features + bak_features + noisy_features_t.permute(0, 1, 3, 2)) / 3
        denoise = self.decoder(noisy_features)
        return noisy + denoise


# 快速傅里叶卷积特征提取器FFCE
class FFCFeatureExtractor(nn.Module):
    def __init__(self, aux_in_channels, noisy_in_channels, mid_channels, out_channels):
        super(FFCFeatureExtractor, self).__init__()
        self.aux_in_channels = aux_in_channels
        self.noisy_in_channels = noisy_in_channels
        self.mid_channels = mid_channels
        self.noisy = conv_block(self.noisy_in_channels, self.mid_channels, kernel_size=3, padding=1, act_type='relu')
        self.aux = conv_block(self.aux_in_channels, self.mid_channels, kernel_size=3, padding=1, act_type='relu')
        self.ffc = FFC(in_channels=mid_channels * 2, out_channels=out_channels, ratio_gin=0.5, ratio_gout=0.5,
                       kernel_size=3,padding=1)

    def forward(self, noisy, aux):
        noisy_features = self.noisy(noisy)
        aux_features = self.aux(aux)
        return self.ffc((noisy_features, aux_features))


def norm(norm_type, out_ch):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(out_ch, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(out_ch, affine=False)
    else:
        raise NotImplementedError('Normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for lrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('Activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# conv norm activation
# 卷积块定义：包含一个卷积层 + 归一化层 + 激活层
def conv_block(in_ch, out_ch, kernel_size, stride=1, dilation=1, padding=0, padding_mode='zeros', norm_type=None,
               act_type='relu', groups=1, inplace=True):
    c = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding,
                  padding_mode=padding_mode, groups=groups)
    n = norm(norm_type, out_ch) if norm_type else None
    a = act(act_type, inplace) if act_type else None
    return sequential(c, n, a)


def test_dm_v3():
    in_channels = 13
    img_size = 120
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    x = torch.randn(1, in_channels, img_size, img_size).to(device)
    model = DenoisingMamba(
        img_size=img_size,
        msfe_mid_channels=64,
        msfe_out_channels=8,
        out_channels=3,
        m_block_num=3,
        if_bidirectional=True,
        device=device
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'DenoisingMamba总参数:{total_params / 1e6}M')
    if torch.cuda.is_available():
        print(f"Pytorch分配给GPU的显存开销: {torch.cuda.memory_allocated() // 1048576}MB")
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    test_dm_v3()
