##
# @Desp：DEMC的Pytorch实现
# @Auth：YJJ
# @Date：2024-06-17
# #
import math
import lightning as l
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from models.loss.loss import RMSELoss
from utils.metric import calculate_psnr, calculate_rmse, calculate_ssim, calculate_psnr_255, calculate_ssim_255
from utils.preprocessing import tonemap, tonemap_t, restore_reinhard, restore_reinhard_t


class DEMCModel(l.LightningModule):
    def __init__(self, lrMilestone, epochs, img_size, aux_in_channels, aux_out_channels, base_channels, save_img_time,
                 lr,loss_weights):
        super(DEMCModel, self).__init__()
        self.aux_out_channels = aux_out_channels
        self.img_size = img_size
        self.lrMilestone = lrMilestone
        self.epochs = epochs
        self.scheduler = None
        self.optimizer = None
        self.save_img_time = save_img_time if save_img_time >= 0 else 10
        self.lr = lr
        self.loss_weights = loss_weights
        self.save_hyperparameters()  # 保存超参
        self.features_encoder_layers = int(math.log2(img_size)) - 2
        print(self.features_encoder_layers)
        self.hdr_encoder_layers = self.features_encoder_layers + 1
        self.demc = DEMC(aux_in_channels=aux_in_channels,
                         aux_out_channels=aux_out_channels,
                         features_encoder_layers=self.features_encoder_layers,
                         hdr_encoder_layers=self.hdr_encoder_layers,
                         base_channels=base_channels)

        # def _initialize_weights(submodule):
        #     if isinstance(submodule, nn.Conv2d):
        #         torch.nn.init.xavier_uniform_(submodule.weight)

        # 二维的拉普拉斯-高斯（Laplacian of Gaussian, LoG）滤波器
        x = np.linspace(-4, 4, 9)
        y = np.linspace(-4, 4, 9)
        x_grid, y_grid = np.meshgrid(x, y)
        self.LoG_filter = torch.tensor(self._LoG(x_grid, y_grid, 1.5),
                                       dtype=torch.float32,
                                       requires_grad=False).repeat(3, 3, 1, 1)
        # self.demc.apply(_initialize_weights)
        # self.loss = RMSELoss()

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
        # loss_PSNR = self.PSNRLoss(denoised,target) # 1-32/36 =0.2 # 1 - 10/36
        # loss_MSSIM = self.MSSSIMLoss(denoised, target)
        # print(loss_MSSIM)
        total_loss = torch.mean(total_loss, dim=[0, 1, 2, 3])
                      # + (0.005 * loss_PSNR)
        print(total_loss.mean())
        return total_loss.mean()

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

    def forward(self, noisy,aux_features):
        # print(x.shape)
        x = self.demc.forward(noisy=noisy,aux=aux_features)
        return x

    def training_step(self, batch, batch_idx):
        gt_hdr = batch[:, :3, :, :]
        noisy_hdr = batch[:, 3:6, :, :]
        aux_features = batch[:, 6:, :, :]
        denoise_hdr = self.forward(noisy=noisy_hdr,aux_features=aux_features)
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and self.trainer.current_epoch > 10:
            sch.step()
        if self.optimizer is not None:
            self.log('lr', self.optimizer.param_groups[0]['lr'])
        return self.loss(denoise_hdr, gt_hdr)

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def _shared_eval(self, batch, batch_idx, prefix):
        # gt_hdr = batch[:, :3, :, :]
        # noisy_hdr = batch[:, 3:6, :, :]
        # aux_features = batch[:, 6:, :, :]
        # denoise_hdr = self.forward(noisy=noisy_hdr, aux_features=aux_features)
        # denoise_hdr = self.forward(noisy=noisy_hdr, aux_features=aux_features)
        # loss = self.loss(denoise_hdr, gt_hdr)
        gt_ldr = batch[:, :3, :, :]
        noisy_ldr = batch[:, 3:6, :, :]
        aux_features = batch[:, 6:, :, :]
        denoise_ldr = self.forward(noisy=noisy_ldr, aux_features=aux_features)
        loss = self.loss(denoise_ldr, gt_ldr)
        self.log(f'{prefix}_loss', loss)
        denoise_hdr = restore_reinhard_t(denoise_ldr)
        gt_hdr = restore_reinhard_t(gt_ldr)
        denoise_ldr_np = denoise_ldr.detach().cpu().numpy()
        gt_ldr_np = gt_ldr.detach().cpu().numpy()
        denoise_hdr_np = restore_reinhard(denoise_ldr_np)
        gt_hdr_np = restore_reinhard(gt_ldr_np)

        denoise_tonemap_np = tonemap(denoise_hdr_np)
        denoise_tonemap_255_np = (denoise_tonemap_np * 255.0).astype(np.uint8)
        gt_tonemap_np = tonemap(gt_hdr_np)
        gt_tonemap_255_np = (gt_tonemap_np * 255.0).astype(np.uint8)

        total_psnr = 0.0
        total_rmse = 0.0
        total_ssim = 0.0
        for i in range(denoise_tonemap_255_np.shape[0]):
            psnr = calculate_psnr_255(denoise_tonemap_255_np[i], gt_tonemap_255_np[i])
            # print("psnr", psnr)
            if psnr is not None:
                # print(psnr)
                total_psnr += psnr
            else:
                assert "PSNR求值错误"
            rmse = calculate_rmse(denoise_tonemap_np[i], gt_tonemap_np[i])
            # print("rmse",rmse)
            if rmse is not None:
                total_rmse += rmse
            else:
                assert "RMSE求值错误"
            ssim = calculate_ssim_255(denoise_tonemap_255_np[i], gt_tonemap_255_np[i])
            # print("ssim",ssim)
            if ssim is not None:
                # print(ssim)
                total_ssim += ssim
            else:
                assert "SSIM求值错误"
        total = denoise_tonemap_255_np.shape[0]
        self.log(f'{prefix}_psnr', total_psnr / total)
        self.log(f'{prefix}_rmse', total_rmse / total)
        self.log(f'{prefix}_ssim', total_ssim / total)
        if prefix == 'val' and self.trainer.current_epoch % self.save_img_time == 0:
            # 保存图片
            batch_size = batch.shape[0]
            noisy_img = torchvision.utils.make_grid(tonemap_t(restore_reinhard_t(batch[:, 3:6, :, :])), nrow=batch_size,
                                                    padding=0, value_range=(0, 1))
            denoise_img = torchvision.utils.make_grid(tonemap_t(denoise_hdr), nrow=batch_size,
                                                      padding=0, value_range=(0, 1))
            gt_img = torchvision.utils.make_grid(tonemap_t(gt_hdr), nrow=batch_size, padding=0,
                                                 value_range=(0, 1))
            # # 多张图片组合输出
            grid_mix = torchvision.utils.make_grid(torch.cat((gt_img, noisy_img, denoise_img), dim=1), nrow=1)
            # # print(grid_target.shape)
            #
            # self.logger.experiment.add_image('noisy', noisy_img)
            # self.logger.experiment.add_image('denoise', denoise_img)
            # self.logger.experiment.add_image('gt', gt_img)
            self.logger.experiment.add_image('mix', grid_mix, self.trainer.current_epoch)
            pass

    def _LoG(self, x, y, sigma):
        return -1.0 / (np.pi * sigma ** 4) \
            * (1.0 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) \
            * np.e ** (-(x ** 2 + y ** 2) / (2 * sigma ** 2))


class DEMC(nn.Module):
    def __init__(self, aux_in_channels=7, aux_out_channels=32, features_encoder_layers=5, hdr_encoder_layers=6,
                 base_channels=64):
        super(DEMC, self).__init__()
        self.ffs = FeatureFusionSubNet(in_channels=aux_in_channels, out_channels=aux_out_channels)
        self.feature_decoder = Encoder(in_channels=aux_out_channels, base_channel=base_channels,
                                       layers=features_encoder_layers)
        self.hdr_encoder = Encoder(in_channels=3, base_channel=base_channels, layers=hdr_encoder_layers)
        self.hdr_decoder = HDRDecoder(base_channel=base_channels, layers=hdr_encoder_layers)

    def forward(self, noisy, aux):
        f = self.ffs(aux)
        fe = self.feature_decoder(f)
        he = self.hdr_encoder(noisy)
        denoise = self.hdr_decoder(he=he, fe=fe)
        return denoise


class FeatureFusionSubNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionSubNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channel=64, layers=5):
        super(Encoder, self).__init__()
        self.layers = layers
        conv_list = []
        conv_with_pool_list = []
        for i in range(layers):
            conv_list.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels if i == 0 else int(base_channel * math.pow(2, i - 1)),
                          out_channels=int(base_channel * math.pow(2, i)),
                          kernel_size=3, stride=1, padding='same'),
                nn.Conv2d(in_channels=int(base_channel * math.pow(2, i)),
                          out_channels=int(base_channel * math.pow(2, i)),
                          kernel_size=3, stride=1, padding='same'),
                nn.ReLU(inplace=True),
            ))
            if i != layers - 1:
                conv_with_pool_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=int(base_channel * math.pow(2, i)),
                                  out_channels=int(base_channel * math.pow(2, i)), kernel_size=3, stride=1,
                                  padding='same'),
                        nn.BatchNorm2d(int(base_channel * math.pow(2, i))),
                        nn.MaxPool2d(kernel_size=2, stride=2) if i != layers - 1 else nn.Identity(),
                        nn.ReLU(inplace=True)
                    )
                )
        self.encoder1 = nn.ModuleList(conv_list)
        self.encoder2 = nn.ModuleList(conv_with_pool_list)

    def forward(self, x):
        fs = []
        for i in range(self.layers):
            x = self.encoder1[i](x)
            fs.append(x)
            if i != self.layers - 1:
                x = self.encoder2[i](x)
        return fs[::-1]


class HDRDecoder(nn.Module):
    def __init__(self, base_channel=64, layers=6):
        super(HDRDecoder, self).__init__()
        self.layers = layers
        conv_list = []
        for i in range(layers):
            conv_list.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=base_channel * int(math.pow(2, layers - 1)),
                                   out_channels=int(base_channel * math.pow(2, layers - 2 - i)),
                                   kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ) if i == 0 else nn.Sequential(
                nn.Conv2d(in_channels=3 * int(base_channel * math.pow(2, layers - 1 - i)),
                          out_channels=int(base_channel * math.pow(2, layers - 1 - i)) if i != layers - 1 else 3,
                          kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=int(base_channel * math.pow(2, layers - 1 - i)),
                                       out_channels=int(base_channel * math.pow(2, layers - 2 - i)),
                                       kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(int(base_channel * math.pow(2, layers - 2 - i))),
                    nn.ReLU(inplace=True),
                ) if i != layers - 1 else nn.Identity()
            ))
        self.decoder = nn.ModuleList(conv_list)

    def forward(self, fe, he):
        for i in range(self.layers):
            if i == 0:
                x = he[0]
                x = self.decoder[0](x)
            else:
                # pingjie
                x = torch.cat([he[i], fe[i - 1], x], dim=1)
                x = self.decoder[i](x)
        return x


def test_encoder():
    x = torch.randn(1, 7, 256, 256).to('cuda')
    m = Encoder(in_channels=7, base_channel=64, layers=6).to('cuda')
    res = m.forward(x)
    for r in res: print(r.shape)


def test_decoder():
    m = HDRDecoder().to('cuda')


def test_demc():
    x = torch.randn(1, 3, 256, 256).to('cuda')
    a = torch.randn(1, 7, 256, 256).to('cuda')
    m = DEMC(base_channels=32).to('cuda')
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'DEMC总参数:{total_params / 1e6}M')
    if torch.cuda.is_available():
        print(f"Pytorch分配给GPU的显存开销: {torch.cuda.memory_allocated() // 1048576}MB")
    out = m.forward(noisy=x, aux=a)
    print(out.shape)
