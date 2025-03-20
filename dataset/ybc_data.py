import os
import random
import shutil
import time
from datetime import datetime
from itertools import count
from typing import Optional, Callable

import OpenEXR
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as tf
from lightning import LightningDataModule
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from utils.exr_utils import exr_to_np, linear_to_srgb
from utils.image import noisy_detection_255, noisy_detection_with_org
from utils.preprocessing import tonemap, tonemap_t, tonemap_wo_clip, reinhard, restore_reinhard, preprocess_normal

# channels
DENOISING_NORMAL = "DenoisingNormal"
DENOISING_ALBEDO = "DenoisingAlbedo"
DENOISING_DEPTH = "DenoisingDepth.V"
GT = "GT"
NOISY = "RGB"

SPP_1 = "1spp"
SPP_2 = "2spp"
SPP_4 = "4spp"
SPP_8 = "8spp"
SPP_16 = "16spp"
SPP_32 = "32spp"
SPP_64 = "64spp"
SPP_128 = "128spp"


class YBCDataModule(LightningDataModule):
    def __init__(self, data_dir: str, npy_dir: str, batch_size: int, num_workers: int, spp=SPP_4, pin_memory=False,
                 sample_size=(120, 120),need_reinhard=False,need_dispose_normal=False):
        super().__init__()
        self.need_dispose_normal = need_dispose_normal
        self.need_reinhard = need_reinhard
        self.spp = spp
        self.npy_dir = npy_dir
        self.sample_size = sample_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pin_memory = pin_memory
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # 训练
        if stage in ['fit', None]:
            self.train_dataset = YBCDataset(
                data_dir=self.data_dir,
                npy_dir=self.npy_dir,
                mode='train',
                spp=self.spp,
                out_size=self.sample_size,
                sample_rate=1,
                need_reinhard=self.need_reinhard,
                need_dispose_normal=self.need_dispose_normal,
            )
            self.val_dataset = YBCDataset(
                data_dir=self.data_dir,
                npy_dir=self.npy_dir,
                mode='val',
                spp=self.spp,
                out_size=self.sample_size,
                sample_rate=1,
                need_reinhard=self.need_reinhard,
                need_dispose_normal=self.need_dispose_normal,
            )
        if stage in ["test", None]:
            pass
        if stage in ["predict", None]:
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=self.pin_memory, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=self.pin_memory, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=self.pin_memory, drop_last=True)


##
# YBC数据集格式 1920 * 1920
# data_dir
# --scene1 (9 Files)
# ----Mix.exr (GT,DenoisingNormal.DenoisingAlbedo,DenoisingDepth.V)
# ----Noisy1spp.exr (B,G,R;NOISY with n spp)
# ----Noisy2spp.exr
# ----Noisy4spp.exr
# ----Noisy8spp.exr
# ----Noisy16spp.exr
# ----Noisy32spp.exr
# ----Noisy64spp.exr
# ----Noisy128spp.exr
# --scene2
# ...
# #

class YBCDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 npy_dir: str,
                 spp: str = SPP_4,
                 channels=None,
                 mode: str = "train",
                 sample_rate: int = 1,
                 out_size: tuple = (120, 120),
                 max_light:float=255.0,
                 tfs=None,
                 crop_size:int=6, # 切块大小
                 train_rate:float=0.9, # 生成数据集的train:test
                 need_reinhard:bool=False, # 是否进行色调映射HDR[0,INF]->LDR[0:1] (GT,Noisy,Albedo)
                 need_dispose_normal=False, # 将法线由[-1.0,1.0]->[0.0,1.0] (Normal)
                 seed="010408"):
        self.need_dispose_normal = need_dispose_normal
        self.need_reinhard = need_reinhard
        self.train_rate = train_rate
        self.npy_dir = npy_dir
        self.max_light = max_light
        if channels is None:
            self.channels = [GT, DENOISING_NORMAL, DENOISING_ALBEDO, DENOISING_DEPTH]
        else:
            self.channels = channels
        if tfs is None:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(5),
                transforms.RandomCrop(out_size),
            ])
        else:
            self.transforms = tfs
        self.out_size = out_size
        self.sample_rate = sample_rate
        self.mode = mode
        self.spp = spp
        self.data_dir = data_dir
        random.seed(seed)
        self.scenes = os.listdir(self.data_dir)
        self.data = []
        # 检查npy是否存在
        spp_dir = os.path.join(self.npy_dir, spp)
        if not os.path.exists(spp_dir):
            os.mkdir(spp_dir)
        train_npy_path = os.path.join(spp_dir, f"train{'_reinhard' if need_reinhard else ''}{'_normal' if need_reinhard else ''}.npy")
        test_npy_path = os.path.join(spp_dir, f"test{'_reinhard' if need_reinhard else ''}{'_normal' if need_reinhard else ''}.npy")
        if os.path.exists(train_npy_path) and os.path.exists(test_npy_path):

            if mode == "train":
                print(mode, ": load npy file. Path: ", train_npy_path)
                self.data = np.load(train_npy_path)
            if mode == "test":
                print(mode, ": load npy file. Path: ", test_npy_path)
                self.data = np.load(test_npy_path)
            if mode == "val":
                print(mode, ": load npy file. Path: ", test_npy_path)
                self.data = np.load(test_npy_path)
            self.print_info()
        else:
            train_data = []
            test_data = []
            array = list(range(36))
            for scene in self.scenes:
                test_ids = random.sample(array, int((1 - self.train_rate) * (6 * 6)))
                print(test_ids)
                scene_full_path = os.path.join(self.data_dir, scene)
                mix_full_path = os.path.join(scene_full_path, "Mix.exr")
                noisy_full_path = os.path.join(scene_full_path, f"Noisy{self.spp}.exr")
                mix_np = exr_to_np(mix_full_path, self.channels)
                noisy_np = exr_to_np(noisy_full_path, [NOISY])
                noisy_np = np.where(noisy_np < 0.0, 0.0, noisy_np)
                gt_np = mix_np[:, :, :3]
                normal_np = mix_np[:, :, 3:6]
                albedo_np = mix_np[:,:,6:9]
                depth_np = mix_np[:, :, 9:10]
                # 手动限制亮度
                # noisy_np = np.clip(noisy_np, 0.0, max_light)
                # gt_np = np.clip(gt_np, 0.0, max_light)
                # mix_np[:, :, 6:9] = np.clip(tonemap_wo_clip(mix_np[:, :, 6:9]), 0.0, 50.0)
                if need_dispose_normal:
                    normal_np = preprocess_normal(normal_np)
                if need_reinhard: # reinhard支持无损还原
                    noisy_np = reinhard(noisy_np)
                    gt_np = reinhard(gt_np)
                    albedo_np = reinhard(albedo_np)


                mix = np.concatenate([gt_np, noisy_np, normal_np,albedo_np,depth_np], axis=-1)
                for i in range(crop_size):
                    for j in range(crop_size):
                        tile = mix[320 * i:320 * (i + 1), 320 * j:320 * (j + 1), :]
                        if i * crop_size + j in test_ids:
                            test_data.append(tile)
                        else:
                            train_data.append(tile)
            print(f"test size: {len(test_data)}\n train size: {len(train_data)}")
            with open(train_npy_path, "wb") as f:
                np.save(f, arr=train_data)
            with open(test_npy_path, "wb") as f:
                np.save(f, arr=test_data)
            if mode == "train":
                self.data = train_data
            else:
                self.data = test_data
            self.print_info()

    # 根据采样率获得更多数据
    def __len__(self):
        return len(self.data) * self.sample_rate

    def print_info(self):
        print(f"{self.mode} size: {len(self.data)}")
        print(f"SPP: {self.spp}")
        if self.need_reinhard:
            print("GT,Noisy and Albedo are reinhard to LDR ,Which are in [0,1.0]!")
        else:
            print("GT,Noisy and Albedo maybe HDR ,Which are in [0,Inf]!")
        if self.need_dispose_normal:
            print("Normal are dispose in [0,1.0]!")
        else:
            print("Normal maybe in [-1.0,1.0]!")
        print(f"One Sample Channels are {'GT' if 'GT' in self.channels else ''} Noisy {'Normal' if 'DenoisingNormal' in self.channels else ''} {'Albedo' if 'DenoisingAlbedo' in self.channels else ''} {'Depth' if 'DenoisingDepth' in self.channels else ''}")

    def __getitem__(self, idx):
        # %运算获取真实场景id
        scene_id = idx % self.data.__len__()
        mix = self.data[scene_id]
        # 执行Transforms
        res = self.transforms(mix)
        ##
        # GT:[:3,:,:]
        # Noisy:[3:6,:,:]
        # Aux:[6:,:,:]
        # #
        return res


def test_ybc_dataset():
    # train_dataset = YBCDataset(data_dir="/media/yujiajing0408/Study/YCB",npy_dir="/media/yujiajing0408/Study/YCB_NP",mode="train")
    # print(len(train_dataset))

    test_dataset = YBCDataset(data_dir="/media/yujiajing0408/Study/YCB", spp=SPP_32,
                              npy_dir="/media/yujiajing0408/Data/YCB_NP",
                              out_size=(320, 320),
                              sample_rate=1,
                              mode = "test",
                              # need_reinhard=True,
                              # need_dispose_normal=True,
                              tfs=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.RandomCrop((320,320)),
                              ]))

    print(len(test_dataset))
    # print(test_dataset[0].shape)
    # print(len(test_dataset))
    # print(test_dataset[0][:3, :, :].max(), test_dataset[0][:3, :, :].min())
    # random.seed(time.time())
    # # sample_list = random.sample(list(range(len(test_dataset))), 10)

    # sample_list = list(range(test_dataset.__len__()))
    # # sample_list = list(range(270,275))
    # # sample_list = [270,275]
    # count=0
    # light = 2
    # for s in sample_list:
    #     d = test_dataset[s]
    #
    #     img_gt = d.permute(1, 2, 0).numpy()
    #     # cv2.imshow("img_gtt_" + str(s),tonemap(img_gt[:, :, 0:3][:,:,::-1]))
    #     cv2.imshow("img_gt_" + str(s), img_gt[:, :, 0:3][:, :, ::-1])
    #     # cv2.imshow("img_gtr_" + str(s), reinhard(img_gt[:, :, 0:3][:, :, ::-1]))
    #     # cv2.imshow("img_gtrr_" + str(s), restore_reinhard(reinhard(img_gt[:, :, 0:3][:, :, ::-1])))
    #     cv2.imshow("img_noisyt_" + str(s), tonemap(img_gt[:, :, 3:6][:,:,::-1]))
    #     cv2.imshow("img_noisy_" + str(s), img_gt[:, :, 3:6][:, :, ::-1])
    #     cv2.imshow("img_normal_" + str(s), img_gt[:, :, 6:9])
    #     cv2.imshow("img_albedo_" + str(s), img_gt[:, :, 9:12][:,:,::-1] ** 2.2)
    #     # cv2.imshow("img_albedo__" + str(s), img_gt[:, :, 9:12][:, :, ::-1] )
    #     cv2.imshow("img_depth_" + str(s), img_gt[:, :, 12:])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     # print(f"{s}--{np.sum(np.isinf(img_gt))}--{np.sum(np.isnan(img_gt))}")
    #     print(f"{s}--负数：{np.sum(img_gt < 0)}--最小值：{img_gt.min()}--最大值{img_gt.max()}--{np.sum(img_gt > light) / (320 * 320 * 3)}")
    # #     if np.sum(img_gt>light)/(320*320*3)>1e-4:
    # #         count+=1
    # #         print(f"{s}--负数：{np.sum(img_gt<0)}--最小值：{img_gt.min()}--最大值{img_gt.max()}--{np.sum(img_gt>light)/(320*320*3)}")
    # # print(f"{count}/{test_dataset.__len__()}")
