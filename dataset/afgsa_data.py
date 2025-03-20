import os
from random import random
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms as tf
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class AFGSADataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, pin_memory=False, mode="train",
                 preprocess=False, img_size=120):
        super().__init__()
        self.img_size = img_size
        self.mode = mode
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pin_memory = pin_memory
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess = preprocess
        self.val_dir = os.path.join(data_dir, 'val.h5')
        self.train_dir = os.path.join(data_dir, 'train.h5')

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # 训练
        if stage in ['fit', None]:
            self.train_dataset = AFGSADataset(
                data_dir=self.train_dir,
                mode=self.mode,
                preprocess=self.preprocess,
                img_size=self.img_size
            ) if self.train_dir is not None else None
            self.val_dataset = AFGSADataset(
                data_dir=self.val_dir,
                mode=self.mode,
                preprocess=self.preprocess,
                img_size=self.img_size
            ) if self.val_dir is not None else None
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


class AFGSADataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 mode='train',
                 preprocess=False,
                 transforms=None,
                 img_size=120
                 ):
        assert data_dir.endswith('.h5'), "dataset_path must be the path to a .h5 file"
        assert os.path.exists(data_dir), "dataset_path is wrong"
        self.data_dir = data_dir
        with h5py.File(data_dir, 'r') as file:
            self.dataset_len = len(file["aux"])
            print(f"{mode}:{self.dataset_len}")

        if transforms is None and mode == "train":
            transforms = {
                "random_crop": (img_size, img_size),  # 随机裁切(W,H)
                "random_flip": (.0, .0),  # 随机翻转(水平,垂直)
                "random_rotate": 0 if mode != "test" else 0,  # 随机旋转
                "color_jitter": (.0, .0, .0, .0),  # 色彩抖动 brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化 根据给定的均值和标准差对图像进行归一化
            }
        self.preprocess = preprocess
        self.power = False
        tf_list = []
        if len(transforms.items()) != 0 and mode == "train":
            if transforms["random_flip"] != (.0, .0):
                tf_list.append(tf.RandomHorizontalFlip(transforms["random_flip"][0]))
                tf_list.append(tf.RandomVerticalFlip(transforms["random_flip"][1]))
            if transforms["color_jitter"] != (.0, .0, .0, .0):
                tf_list.append(
                    tf.ColorJitter(brightness=transforms["color_jitter"][0], contrast=transforms["color_jitter"][1],
                                   saturation=transforms["color_jitter"][2], hue=transforms["color_jitter"][3]))
            if transforms["random_rotate"] != .0:
                tf_list.append(tf.RandomRotation(degrees=transforms["random_rotate"]))
            if transforms["random_crop"][0] != .0 and transforms["random_crop"][1] != 0.0:
                tf_list.append(tf.RandomCrop(transforms["random_crop"]))
            # if transforms["normalize"] is not None:
            #     tf_list.append(tf.Normalize(mean=transforms["normalize"][0],std=transforms["normalize"][1]))
            self.transforms = tf.Compose(tf_list)
        if mode == "test":
            self.transforms = tf.CenterCrop(transforms["random_crop"])

    # def preprocess_data(self, data, t=2):
    #     for i in range(t):
    #         data = np.log(data + 1)
    #
    #     return data


    def preprocess_data(self, data,power):
        return np.power(data,power)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        file = h5py.File(self.data_dir, 'r')
        gt = file['gt'][index]
        noisy = file['noisy'][index]
        if self.preprocess:
            gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
            noisy_g = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
            diff = np.abs(gt_g - noisy_g)
            th = 10e-2
            noisy[:, :, 0][(diff >= th)] = 0.0
            noisy[:, :, 1][(diff >= th)] = 0.0
            noisy[:, :, 2][(diff >= th)] = 0.0
        aux = file['aux'][index]
        normal = aux[:, :, 3]
        # gt + noisy + depth + normal + albedo
        org = np.concatenate((gt, noisy, np.expand_dims(aux[:, :, 3], axis=2), aux[:, :, :3], aux[:, :, 4:]),
                             axis=2).transpose((2, 0, 1))
        # gt + noisy + depth + albedo + normal
        # org = np.concatenate((gt, noisy, np.expand_dims(aux[:, :, 3], axis=2), aux[:, :, 4:], aux[:, :, :3]),
        #                      axis=2).transpose((2, 0, 1))
        if self.power:
            power=(random()-0.5)/4 + 0.5
            org[0:3] = self.preprocess_data(org[0:3],power)
            org[3:6] = self.preprocess_data(org[3:6],power)
        res = torch.from_numpy(org)
        # print("transform前", res.shape)
        if self.transforms is not None:
            res = self.transforms(res)
        return res


def test_afgsa_dataset():
    ds = AFGSADataset(data_dir=r'/home/yujiajing0408/PycharmProjects/MD/datas/afgsa/val.h5',preprocess=True)
    # ds = AFGSADataset(data_dir=r'/home/yujiajing0408/PycharmProjects/MD/datas/afgsa/train.h5',preprocess=True)
    # d = ds[38].numpy().transpose((1, 2, 0))
    # cv2.imshow('gt', d[:, :, :3])
    # cv2.imshow('noisy', d[:, :, 3:6])
    # cv2.imshow('normal', d[:, :, 6:9])
    # cv2.imshow('albedo', d[:, :, 9])
    # cv2.imshow('albedo',d[:,:,10:13])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for i in range(ds.__len__()):
        img_gt = ds[i].numpy().transpose((1, 2, 0))[:,:,0:3]
        # print(f"{s}--{np.sum(np.isinf(img_gt))}--{np.sum(np.isnan(img_gt))}")
        if np.sum(img_gt>5):
            print(f"{i}--负数：{np.sum(img_gt < 0)}--最小值：{img_gt.min()}--最大值{img_gt.max()}")
    # gt = d[:, :, :3]
    # noisy = d[:, :, 3:6]
    # gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    # noisy_g = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    # # norm_gt = (gt_g - gt_g.min()) / (gt_g.max() - gt_g.min())
    # # norm_noisy = (noisy_g - noisy_g.min()) / (noisy_g.max() - noisy_g.min())
    # # diff = np.abs(norm_gt - norm_noisy)
    # diff = np.abs(gt_g - noisy_g)
    # th = 10e-2
    # print(diff.max())
    # print(np.sum((diff > th)))
    # print(diff.size)
    # print("mask_rate:",np.sum((diff > th))/diff.size)
    # org_noisy = noisy
    # print(noisy.shape)
    # noisy[:,:, 0][(diff >= th)] = 0.0
    # noisy[:, :, 1][(diff >= th)] = 0.0
    # noisy[:, :, 2][(diff >= th)] = 0.0
    # cv2.imshow('masked_noisy', noisy)
    # cv2.imshow('diff', diff)
    # # cv2.imshow('depth', d[:, :, 6])
    # # cv2.imshow('normal', d[:, :, 7:10])
    # # cv2.imshow('texture', d[:, :, 10:])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    total = 0.0
    for i in range(ds.__len__()):
        d = ds[i].numpy().transpose((1, 2, 0))
        total+=np.sum(d > 1.0)/d.size
    print(total/ds.__len__())
