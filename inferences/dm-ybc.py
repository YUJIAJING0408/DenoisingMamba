# test interface
#
import argparse
import json
import math
import os
import time
import PIL
import cv2
import numpy as np
import pyexr
import torchvision.transforms
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from dataset.ybc_data import YBCDataset
from models.DenoisingMamba.v3 import DenoisingMambaModel
from utils.metric import calculate_rmse,calculate_psnr_255,calculate_ssim_255
from utils.preprocessing import restore_reinhard, tonemap


# 解析命令行参数
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-td", "--test_datas", type=str, help="")
    args.add_argument("-is", "--image_size", type=int, default=600)
    args.add_argument("-m", "--model_path", type=str, help="训练模型")
    args.add_argument("-o", "--out_path", type=str, default=None, help="输出位置（非必须，不填输出到根目录）")
    args.add_argument("-d", "--device", default="cuda", help="运行环境")
    args = args.parse_args()
    return args


def build_data(args):
    test_data = YBCDataset(data_dir="/media/yujiajing0408/Study/YCB", spp="4spp",
                              npy_dir="/media/yujiajing0408/Data/YCB_NP",
                              sample_rate=1,
                              mode = "test",
                              need_reinhard=True,
                              need_dispose_normal=True,
                              tfs=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.CenterCrop((320,320)),
                                  transforms.Resize((240,240),antialias=False),
                              ]))
    return test_data


if __name__ == '__main__':
    args = parse_args()
    # args.process = False
    test_data = build_data(args)
    model = DenoisingMambaModel.load_from_checkpoint(
        f'/media/yujiajing0408/Data/MD_logs/AFGM-v3-ybc/[2025-02-4:21]-|epoch=1500|-|lr=0.002|-|loss_rate=0.65:0.35|_32SPP|PSNR-02:04:3/checkpoints/last.ckpt',
        strict=False,
        mix_rule={
            "noisy": (0, 3),
            "aux": (3,9)
        },
        if_abs_pos_embed=True
    ).to(args.device)

    model.img_size = args.image_size
    model.eval()
    model.freeze()
    a_s = []
    a_p = []
    a_r = []
    a_t = []
    args.out_path = r'/home/yujiajing0408/PycharmProjects/MD/outputs/MB-test'
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os.makedirs(os.path.join(args.out_path, "mix"))
        os.makedirs(os.path.join(args.out_path, "denoise"))
        os.makedirs(os.path.join(args.out_path, "gt"))
        os.makedirs(os.path.join(args.out_path, "noisy"))
    with torch.no_grad():
        with open(os.path.join(args.out_path, 'datas.txt'), 'w') as file:
            for i in tqdm(range(test_data.__len__())):
            # for i in tqdm([i * 5 for i in range(63)]):
                batch = test_data[i].unsqueeze(0).to(args.device)
                gt_ldr = batch[:, :3, :, :].squeeze()
                noisy_ldr_np = batch[:, 3:6, :, :].squeeze().cpu().numpy() # [3,120,120]
                noisy_hdr_np = restore_reinhard(noisy_ldr_np)
                noisy_tonemap_np = tonemap(noisy_hdr_np)
                noisy_tonemap_255_np = (noisy_tonemap_np * 255.0).astype(np.uint8)
                noisy_with_aux_features = batch[:, 3:, :, :]
                begin = time.perf_counter()
                denoise_ldr = model.forward(noisy_with_aux_features).squeeze(0)
                end = time.perf_counter()
                t = end - begin
                print(t)
                # 转为np
                denoise_ldr_np = denoise_ldr.detach().cpu().numpy()
                gt_ldr_np = gt_ldr.detach().cpu().numpy()
                # LDR->HDR(无损Reinhard)
                denoise_hdr_np = restore_reinhard(denoise_ldr_np)
                gt_hdr_np = restore_reinhard(gt_ldr_np)
                # HDR->LDR(有损Tone map映射)
                denoise_tonemap_np = tonemap(denoise_hdr_np)
                denoise_tonemap_255_np = (denoise_tonemap_np * 255.0).astype(np.uint8)
                gt_tonemap_np = tonemap(gt_hdr_np)
                gt_tonemap_255_np = (gt_tonemap_np * 255.0).astype(np.uint8)

                gt_img = PIL.Image.fromarray(np.transpose(gt_tonemap_255_np, (1, 2, 0)),"RGB")
                denoise_img = PIL.Image.fromarray(np.transpose(denoise_tonemap_255_np,(1,2,0)), "RGB")
                noisy_img = PIL.Image.fromarray(np.transpose(noisy_tonemap_255_np,(1,2,0)), "RGB")
                gt_img = gt_img.resize((320,320))
                denoise_img = denoise_img.resize((320, 320))
                noisy_img = noisy_img.resize((320, 320))

                gt_img.save(os.path.join(args.out_path, "gt", "{}.png".format(i + 1)))
                denoise_img.save(os.path.join(args.out_path, "denoise", "{}.png".format(i + 1)))
                noisy_img.save(os.path.join(args.out_path, "noisy", "{}.png".format(i + 1)))

                # save
                mix = PIL.Image.new("RGB",(gt_img.width * 3,gt_img.height))
                mix.paste(noisy_img, (0,0))
                mix.paste(denoise_img, (gt_img.width,0))
                mix.paste(gt_img, (gt_img.width * 2,0))
                mix.save(os.path.join(args.out_path,"mix","{}.png".format(i + 1)))


                r = calculate_rmse(denoise_tonemap_np, gt_tonemap_np)
                p = calculate_psnr_255(denoise_tonemap_255_np,gt_tonemap_255_np)
                s = calculate_ssim_255(denoise_tonemap_255_np,gt_tonemap_255_np)
                # print(f"------第{i + 1}场景-------")
                # print("|RMSE:{:.10f}|".format(r))
                # print("|PSNR:{:.9f}|".format(p))
                # print("|SSIM:{:.10f}|".format(s))
                a_s.append(s)
                a_p.append(p)
                a_r.append(r)
                a_t.append(t)
                # print("|耗时：{:.10f}|".format(t))
                # print("-------------------")
                file.write(f"------第{i + 1}场景-------\n")
                file.write("|RMSE:{:.10f}|\n".format(r))
                file.write("|PSNR:{:.9f}|\n".format(p))
                file.write("|SSIM:{:.10f}|\n".format(s))
                file.write("|耗时：{:.10f}|\n".format((end - begin)))
                file.write("-------------------\n")
                del batch
            file.write("RMSE_AVG:{}\n".format(np.mean(a_r)))
            file.write("PSNR_AVG:{}\n".format(np.mean(a_p)))
            file.write("SSIM_AVG:{}\n".format(np.mean(a_s)))
            file.write("TIME_AVG:{}\n".format(np.mean(a_t)))
    print("AVG_SSIM",np.mean(a_s, axis=0))
    print("AVG_PSNR",np.mean(a_p, axis=0))
    print("AVG_RMSE",np.mean(a_r, axis=0))
    print("AVG_TIME", np.mean(a_t, axis=0))
    print(np.mean(a_t))
    print(f"Pytorch分配给GPU的显存开销: {torch.cuda.max_memory_allocated() // 1048 ** 2}MB")
    print("\tDone!")