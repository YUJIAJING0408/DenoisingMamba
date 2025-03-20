import numpy as np
import math
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def calculate_relative_mes(img1, img2, l1: float = 0.5, l2: float = 0.01):
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_relative_mes(img1[i], img2[i])
        return temp

    # 归一化
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    rel_mse = l1 * ((img1 - img2) ** 2 / (img2 ** 2 + l2))
    return np.mean(rel_mse)


def calculate_psnr_255(img1, img2):
    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_psnr(img1[i], img2[i])
        return temp

    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr(img1, img2):
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_psnr(img1[i], img2[i])
        return temp

    # if img1.dtype != np.int8 and img1.max() > 1.0 and img2.max()>1.0:
    # 归一化
    # print("归一化")
    # img1 and img2 have range [0, 1.0]
    # 归一化,img2是原图
    # img1 = (img1-img2.min()) / (img2.max() - img2.min())
    # img2 = (img2-img2.min()) / (img2.max() - img2.min())
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 50.0
    res = 20 * math.log10(255.0 / math.sqrt(mse))
    return res


def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # 一维高斯核
    kernel = cv2.getGaussianKernel(11, 1.5)
    # 二维
    window = np.outer(kernel, kernel.transpose())
    # 滤波结果排除边缘影响，
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) *
                                                            (sigma1_sq + sigma2_sq + c2))
    res = ssim_map.mean()
    return res

def calculate_ssim_255(img1, img2):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_ssim(img1[i], img2[i])
        return temp

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions!')


# 输入(b,c,h,w)
def calculate_ssim(img1, img2):
    # img1 and img2 have range [0, 255]
    # print(img1.shape[0])
    # print(img2.shape)
    # if np.max(img2) <= 1.0:
    img1 = img1.clip(0, 1.0) * 255.0
    img2 = img2.clip(0, 1.0) * 255.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_ssim(img1[i], img2[i])
        return temp

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions!')


def calculate_sharpness(image):
    # 使用Sobel算子计算梯度
    # sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float64).view(1, 1, 3, 3)
    # sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64).view(1, 1, 3, 3)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float64).view(1, 1, 3, 3)

    # 应用Sobel算子
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=1)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=1)

    # 计算梯度幅度
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # 锐度度量可以是梯度幅度的平均值或标准差
    sharpness = torch.mean(grad_magnitude)

    return sharpness


def sharpness_difference(noisy_image, target_image):
    # 计算两个图像的锐度
    noisy_sharpness = calculate_sharpness(noisy_image)
    # print(noisy_sharpness)
    target_sharpness = calculate_sharpness(target_image)
    # print(target_sharpness)

    # 计算锐度差异，这里使用绝对值差作为示例
    difference = torch.abs(noisy_sharpness - target_sharpness)

    return difference


def calculate_rmse_old(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions!')

        # multiple images
    if img1.ndim == 4:
        rmse_list = []
        for i in range(len(img1)):
            rmse_list.append(calculate_rmse_old(img1[i], img2[i]))
        return np.mean(rmse_list)  # 返回所有图像RMSE的均值

    # 计算RMSE
    diff = img1 - img2
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse


def calculate_rmse(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions!')

    #     # multiple images
    # if img1.ndim == 4:
    #     rmse_list = []
    #     for i in range(len(img1)):
    #         rmse_list.append(calculate_rmse(img1[i], img2[i]))
    #     return np.mean(rmse_list)  # 返回所有图像RMSE的均值

    # 计算RMSE
    num = (img1 - img2) ** 2
    denom = img2 ** 2 + 1.0e-2
    relative_mse = np.divide(num, denom)
    relative_mse_mean = 0.5 * np.mean(relative_mse)
    return relative_mse_mean


def sharp(img):
    # 使用Sobel算子计算x和y方向的梯度
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # 计算梯度幅度（近似于锐度）
    # 使用np.sqrt(a**2 + b**2)计算每个像素的梯度幅度
    # 但为了效率，我们使用np.hypot函数，它是sqrt(a**2 + b**2)的更快实现
    gradient_magnitude = np.hypot(grad_x, grad_y)

    # 你可以选择对梯度幅度进行平均、中值或其他统计计算
    # 来得到一个单一的锐度值
    # 例如，计算所有像素梯度幅度的平均值
    sharpness_value = np.mean(gradient_magnitude)
    print(f"Sharpness Value: {sharpness_value}")

    return math.fabs(sharpness_value)


def calculate_sharp(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions!')

        # multiple images
    if img1.ndim == 4:
        rmse_list = []
        for i in range(len(img1)):
            rmse_list.append(calculate_sharp(img1[i], img2[i]))
        return np.mean(rmse_list)  # 返回所有图像RMSE的均值

    # 计算RMSE
    sharp_img1 = sharp(img1)
    sharp_img2 = sharp(img2)
    s = math.fabs(1.0 - sharp_img1 / sharp_img2)
    return s


if __name__ == '__main__':
    # img1 = np.random.rand(100, 100)  # 假设这是第一个图像
    # img2 = np.clip((img1 + np.random.randint(-5, 5, (100, 100), dtype=np.int16) / 255), 0.0, 1.0)
    # img1_t = torch.from_numpy(img1).reshape(1, 100, 100)
    # img2_t = torch.from_numpy(img2).reshape(1, 100, 100)
    # # print(sharpness_difference(img1_t, img2_t))
    # print(calculate_relative_mes(img1, img2))

    # img2 = np.random.rand(4, 3)  # 假设这是第二个图像
    # print(img1)
    # print(img2)
    # rmse_value = calculate_rmse(img1, img2)
    # print(rmse_value)
    # print(img1)
    # print(calculate_sharp(img1, img2))
    org_img = np.random.uniform(low=.0, high=1.05, size=(100, 100, 3)).astype(np.float32)
    org_img_255 = org_img * 255.0
    img_255 = org_img_255 + np.random.randint(-100, 100, (100, 100,3), dtype=np.int8)
    img_1 = img_255 / 255.0
    print(calculate_psnr(img_1, org_img))
    print(calculate_ssim(img_1.clip(0.0, 1.0).transpose(2,0,1), org_img.clip(0.0, 1.0).transpose(2,0,1)))
    print(calculate_ssim(img_1.transpose(2,0,1), org_img.transpose(2,0,1)))