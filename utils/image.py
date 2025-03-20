import os
import shutil

import Imath
import OpenEXR
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # 不要修改位置，必须在cv2上面否则打开EXR时会报初始化错误
import cv2

import numpy as np


def sliding_window(input_tensor, window_size):
    # 假设 input_tensor 的形状是 (batch_size, channels, height, width)
    # 并且 window_size 是一个二元组，表示 (height_window, width_window)
    batch_size, channels, height, width = input_tensor.size()
    stride = (window_size[0], window_size[1])  # 假设步长与窗口大小相同，即无重叠
    # 计算可以划分的窗口数量
    num_windows_height = (height - window_size[0]) // stride[0] + 1
    num_windows_width = (width - window_size[1]) // stride[1] + 1
    # 初始化一个空的列表来存储所有的窗口
    windows = []
    # 遍历每个可能的窗口位置
    for i in range(num_windows_height):
        for j in range(num_windows_width):
            # 使用切片来提取当前的窗口
            window = input_tensor[:, :, i:i + window_size[0], j:j + window_size[1]]
            windows.append(window)
            # 如果需要，可以将列表转换为张量（例如，使用 torch.stack）
    # 但请注意，这将增加一个额外的维度来表示窗口的索引
    # 如果需要，可以进一步处理这个额外的维度
    windows_tensor = torch.stack(windows,
                                 dim=0)  # 这将得到一个 (num_windows, batch_size, channels, height_window, width_window) 的张量
    return windows_tensor  # 返回窗口列表


def merge_windows(windows_tensor, org_window_size):
    windows_len, batch_size, channels, height, width = windows_tensor.size()
    assert windows_len * height * width == org_window_size[0] * org_window_size[1], \
        "输入不匹配"
    windows_height_len = org_window_size[0] // height
    windows_width_len = org_window_size[1] // width
    windows = []
    for i in range(windows_len):
        for j in range(windows_height_len):
            window = windows_tensor[:, :, :, :]


def restore_from_sliding_windows(windows_tensor, original_size, window_size):
    # original_size: (batch_size, channels, height, width)
    # window_size: (height_window, width_window)
    print(windows_tensor.shape)
    batch_size, channels, height, width = original_size
    if batch_size != windows_tensor.shape[1]:
        batch_size = windows_tensor.shape[1]
    stride = window_size  # 假设步长与窗口大小相同

    # 计算窗口数量
    num_windows_height = (height - window_size[0]) // stride[0] + 1
    num_windows_width = (width - window_size[1]) // stride[1] + 1

    # 初始化一个全零的张量，形状与原始输入相同
    restored_tensor = torch.zeros(batch_size, channels, height, width, dtype=windows_tensor.dtype,
                                  device=windows_tensor.device)
    # for i in range(windows_tensor.shape[0]):
    #     if windows_tensor[i].shape != (16,10,4,4):
    #         print(windows_tensor[i].shape)
    # 遍历每个窗口，并将它们放回原始位置
    for i in range(num_windows_height):
        for j in range(num_windows_width):
            # 计算窗口在原始张量中的起始索引
            start_h = i * stride[0]
            start_w = j * stride[1]
            # 从windows_tensor中获取对应的窗口
            window = windows_tensor[i * num_windows_width + j]
            # 将窗口放回到restored_tensor的对应位置
            restored_tensor[:, :, start_h:start_h + window_size[0], start_w:start_w + window_size[1]] = window

    return restored_tensor


def exr_to_dict(exr_path, channels):
    img = OpenEXR.InputFile(str(exr_path))
    dw = img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels_str = img.channels(channels, float_type)

    out = {}
    for channel_name, channel_str in zip(channels, channels_str):
        out[channel_name] = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)

    return out


def copy_folder(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_folder(s, d)
        else:
            shutil.copy2(s, d)  # 使用copy2保留文件元数据

def noisy_detection_with_org(noisy:np.ndarray,org:np.ndarray,need_bool=True)->np.ndarray:
    noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(org, noisy)
    _,mask = cv2.threshold(diff, 1, 1, cv2.THRESH_BINARY)
    if need_bool:
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    # 形态学操作（连接离散的噪声点）
    kernel = np.ones((15, 15), np.uint8)  # 定义形态学操作的内核大小
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算（先膨胀后腐蚀）
    # 填充空洞（确保mask区域是连续的）
    mask_filled = mask_closed.copy()
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(mask_filled, [cnt], 0, 1, -1)  # 填充轮廓内部
    return mask_filled
    pass

def noisy_detection_255(noisy:np.ndarray,need_bool:bool=True)->np.ndarray:
    noisy = cv2.cvtColor(noisy,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(noisy, (3,3), 0)
    diff = cv2.absdiff(noisy, blurred)
    _,mask = cv2.threshold(diff,3,1,cv2.THRESH_BINARY)
    if need_bool:
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    # 形态学操作（连接离散的噪声点）
    kernel = np.ones((15, 15), np.uint8)  # 定义形态学操作的内核大小
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算（先膨胀后腐蚀）
    # 填充空洞（确保mask区域是连续的）
    mask_filled = mask_closed.copy()
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(mask_filled,[cnt], 0, 1, -1)  # 填充轮廓内部
    return mask_filled

if __name__ == '__main__':
    # exr_path = r'E:\Develop\Projects\python\RTDN\RTDN\datas\source\RTAD\chess\denoise-normal.exr'
    # print(OpenEXR.InputFile(exr_path).header())
    # exr_dict = exr_to_dict(exr_path, ["X","Y","Z"])
    # print(exr_dict["X"].shape)
    # print(exr_dict["Y"].shape)
    # print(exr_dict["Z"].shape)

    exr_path = r'E:\Develop\Projects\python\RTDN\RTDN\datas\source\RTAD\turkish\denoise-depth.exr'
    print(OpenEXR.InputFile(exr_path).header())
    # img = cv2.imread(exr_path,cv2.IMREAD_UNCHANGED)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
