import OpenEXR
import numpy as np
import cv2

def exr_to_np(path: str, channels: list) -> np.ndarray:
    if channels is None or len(channels) == 0:
        return np.array([])
    if path.endswith('.exr'):
        out = []
        with OpenEXR.File(path) as exr_file:
            exr_dict = exr_file.channels()
            for channel in channels:
                exr_arr = exr_dict[channel].pixels
                if len(exr_arr.shape) == 3:
                    out.append(exr_arr[:, :, 0])
                    out.append(exr_arr[:, :, 1])
                    out.append(exr_arr[:, :, 2])
                elif len(exr_arr.shape) == 2:
                    out.append(exr_arr if channel!="DenoisingDepth.V" else normalization(exr_arr))
            return np.stack(out,axis=-1,dtype=np.float32)
    else:
        return np.array([])

# 颜色空间变换 线性->SRGB
def linear_to_srgb(image:np.ndarray) -> np.ndarray:
    return np.clip(np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(image, 1/2.4) - 0.055),0.0,1.0)

# 将 sRGB 转换回线性颜色空间
def srgb_to_linear(image):
    # sRGB 到线性空间的转换公式
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

def normalization(arr:np.ndarray)->np.ndarray:
    return (arr - arr.min())/(arr.max()-arr.min())

# mix_np = exr_to_np("/media/yujiajing0408/Study/YCB/10/Mix.exr",["DenoisingDepth.V"])
# # noisy_np = exr_to_np("Noisy1spp.exr",["RGB"])
# # print(OpenEXR.InputFile("Mix.exr").header())
# # print(OpenEXR.InputFile("Noisy1spp.exr").header())
# # cv2.imshow("image",linear_to_srgb( exr_np[:,:,:3][:,:,::-1]))
# # cv2.imshow("noisy",exr_np[:,:,3:6][:,:,::-1])
# mix_np = (mix_np-mix_np.min())/(mix_np.max()-mix_np.min())
# print(mix_np.max(),mix_np.min())
# cv2.imshow("depth",mix_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()