import numpy as np
import torch
import random
from random import randint


# constants
eps = 0.00316

def preprocess_diffuse_t(diffuse,albedo):
    return diffuse / (albedo + eps)

def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)

def preprocess_specular_t(specular):
    # assert np.sum(specular < 0) == 0, "Negative value in specular component!"
    return torch.log(specular + 1)

def preprocess_specular(specular):
    # assert np.sum(specular < 0) == 0, "Negative value in specular component!"
    return np.log(specular + 1)


def preprocess_depth(depth):
    depth = np.clip(depth, 0.0, np.max(depth))
    max_feature = np.max(depth)
    if max_feature != 0:
        depth /= max_feature
    return depth

def preprocess_normal_t(normal):
    # normal = torch.nan_to_num(normal,nan=0.0, posinf=1.0, neginf=-1.0)
    normal = (normal + 1.0) * 0.5
    normal = torch.clip(normal, 0.0, 1.0)
    return normal

def preprocess_normal(normal):
    normal = np.nan_to_num(normal)
    normal = (normal + 1.0) * 0.5
    normal = np.maximum(np.minimum(normal, 1.0), 0.0)
    return normal

def postprocess_diffuse_t(diffuse, albedo):
    return diffuse * (albedo + eps)

def postprocess_diffuse(diffuse, albedo):
    return diffuse * (albedo + eps)

def postprocess_specular_t(specular):
    return torch.exp(specular) - 1

def postprocess_specular(specular):
    return np.exp(specular) - 1

def tonemap(matrix: np.ndarray, gamma: float = 2.2):
    return np.clip(np.clip(matrix,0.0,matrix.max()) ** (1.0 / gamma), 0, 1)

def tonemap_wo_clip(matrix: np.ndarray, gamma: float = 2.2):
    return np.clip(matrix,0.0,matrix.max()) ** (1.0 / gamma)

def tonemap_t(matrix: torch.Tensor, gamma: float = 2.2):
    return torch.clip(torch.clip(matrix,0.0,matrix.max()) ** (1.0 / gamma), 0, 1)

# Reinhard 色调映射
def reinhard(hdr:np.ndarray)->np.ndarray:
    hdr = np.clip(hdr, 0.0, hdr.max()) # 避免负数
    return hdr / (hdr + 1)

def reinhard_t(hdr:torch.Tensor)->torch.Tensor:
    hdr = torch.clip(hdr, 0.0, hdr.max()) # 避免负数
    return hdr / (hdr + 1)

def restore_reinhard(ldr:np.ndarray)->np.ndarray:
    ldr = np.clip(ldr, 0.0, (1.0 - 1e-5)) # 避免下面计算时分母为零
    return ldr / (1 - ldr)

def restore_reinhard_t(ldr:torch.Tensor)->torch.Tensor:
    ldr = torch.clip(ldr, 0.0, (1.0 - 1e-5)) # 避免下面计算时分母为零
    return ldr / (1 - ldr)

def test():
    random_tensor = (torch.rand(2, 2) - 0.5) * 5
    print(tonemap_t(random_tensor))
    print(tonemap(random_tensor.numpy()))
