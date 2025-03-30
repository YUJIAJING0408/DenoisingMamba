# MAMBA for pic
import math

import torch
from torch import nn, Tensor
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
from mamba_ssm.modules.mamba_simple import Mamba
from einops import rearrange
from typing import List, Optional, Sequence, Tuple, Any, Callable
from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn, layer_norm_fn
except ImportError:
    # 加载失败
    RMSNorm, rms_norm_fn, layer_norm_fn = None, None, None


# 将图像变为时序:(1,3,120,120) -> (1,75,24,24) -> (1,75,576) -> (1,576,75)
class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size: tuple,
                 patch_size: int,
                 stride,
                 in_channels,
                 norm_layer=None):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.in_channels = in_channels
        self.embedding_dim = (patch_size ** 2) * in_channels
        self.grid_size = ((self.img_size[0] - self.patch_size[0]) // stride + 1,
                          (self.img_size[1] - self.patch_size[1]) // stride + 1)
        self.patches = self.grid_size[0] * self.grid_size[1]  # 切完片的个数（时序）
        self.patch = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size=self.patch_size, stride=self.stride)
        self.norm = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        self.patches = self.grid_size[0] * self.grid_size[1]  # 切完片的个数（时序）
        if self.training:  # 训练时固定大小，测试时可以用大小同时可以被3,4,5,6整除任意正方形大小的数据
            assert h == self.img_size[0] and w == self.img_size[1], \
                f"输入图片{h} * {w}不匹配模型的{(self.img_size[0])} * {(self.img_size[1])}"
        x = self.patch(x)
        x = x.flatten(start_dim=2, end_dim=3).transpose(1, 2)
        x = self.norm(x)
        return x


class UnPatchEmbedding(nn.Module):
    def __init__(self,
                 grid_size: tuple,
                 patch_size,
                 stride,
                 in_channels,
                 out_channels,
                 ):
        super(UnPatchEmbedding, self).__init__()
        self.grid_size = grid_size
        self.patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.change_channel = nn.Conv2d(self.in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if len(x.shape) == 3:
            b, s, c = x.shape
            # x = x.transpose(1, 2)
            if b * s * c == b * self.in_channels * self.grid_size[0] * self.grid_size[1] * self.patch_size[0] * \
                    self.patch_size[1]:
                x = rearrange(x, 'b (h w) (c ph pw) -> b c (h ph) (w pw)',
                              h=self.grid_size[0], w=self.grid_size[1], ph=self.patch_size[0], pw=self.patch_size[1])
            else:
                x = rearrange(x, 'b (h w) (c ph pw) -> b c (h ph) (w pw)',
                              h=int(math.sqrt(s)), w=int(math.sqrt(s)), ph=self.patch_size[0], pw=self.patch_size[1])
            # print(x.shape)
            return self.change_channel(x)


class Block(nn.Module):
    def __init__(self, layer_idx: int, patch_embed_dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False,
                 drop_path_rate=0.):
        super(Block, self).__init__()
        self.layer_idx = layer_idx
        self.patch_embed_dim = patch_embed_dim  # 经过PatchEmbedding后的emb_dim
        self.residual_in_fp32 = residual_in_fp32  # 残差是否用32位浮点
        self.fused_add_norm = fused_add_norm  # 是否用RMSNorm
        self.mixer_cls = mixer_cls(patch_embed_dim)  #
        self.norm = norm_cls(patch_embed_dim)
        self.drop_path = DropPath(drop_path_rate)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm导入失败"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), "Block: fused_add_norm只支持LayerNorm和RMSNorm"

    def forward(self, hidden_states: Tensor = None, residual: Optional[Tensor] = None, inference_params=None):
        if not self.fused_add_norm:
            # 自定义norm
            if residual is None:
                # 第一次传入时，残差为空，将输入赋给残差
                residual = hidden_states
            else:
                # 后续传入，将输入一部分遗忘后与残差相加
                residual = self.drop_path(hidden_states) + residual
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(dtype=torch.float32)
        else:
            # 如果传入的norm不是RMSNorm则使用layer_norm_fn(比nn.LayerNorm优化更好，收敛更快)
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(hidden_states, self.norm.weight, self.norm.bias,
                                                            residual=residual,
                                                            prenorm=True,
                                                            residual_in_fp32=self.residual_in_fp32,
                                                            eps=self.norm.eps)
            else:
                hidden_states, residual = fused_add_norm_fn(self.drop_path(hidden_states), self.norm.weight,
                                                            self.norm.bias,
                                                            residual=residual,
                                                            prenorm=True,
                                                            residual_in_fp32=self.residual_in_fp32,
                                                            eps=self.norm.eps)
        # 计算出新的hidden_states
        hidden_states = self.mixer_cls(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_block(
        patch_embed_dim,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        drop_path_rate=0.0,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device="cuda",
        dtype=None,
        if_bimamba=False,  # 单向mamba
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}  # ssm自定义配置文件
    factory_kwargs = {'device': device, 'dtype': dtype}
    # partial函数用于创建一个部分应用的函数对象。部分应用是指在调用函数时，预先设置部分参数，然后返回一个新的函数，该函数接受剩余参数并返回结果。
    # 构建一个Mamba,可选ssm配置
    mix_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                      init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    # 构建一个LayerNorm，如果rms_norm设置为F时
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm,
                       eps=norm_epsilon, **factory_kwargs)
    # 构建块
    block = Block(
        layer_idx=layer_idx,
        patch_embed_dim=patch_embed_dim,
        mixer_cls=mix_cls,
        norm_cls=norm_cls,
        drop_path_rate=drop_path_rate,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    return block


# (1,in_channels,h,w) -> (1,in_channels,h,w)
class VimOutPic(nn.Module):
    def __init__(self,
                 img_size=120,
                 patch_size=5,
                 stride=5,
                 depth=24,
                 in_channels=3,
                 ssm_cfg=None,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 norm_epsilon=1e-5,
                 rms_norm=False,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 if_abs_pos_embed=False,  # PositionEmbedding是否用绝对位置编码
                 if_rope=False,  # 是否需要将某一维打乱
                 if_rope_residual=False,  # 是否需要将残差打乱
                 flip_img_sequences_ratio=-1.0,
                 if_bimamba=False,  # False为单向mamba
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 **kwargs
                 ):
        super(VimOutPic, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        kwargs.update(factory_kwargs)  # 追加参数
        self.img_size = to_2tuple(img_size)
        self.in_channels = in_channels
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.if_bidirectional = if_bidirectional
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        # patch_embedding
        # print(patch_size)
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=patch_size,
            stride=stride,
            in_channels=in_channels,
        )
        self.patch_embed_dim = self.patch_embed.embedding_dim
        self.un_patch_embed = UnPatchEmbedding(
            grid_size=self.patch_embed.grid_size,
            patch_size=patch_size,
            stride=stride,
            in_channels=self.patch_embed.in_channels,
            out_channels=self.patch_embed.in_channels,
        )
        self.patch_embed_patches = self.patch_embed.patches
        # position_embedding
        if if_abs_pos_embed:
            # 将——pos_embed设置为可学习参数
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed_patches, self.patch_embed_dim))
            self.pos_drop = nn.Dropout(drop_rate)
        # rope 旋转位置编码
        if if_rope:
            half_head_dim = self.patch_embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len,
            )
        # drop_path_out
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth - 1)]
        inter_dpr = [0.0] + dpr  # 保证前两层不失活 [0.0,0.0,0.01,...,0.1]
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # 构建块
        self.layers = nn.ModuleList([
            create_block(
                patch_embed_dim=self.patch_embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path_rate=inter_dpr[i],
                if_devide_out=if_devide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs
            ) for i in range(depth)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            self.patch_embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        # 调节参数满足正态分布
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        # print("after_patch",x.shape)
        B, M, _ = x.shape
        # 使用绝对位置编码
        if self.if_abs_pos_embed:
            if x.shape[1] != self.pos_embed.shape[1]:
                pos_embed = self.pos_embed.repeat((1, int(x.shape[1]/self.pos_embed.shape[1]), 1))
                x = x + pos_embed
            else:
                x = x + self.pos_embed
            # 遗忘部分位置编码，增大网络鲁棒性
            x = self.pos_drop(x)
        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])  # 第1维度反转
            if_flip_img_sequences = True
        residual, hidden_states = None, x
        if not self.if_bidirectional:
            # 单向传播
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:
                    # 翻转了序列，并且旋转位置编码
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)
                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
        else:
            # 双向传播
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)
                hidden_states_forward, residual_forward = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params)
                hidden_states_backward, residual_backward = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual is None else residual.flip([1]),
                    inference_params=inference_params)  # 反向传播时传入的数据需要flip，否则变为两次正向传播，无意义
                # 总和为两次传播之和，相加时注意反向传播的结果需要被flip回来
                hidden_states = hidden_states_forward + hidden_states_backward.flip([1])
                residual = residual_forward + residual_backward.flip([1])
        if not self.fused_add_norm:
            # 自定义正则化
            if residual is None:
                # 第一次进
                residual = hidden_states
            else:
                # 第一次后，残差等于自己加上部分被遗忘的隐藏状态
                residual = residual + self.drop_path(hidden_states)
            # 隐藏状态正则化
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # 使用fused_add_norm
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        unpack = hidden_states
        out = self.un_patch_embed(unpack)
        return out

    def forward(self, x, inference_params=None):
        mamba_out = self.forward_features(x, inference_params)
        return mamba_out


def test_patch_embedding_unpatch():
    in_channels = 3
    x = torch.randn(1, in_channels, 120, 120)
    p = PatchEmbedding(img_size=(120, 120), patch_size=5, stride=5, in_channels=in_channels)
    p.eval()
    o = p.forward(x)
    print(o.shape)
    no = torch.randn(1, 2304, 75)
    up = UnPatchEmbedding(grid_size=p.grid_size, patch_size=5, stride=5, in_channels=p.in_channels,
                          out_channels=p.in_channels)
    y = up.forward(no)
    print(y.shape)


def test_vim_out_pic():
    in_channels = 3
    img_size = 120
    device = "cuda"
    x = torch.randn(1, in_channels, img_size, img_size).to(device)
    print(x.shape)
    vop = VimOutPic(
        img_size=120,
        patch_size=4,
        stride=4,
        depth=32,
        in_channels=in_channels,
        drop_rate=0.1,
        drop_path_rate=0.1,
        device=device,
        flip_img_sequences_ratio=-1.0,
        if_bidirectional=True,
        if_abs_pos_embed=True
    ).to(device)
    total_params = sum(p.numel() for p in vop.parameters() if p.requires_grad)
    print(f'vim_out_pic总参数:{total_params / 1e6}M')
    if torch.cuda.is_available():
        print(f"Pytorch分配给GPU的显存开销: {torch.cuda.memory_allocated() // 1048576}MB")
    y = vop(x)
    print(y.shape)


if __name__ == '__main__':
    # test_patch_embedding_unpatch()
    test_vim_out_pic()
