from torch import nn
from triton.interpreter.interpreter import torch

from models.modules.convs.cga import SpatialAttention, ChannelAttention, PixelAttention


# 特征融合
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


def test_cga_fusion():
    x = torch.randn((1, 12, 32, 32)).to("cuda")
    y = torch.randn((1, 12, 32, 32)).to("cuda")
    f = CGAFusion(dim=12,reduction=4).to("cuda")
    result = f(x, y)
    print(result.shape)