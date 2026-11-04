# attentions.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.mlp(avg_out).view(b, c, 1, 1)
        max_out = self.mlp(max_out).view(b, c, 1, 1)
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(cat))
        return x * attn


class CBAM(nn.Module):
    """CBAM 通道+空间注意力，输入 [B,C,H,W]"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class SimAM(nn.Module):
    """无参轻量注意力，输入 [B,C,H,W]"""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True)/n + self.e_lambda)) + 0.5
        return x * y.sigmoid()


# 工厂函数：消融实验
def get_attention_layer(attn_name: str, in_channels: int):
    attn_name = attn_name.lower()
    if attn_name == "none":
        return nn.Identity()
    elif attn_name == "cbam":
        return CBAM(in_channels=in_channels)
    elif attn_name == "simam":
        return SimAM()
    else:
        raise ValueError(f"not support attn_name: {attn_name}")
