import torch
import torch.nn as nn
from attentions import get_attention_layer


class InceptionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c = channels // 4
        self.branch1 = nn.Conv2d(channels, c, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, c, 1),
            nn.Conv2d(c, c, 3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, c, 1),
            nn.Conv2d(c, c, 5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, c, 1)
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
        return self.act(self.bn(out))


class SimVP(nn.Module):
    """
    Input shape [B, T_in, C_in, H, W].
    Output shape [B, T_out, C_out, H, W].
    """
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1,
                 time_step=24, predict_step=24, n_layers=2, attn_name="none"):
        super().__init__()
        self.time_step = time_step
        self.predict_step = predict_step
        self.output_dim = output_dim

        # Attention operates on each encoded frame before temporal concatenation.
        self.attn = get_attention_layer(attn_name, in_channels=hidden_dim)

        # 空间编码器：每帧独立处理
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, stride=2, padding=1),  # H/2, W/2
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # 时序建模：把T帧当作通道，用Inception并行处理
        self.temporal = nn.Sequential(*[
            InceptionModule(hidden_dim * time_step)
            for i in range(n_layers)
        ])

        # 时序映射：从T_in帧映射到T_out帧
        self.time_proj = nn.Conv2d(
            hidden_dim * time_step,
            hidden_dim * predict_step,
            1
        )

        # 空间解码器
        self.spatial_dec = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # 逐帧空间编码
        x_enc = x.reshape(B * T, C, H, W)
        x_enc = self.spatial_enc(x_enc)         # [B*T, hidden, H/2, W/2]

        # Per-frame attention uses [B*T, hidden, H/2, W/2].
        x_enc = self.attn(x_enc)

        _, hid, h2, w2 = x_enc.shape
        x_enc = x_enc.reshape(B, T * hid, h2, w2)  # [B, T*hidden, H/2, W/2]

        # 时序建模（完全并行）
        x_tmp = self.temporal(x_enc)            # [B, T*hidden, H/2, W/2]

        # （注意力只在逐帧阶段做一次，不再对拼接后的大张量再做一次）

        # 时序映射到预测帧数
        x_proj = self.time_proj(x_tmp)          # [B, T_out*hidden, H/2, W/2]
        x_proj = x_proj.reshape(
            B * self.predict_step, hid, h2, w2
        )

        # 逐帧空间解码
        out = self.spatial_dec(x_proj)          # [B*T_out, output_dim, H, W]
        out = out.reshape(B, self.predict_step, self.output_dim, H, W)
        return out
