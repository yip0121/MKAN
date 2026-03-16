import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.temporal = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.channel = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):  # [B, L, C]
        t = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + t
        x = x + self.channel(self.norm(x))
        return x


class Model(nn.Module):
    """TimeMixer-style baseline with temporal depthwise mixing + channel MLP."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.num_quantiles = len(configs.quantiles)

        channels = max(8, configs.d_model)
        layers = max(1, getattr(configs, 'time_mixer_layers', configs.e_layers))
        kernel = max(3, getattr(configs, 'time_mixer_kernel', 5))
        if kernel % 2 == 0:
            kernel += 1

        self.input_proj = nn.Linear(configs.enc_in, channels)
        self.blocks = nn.ModuleList([MixerBlock(channels, kernel, configs.dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, self.pred_len * self.num_quantiles)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.input_proj(x_enc)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feat = x[:, -1, :]
        pred = self.head(feat)
        return pred.view(pred.size(0), self.pred_len, self.num_quantiles)
