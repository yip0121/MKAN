import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.ChebyKANLayer import ChebyKANLinear
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(in_features, out_features, order)

    def forward(self, x):
        bsz, n_tokens, channels = x.shape
        x = self.fc1(x.reshape(bsz * n_tokens, channels))
        x = x.reshape(bsz, n_tokens, -1).contiguous()
        return x


class FrequencyDecomp(nn.Module):
    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.configs = configs

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
            ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class FrequencyMixing(nn.Module):
    def __init__(self, configs):
        super(FrequencyMixing, self).__init__()
        self.configs = configs
        self.front_block = M_KAN(
            configs.d_model,
            self.configs.seq_len // (self.configs.down_sampling_window ** self.configs.down_sampling_layers),
            order=configs.begin_order,
            dropout=configs.dropout,
        )

        self.front_blocks = torch.nn.ModuleList(
            [
                M_KAN(
                    configs.d_model,
                    self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
                    order=i + configs.begin_order + 1,
                    dropout=configs.dropout,
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
            ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class DilatedTCNBlock(nn.Module):
    """TCN-style dilated convolution block (same-length, residual)."""

    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(-1, -2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.transpose(-1, -2)
        return x + residual


class M_KAN(nn.Module):
    def __init__(self, d_model, seq_len, order, dropout=0.1):
        super().__init__()
        self.channel_mixer = nn.Sequential(ChebyKANLayer(d_model, d_model, order))
        dilation = max(1, 2 ** max(order - 1, 0))
        self.tcn_block = DilatedTCNBlock(d_model, dilation=dilation, dropout=dropout)

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.tcn_block(x)
        return x1 + x2


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.quantiles = list(configs.quantiles)
        self.num_quantiles = len(self.quantiles)

        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        self.res_blocks = nn.ModuleList([FrequencyDecomp(configs) for _ in range(configs.e_layers)])
        self.add_blocks = nn.ModuleList([FrequencyMixing(configs) for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(configs.d_model, self.num_quantiles, bias=True)
        self.predict_layer = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc):
        x_enc = self.__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc):
            bsz, t_steps, n_channels = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(bsz * n_channels, t_steps, 1)
            x_list.append(x)

        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.projection_layer(dec_out)
        dec_out = dec_out.reshape(bsz, n_channels, self.pred_len, self.num_quantiles)

        # SOH pipeline is single-channel. Keep the primary channel for quantile outputs.
        dec_out = dec_out[:, 0, :, :]

        # denorm each quantile with the same source normalization statistics
        dec_out = torch.cat(
            [self.normalize_layers[0](dec_out[:, :, q:q + 1], 'denorm') for q in range(self.num_quantiles)],
            dim=-1,
        )
        return dec_out

    def __multi_level_process_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]
        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        return x_enc_sampling_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc)
        raise ValueError('Other tasks implemented yet')
