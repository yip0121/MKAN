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
        if len(level_list) <= 1:
            return level_list

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
        if len(level_list) <= 1:
            return [self.front_block(level_list[0])]

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




class HaarDWTFrontEnd(nn.Module):
    """Multi-level per-channel Haar DWT with single-band reconstruction to original length."""

    def __init__(self, levels: int):
        super().__init__()
        self.levels = max(0, int(levels))
        self.sqrt2 = 2 ** 0.5

    def _dwt_step(self, x):
        # x: [B, C, T]
        orig_len = x.size(-1)
        if orig_len % 2 == 1:
            x = torch.cat([x, x[..., -1:]], dim=-1)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        approx = (even + odd) / self.sqrt2
        detail = (even - odd) / self.sqrt2
        return approx, detail, orig_len

    def _idwt_step(self, approx, detail, target_len):
        # approx/detail: [B, C, T/2]
        even = (approx + detail) / self.sqrt2
        odd = (approx - detail) / self.sqrt2
        out = torch.stack([even, odd], dim=-1).reshape(approx.size(0), approx.size(1), -1)
        return out[..., :target_len]

    def _decompose(self, x):
        details = []
        lengths = []
        approx = x
        for _ in range(self.levels):
            approx, detail, orig_len = self._dwt_step(approx)
            details.append(detail)
            lengths.append(orig_len)
        return approx, details, lengths

    def _reconstruct_single_band(self, approx, details, lengths, band_idx):
        # band_idx=0 -> low band (approx at deepest level)
        # band_idx=i (1..levels) -> detail band at level i (coarse-to-fine mapped by reverse index)
        if band_idx == 0:
            rec = approx
            for lvl in reversed(range(self.levels)):
                zero_d = torch.zeros_like(details[lvl])
                rec = self._idwt_step(rec, zero_d, lengths[lvl])
            return rec

        detail_pick = self.levels - band_idx
        rec = torch.zeros_like(approx)
        for lvl in reversed(range(self.levels)):
            d = details[lvl] if lvl == detail_pick else torch.zeros_like(details[lvl])
            rec = self._idwt_step(rec, d, lengths[lvl])
        return rec

    def forward(self, x):
        # x: [B, C, T]
        if self.levels <= 0:
            return [x]
        approx, details, lengths = self._decompose(x)
        bands = [self._reconstruct_single_band(approx, details, lengths, 0)]
        for i in range(1, self.levels + 1):
            bands.append(self._reconstruct_single_band(approx, details, lengths, i))
        return bands

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

        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        self.wavelet_levels = max(0, int(configs.down_sampling_layers))
        self.num_bands = self.wavelet_levels + 1
        self.dwt_frontend = HaarDWTFrontEnd(self.wavelet_levels)

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.normalize = Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)

        # Low-frequency band keeps lower order; higher-frequency bands use higher order.
        self.band_blocks = nn.ModuleList([
            M_KAN(configs.d_model, configs.seq_len, order=configs.begin_order + i, dropout=configs.dropout)
            for i in range(self.num_bands)
        ])

        self.projection_layer = nn.Linear(configs.d_model, self.num_quantiles, bias=True)
        self.predict_layer = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc):
        # x_enc: [B, T, C]
        bsz, _, n_channels = x_enc.size()
        x_norm = self.normalize(x_enc, 'norm')

        # DWT + single-band reconstruction per channel in [B, C, T]
        bands = self.dwt_frontend(x_norm.permute(0, 2, 1).contiguous())

        band_outs = []
        for i, band in enumerate(bands):
            # band: [B, C, T] -> [B*C, T, 1]
            band_in = band.permute(0, 2, 1).contiguous().reshape(bsz * n_channels, self.seq_len, 1)
            emb = self.enc_embedding(band_in, None)
            band_outs.append(self.band_blocks[i](emb))

        # Replace old CFD fusion with simple band-wise sum, then prediction head
        fused = torch.stack(band_outs, dim=0).sum(dim=0)

        dec_out = self.predict_layer(fused.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.projection_layer(dec_out)
        dec_out = dec_out.reshape(bsz, n_channels, self.pred_len, self.num_quantiles)

        # SOH pipeline is single-channel. Keep the primary channel for quantile outputs.
        dec_out = dec_out[:, 0, :, :]

        # denorm each quantile with the same source normalization statistics
        dec_out = torch.cat(
            [self.normalize(dec_out[:, :, q:q + 1], 'denorm') for q in range(self.num_quantiles)],
            dim=-1,
        )
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc)
        raise ValueError('Other tasks implemented yet')
