import torch
import torch.nn as nn


class Model(nn.Module):
    """CNN + LSTM baseline for univariate/multivariate sequence forecasting."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.num_quantiles = len(configs.quantiles)
        in_channels = configs.enc_in
        cnn_channels = getattr(configs, 'cnn_channels', max(8, configs.d_model))
        hidden_size = getattr(configs, 'lstm_hidden', max(16, configs.d_model))
        lstm_layers = getattr(configs, 'lstm_layers', max(1, configs.e_layers))

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=configs.dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, self.pred_len * self.num_quantiles)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, L, C]
        x = x_enc.transpose(1, 2)  # [B, C, L]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, L, C']
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        pred = self.head(feat)
        return pred.view(pred.size(0), self.pred_len, self.num_quantiles)
