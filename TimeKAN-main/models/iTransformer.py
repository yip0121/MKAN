import torch
import torch.nn as nn


class Model(nn.Module):
    """Lightweight iTransformer-style baseline using TransformerEncoder."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.num_quantiles = len(configs.quantiles)

        d_model = configs.d_model
        n_heads = max(1, configs.n_heads)
        n_layers = max(1, getattr(configs, 'itf_layers', configs.e_layers))
        d_ff = max(d_model, configs.d_ff)

        self.input_proj = nn.Linear(configs.enc_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, configs.seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=configs.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.pred_len * self.num_quantiles)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.input_proj(x_enc)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        feat = x.mean(dim=1)
        pred = self.head(feat)
        return pred.view(pred.size(0), self.pred_len, self.num_quantiles)
