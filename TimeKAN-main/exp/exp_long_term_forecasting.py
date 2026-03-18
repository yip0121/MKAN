import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, R2
import pandas as pd
import matplotlib.pyplot as plt

from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_with_interval

warnings.filterwarnings('ignore')
plt.switch_backend('agg')


class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        q = torch.tensor(quantiles, dtype=torch.float32)
        self.register_buffer('quantiles', q)

    def forward(self, pred, target):
        # pred: [B, T, Q], target: [B, T, 1]
        target = target.expand_as(pred)
        errors = target - pred
        q = self.quantiles.view(1, 1, -1)
        loss = torch.maximum(q * errors, (q - 1) * errors)
        return loss.mean()


class MSEOnMedianLoss(nn.Module):
    def __init__(self, median_idx):
        super().__init__()
        self.median_idx = int(median_idx)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_median = pred[:, :, self.median_idx:self.median_idx + 1]
        return self.mse(pred_median, target)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.quantiles = np.array(self.args.quantiles, dtype=np.float32)
        self.q_lower_idx = int(np.argmin(np.abs(self.quantiles - 0.05)))
        self.q_median_idx = int(np.argmin(np.abs(self.quantiles - 0.5)))
        self.q_upper_idx = int(np.argmin(np.abs(self.quantiles - 0.95)))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        loss_type = getattr(self.args, 'loss_type', 'mse').lower()
        if loss_type == 'pinball':
            criterion = PinballLoss(self.args.quantiles)
        elif loss_type == 'mse':
            criterion = MSEOnMedianLoss(self.q_median_idx)
        else:
            raise ValueError(f'Unsupported loss_type: {loss_type}')
        return criterion.to(self.device)

    def _build_decoder_input(self, batch_y):
        if self.args.down_sampling_layers == 0:
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            return dec_inp
        return None

    def _forward_model(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        dec_inp = self._build_decoder_input(batch_y)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def _point_estimate(self, pred_quantiles):
        return pred_quantiles[:, :, self.q_median_idx:self.q_median_idx + 1]

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._forward_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                total_loss.append(criterion(outputs, batch_y).item())

        self.model.train()
        return np.average(total_loss)

    def train(self, setting, save_checkpoint=True):
        _, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.project_root, self.args.checkpoints.lstrip('./').rstrip('/'), setting)
        if save_checkpoint and not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if save_checkpoint else None

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._forward_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            if save_checkpoint:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print('Early stopping')
                    break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        if save_checkpoint:
            best_model_path = path + '/checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def _test_single_step_loader(self, test_loader):
        preds_q = []
        trues = []
        bases = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._forward_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred_q = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                base = batch_x[:, -1:, :].detach().cpu().numpy()
                preds_q.append(pred_q)
                trues.append(true)
                bases.append(base)

        preds_q = np.array(preds_q).reshape(-1, preds_q[0].shape[-2], preds_q[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
        bases = np.array(bases).reshape(-1, bases[0].shape[-2], bases[0].shape[-1])
        return preds_q, trues, bases

    def _test_multi_step_direct(self, test_data):
        source_series = test_data.data_x.astype(np.float32).copy()

        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        label_plus_pred = self.args.label_len + pred_len

        preds_q = []
        trues = []
        bases = []

        stride = max(1, int(getattr(self.args, 'multi_step_stride', 1)))
        if getattr(self.args, 'eval_last_step_only', True) and pred_len > 1 and stride != 1:
            print('[Test] eval_last_step_only=True enforces stride=1 to provide dense, non-overlapping last-step timeline.')
            stride = 1
        start = 0
        with torch.no_grad():
            while start + seq_len + pred_len <= len(source_series):
                x_window = source_series[start:start + seq_len]
                y_true = source_series[start + seq_len:start + seq_len + pred_len]

                batch_x = torch.from_numpy(x_window).unsqueeze(0).float().to(self.device)
                batch_y = torch.zeros((1, label_plus_pred, x_window.shape[-1]), dtype=torch.float32).to(self.device)
                batch_x_mark = torch.zeros((1, seq_len, 1), dtype=torch.float32).to(self.device)
                batch_y_mark = torch.zeros((1, label_plus_pred, 1), dtype=torch.float32).to(self.device)

                outputs = self._forward_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred_q = outputs.detach().cpu().numpy()
                if pred_q.ndim == 3:
                    pred_q = pred_q[0]
                elif pred_q.ndim == 1:
                    pred_q = pred_q[:, None]

                preds_q.append(pred_q)
                trues.append(y_true)
                bases.append(x_window[-1:, :])

                start += stride

        preds_q = np.array(preds_q)
        trues = np.array(trues)
        bases = np.array(bases)
        print('test shape:', preds_q.shape, trues.shape)
        preds_q = preds_q.reshape(-1, preds_q.shape[-2], preds_q.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        bases = bases.reshape(-1, bases.shape[-2], bases.shape[-1])
        print('test shape:', preds_q.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds_q.shape
            preds_q = test_data.inverse_transform(preds_q.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        if len(preds_q) == 0:
            raise ValueError('Not enough samples for the current seq_len/pred_len in multi-step direct mode.')

        return preds_q, trues, bases


    def _save_dwt_decomposition_plot(self, test_data, result_folder):
        model_obj = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(model_obj, 'dwt_frontend'):
            return

        series = test_data.data_x.astype(np.float32)
        if len(series) < self.args.seq_len:
            return

        x_window = series[:self.args.seq_len]
        if x_window.ndim == 1:
            x_window = x_window[:, None]

        x_tensor = torch.from_numpy(x_window).unsqueeze(0).permute(0, 2, 1).to(self.device)
        with torch.no_grad():
            bands = model_obj.dwt_frontend(x_tensor)

        orig = x_window[:, 0]
        bands_np = [b.detach().cpu().numpy()[0, 0, :] for b in bands]

        nrows = len(bands_np) + 1
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 2.2 * nrows), sharex=True)
        axes[0].plot(orig, color='black', linewidth=1.5)
        axes[0].set_title('Original sequence (first channel)')

        for i, band in enumerate(bands_np):
            name = 'Low band (A)' if i == 0 else f'High band D{i}'
            axes[i + 1].plot(band, linewidth=1.2)
            axes[i + 1].set_title(name)

        plt.tight_layout()
        fig.savefig(os.path.join(result_folder, 'dwt_bands_overview.png'), bbox_inches='tight')
        plt.close(fig)

    def _save_kan_activation_plot(self, result_folder, num_points=200):
        model_obj = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(model_obj, 'band_blocks'):
            return

        bands = list(model_obj.band_blocks)
        if len(bands) == 0:
            return

        def _freq_label(i, total):
            if total <= 1:
                return 'Low', 'SEI生长'
            ratio = i / max(total - 1, 1)
            if ratio <= 1 / 3:
                return 'Low', 'SEI生长'
            if ratio <= 2 / 3:
                return 'Mid', '机械裂纹'
            return 'High', 'dead Li膝点'

        fig, axes = plt.subplots(len(bands), 1, figsize=(10, 3.0 * len(bands)), dpi=220, sharex=True)
        if len(bands) == 1:
            axes = [axes]

        plotted = 0
        for i, block in enumerate(bands):
            try:
                cheby_layer = block.channel_mixer[0].fc1
                x_plot, phi = cheby_layer.get_activation_curve(num_points=num_points)
            except Exception:
                continue

            freq, mech = _freq_label(i, len(bands))
            axes[i].plot(x_plot, phi, linewidth=1.8)
            axes[i].set_ylabel('phi(x)')
            axes[i].grid(alpha=0.25)
            axes[i].set_title(f'Band {i} (order={cheby_layer.degree}) - {freq} freq | {mech}')
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            return

        axes[-1].set_xlabel('x')
        plt.tight_layout()
        save_path = os.path.join(result_folder, 'kan_activation_functions.png')
        fig.savefig(save_path, dpi=320, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _restore_from_delta(preds_q, trues, bases, true_is_delta):
        base_q = bases[:, :1, :1]
        preds_abs = preds_q + base_q
        if true_is_delta:
            trues_abs = trues + base_q
        else:
            trues_abs = trues
        return preds_abs, trues_abs

    @staticmethod
    def _restore_from_delta(preds_q, trues, bases, true_is_delta):
        base_q = bases[:, :1, :1]
        preds_abs = preds_q + base_q
        if true_is_delta:
            trues_abs = trues + base_q
        else:
            trues_abs = trues
        return preds_abs, trues_abs

    @staticmethod
    def _save_array_csv(array, path):
        flat = array.reshape(array.shape[0], -1)
        pd.DataFrame(flat).to_csv(path, index=False)

    def test(self, setting, test=0, save_outputs=True, save_dwt_plot=True):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.project_root, 'checkpoints', setting, 'checkpoint.pth')))

        self.model.eval()
        if self.args.pred_len == 1:
            print('[Test] Single-step mode active (pred_len=1, sliding window stride=1).')
            preds_q, trues, bases = self._test_single_step_loader(test_loader)
            true_is_delta = True
        else:
            print(f'[Test] Multi-step direct mode active (pred_len={self.args.pred_len}, stride={self.args.multi_step_stride}, true-history input).')
            preds_q, trues, bases = self._test_multi_step_direct(test_data)
            true_is_delta = False

        if self.args.prediction_target == 'delta':
            preds_q, trues = self._restore_from_delta(preds_q, trues, bases, true_is_delta=true_is_delta)
            print('[Test] prediction_target=delta: restored predictions/truth to absolute SOH scale before evaluation.')

        if self.args.eval_last_step_only and preds_q.shape[1] > 1:
            preds_q = preds_q[:, -1:, :]
            trues = trues[:, -1:, :]
            print('[Test] eval_last_step_only=True: keeping only the final step per forecast window for metrics/plots.')

        if getattr(self.args, 'clip_predictions', False):
            observed = test_data.data_x.astype(np.float32)
            clip_low = float(np.min(observed) - float(getattr(self.args, 'clip_margin', 0.0)))
            clip_high = float(np.max(observed) + float(getattr(self.args, 'clip_margin', 0.0)))
            preds_q = np.clip(preds_q, clip_low, clip_high)
            print(f'[Test] Clipped predictions to [{clip_low:.4f}, {clip_high:.4f}] for stability.')

        preds = preds_q[:, :, self.q_median_idx:self.q_median_idx + 1]
        pred_upper = preds_q[:, :, self.q_upper_idx:self.q_upper_idx + 1]
        pred_lower = preds_q[:, :, self.q_lower_idx:self.q_lower_idx + 1]

        print('test shape:', preds_q.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        r2 = R2(preds, trues)
        print(f'mse:{mse}, mae:{mae}')
        print(f'rmse:{rmse}, mape:{mape}, mspe:{mspe}, r2:{r2}')

        metrics_payload = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'r2': float(r2),
        }

        if not save_outputs:
            return metrics_payload

        result_folder = os.path.join(self.args.project_root, 'results', setting) + os.sep
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        with open(os.path.join(self.args.project_root, 'result_long_term_forecast.txt'), 'a') as f:
            f.write(setting + '  \n')
            f.write(f'mse:{mse}, mae:{mae}, r2:{r2}')
            f.write('\n\n')

        pd.DataFrame([metrics_payload]).to_csv(os.path.join(result_folder, 'metrics.csv'), index=False)
        self._save_array_csv(preds, os.path.join(result_folder, 'pred.csv'))
        self._save_array_csv(preds_q, os.path.join(result_folder, 'pred_quantiles.csv'))
        self._save_array_csv(trues, os.path.join(result_folder, 'true.csv'))

        true_series = trues[:, :, -1].reshape(-1)
        pred_series = preds[:, :, -1].reshape(-1)
        lower_series = pred_lower[:, :, -1].reshape(-1)
        upper_series = pred_upper[:, :, -1].reshape(-1)

        visual_with_interval(
            true_series,
            pred_series,
            lower_series,
            upper_series,
            os.path.join(result_folder, 'prediction_vs_truth_with_interval.png')
        )
        visual(true_series, pred_series, os.path.join(result_folder, 'prediction_vs_truth.png'))

        pd.DataFrame(
            {
                'true': true_series,
                'pred_median_q0.50': pred_series,
                f'pred_lower_q{self.quantiles[self.q_lower_idx]:.2f}': lower_series,
                f'pred_upper_q{self.quantiles[self.q_upper_idx]:.2f}': upper_series,
            }
        ).to_csv(os.path.join(result_folder, 'prediction_vs_truth.csv'), index=False)

        self._save_kan_activation_plot(result_folder)

        if save_dwt_plot:
            self._save_dwt_decomposition_plot(test_data, result_folder)

        print('Saved visualization to:', os.path.join(result_folder, 'prediction_vs_truth.png'))
        print('Saved interval visualization to:', os.path.join(result_folder, 'prediction_vs_truth_with_interval.png'))
        print('Saved prediction csv to:', os.path.join(result_folder, 'prediction_vs_truth.csv'))
        print('Saved metrics to:', os.path.join(result_folder, 'metrics.csv'))
        print('Saved quantiles to:', os.path.join(result_folder, 'pred_quantiles.csv'))
        print('Saved KAN activation plot to:', os.path.join(result_folder, 'kan_activation_functions.png'))
        if save_dwt_plot:
            print('Saved DWT band overview to:', os.path.join(result_folder, 'dwt_bands_overview.png'))
        return metrics_payload
