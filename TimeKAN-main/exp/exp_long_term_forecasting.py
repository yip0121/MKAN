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
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
        return nn.MSELoss()

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

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

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
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    def _test_single_step_loader(self, test_loader):
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._forward_model(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
        return preds, trues

    def _test_multi_step_autoregressive(self, test_data):
        """Non-overlapping multi-step evaluation with recursive input update.

        When pred_len > 1:
        - stride = pred_len
        - next prediction starts from previous prediction end
        - newly predicted values are fed back as future input
        """
        source_series = test_data.data_x.astype(np.float32).copy()
        rolling_series = source_series.copy()

        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        label_plus_pred = self.args.label_len + pred_len

        preds = []
        trues = []

        start = 0
        chunk_idx = 0
        with torch.no_grad():
            while start + seq_len + pred_len <= len(source_series):
                x_window = rolling_series[start:start + seq_len]
                y_true = source_series[start + seq_len:start + seq_len + pred_len]

                batch_x = torch.from_numpy(x_window).unsqueeze(0).float().to(self.device)
                batch_y = torch.zeros((1, label_plus_pred, x_window.shape[-1]), dtype=torch.float32).to(self.device)
                batch_x_mark = torch.zeros((1, seq_len, 1), dtype=torch.float32).to(self.device)
                batch_y_mark = torch.zeros((1, label_plus_pred, 1), dtype=torch.float32).to(self.device)

                outputs = self._forward_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                if pred.ndim == 3:
                    pred = pred[0]
                elif pred.ndim == 1:
                    pred = pred[:, None]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(y_true)

                # Feed predictions back as future inputs
                rolling_series[start + seq_len:start + seq_len + pred_len] = pred

                # Next block starts from previous block end (non-overlap)
                start += pred_len
                chunk_idx += 1

        if len(preds) == 0:
            raise ValueError('Not enough test samples for the current seq_len/pred_len in multi-step mode.')

        return np.array(preds), np.array(trues)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        if self.args.pred_len == 1:
            print('[Test] Single-step mode active (pred_len=1).')
            preds, trues = self._test_single_step_loader(test_loader)
        else:
            print(f'[Test] Multi-step autoregressive mode active (pred_len={self.args.pred_len}, stride={self.args.pred_len}).')
            preds, trues = self._test_multi_step_autoregressive(test_data)

        print('test shape:', preds.shape, trues.shape)

        result_folder = './results/' + setting + '/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        r2 = R2(preds, trues)
        print(f'mse:{mse}, mae:{mae}')
        print(f'rmse:{rmse}, mape:{mape}, mspe:{mspe}, r2:{r2}')

        with open('result_long_term_forecast.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write(f'mse:{mse}, mae:{mae}, r2:{r2}')
            f.write('\n\n')

        np.save(result_folder + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(result_folder + 'pred.npy', preds)
        np.save(result_folder + 'true.npy', trues)

        true_series = trues[:, :, -1].reshape(-1)
        pred_series = preds[:, :, -1].reshape(-1)
        visual(true_series, pred_series, os.path.join(result_folder, 'prediction_vs_truth.png'))
        save_to_csv(true_series, pred_series, os.path.join(result_folder, 'prediction_vs_truth.csv'))
        print('Saved visualization to:', os.path.join(result_folder, 'prediction_vs_truth.png'))
        print('Saved prediction csv to:', os.path.join(result_folder, 'prediction_vs_truth.csv'))
