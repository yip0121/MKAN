import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Dataset_BatterySOH(Dataset):
    """Battery SOH dataset for MKAN/TimeKAN.

    Expected file format:
    - Column 1: cycle index
    - Column 4: SOH value (already precomputed)
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='battery.xlsx',
                 target='soh', prediction_target='absolute', scale=False, timeenc=0, freq='h', seasonal_patterns=None,
                 train_ratio=0.7, val_ratio=0.1):
        if size is None:
            self.seq_len = 20
            self.label_len = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.prediction_target = prediction_target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))

        if df_raw.shape[1] < 4:
            raise ValueError('Battery SOH file must contain at least 4 columns: cycle in col-1 and SOH in col-4.')

        df_battery = df_raw.iloc[:, [0, 3]].copy()
        df_battery.columns = ['cycle', 'soh']
        data = df_battery[['soh']].values.astype(np.float32)

        num_train = int(len(data) * self.train_ratio)
        num_vali = int(len(data) * self.val_ratio)
        num_test = len(data) - num_train - num_vali

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = data[border1:border2]
        if self.prediction_target == 'delta':
            data_delta = np.diff(data, axis=0, prepend=data[[0]])
            self.data_y = data_delta[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.prediction_target == 'delta':
            base = seq_x[-1:, :]
            seq_y = seq_y - base
        seq_x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
