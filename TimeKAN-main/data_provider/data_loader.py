import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Dataset_BatterySOH(Dataset):
    """Battery SOH dataset for MKAN/TimeKAN and compare baselines."""

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='battery.xlsx',
                 target='soh', scale=False, timeenc=0, freq='h', seasonal_patterns=None,
                 train_ratio=0.7, val_ratio=0.1, target_col=None, target_col_idx=-1):
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
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_col = target_col
        self.target_col_idx = target_col_idx

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))

        if self.target_col is not None and self.target_col != '':
            if self.target_col not in df_raw.columns:
                raise ValueError(f"target_col '{self.target_col}' not in columns: {list(df_raw.columns)}")
            target_series = df_raw[self.target_col]
            selected_name = self.target_col
        elif self.target_col_idx is not None and int(self.target_col_idx) >= 0:
            col_idx = int(self.target_col_idx)
            if col_idx >= df_raw.shape[1]:
                raise ValueError(f'target_col_idx {col_idx} out of range for shape {df_raw.shape}')
            selected_name = str(df_raw.columns[col_idx])
            target_series = df_raw.iloc[:, col_idx]
        else:
            # Backward-compatible default: use 4th column if available, else last column.
            if df_raw.shape[1] < 4:
                selected_name = str(df_raw.columns[-1])
                target_series = df_raw.iloc[:, -1]
                print('[Data] Warning: less than 4 columns found, fallback to last column as target.')
            else:
                selected_name = str(df_raw.columns[3])
                target_series = df_raw.iloc[:, 3]

        data = target_series.values.astype(np.float32).reshape(-1, 1)
        print(f'[Data] Target column in use: {selected_name}')

        num_train = int(len(data) * self.train_ratio)
        num_vali = int(len(data) * self.val_ratio)
        num_test = len(data) - num_train - num_vali

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
