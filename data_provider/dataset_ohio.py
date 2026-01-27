import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')


class Dataset_Ohio(Dataset):

    def __init__(self, root_path, flag='train', size=None,
                 features='S', target='cgm', scale=True,
                 timeenc=0, freq='5min', data_year='train_2018'):

        if size is None:
            self.seq_len = 288
            self.label_len = 144
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = os.path.join(root_path, data_year)

        self.train_subjects = ['559', '563', '570', '575']
        self.val_subjects = ['588']
        self.test_subjects = ['591']

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.set_type == 0:
            subjects = self.train_subjects
        elif self.set_type == 1:
            subjects = self.val_subjects
        else:
            subjects = self.test_subjects

        all_data = []
        for pid in subjects:
            csv_path = os.path.join(self.root_path, f"{pid}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"file not exist: {csv_path}")
            df = pd.read_csv(csv_path)
            df["pid"] = pid
            all_data.append(df)

        df_raw = pd.concat(all_data, ignore_index=True)

        assert self.target in df_raw.columns, f"no column {self.target}"
        if self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw.drop(columns=['datetime', 'pid'], errors='ignore')

        if self.scale:
            scaler_path = os.path.join(self.root_path, 'ohio_scaler_params.npz')
            if self.set_type == 0:
                self.scaler.fit(df_data.values)
                np.savez(scaler_path, mean=self.scaler.mean_, scale=self.scaler.scale_)
                data = self.scaler.transform(df_data.values)
            else:
                if os.path.exists(scaler_path):
                    params = np.load(scaler_path)
                    self.scaler.mean_ = params['mean']
                    self.scaler.scale_ = params['scale']
                    data = self.scaler.transform(df_data.values)
                else:
                    raise FileNotFoundError(f"no scaler params file: {scaler_path}")
        else:
            data = df_data.values

        if 'datetime' in df_raw.columns:
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
            df_stamp = df_raw[['datetime']]
        elif 'date' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            df_stamp = df_raw[['date']].rename(columns={'date': 'datetime'})
        else:
            raise ValueError("no datetime or date column")

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.datetime.apply(lambda r: r.month)
            df_stamp['day'] = df_stamp.datetime.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.datetime.apply(lambda r: r.weekday())
            df_stamp['hour'] = df_stamp.datetime.apply(lambda r: r.hour)
            df_stamp['minute'] = df_stamp.datetime.apply(lambda r: r.minute)
            df_stamp['minute'] = df_stamp['minute'] // 5
            data_stamp = df_stamp.drop(['datetime'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
