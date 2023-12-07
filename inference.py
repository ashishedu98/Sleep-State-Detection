import pandas as pd
import numpy as np
import gc
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import joblib
import random
import math
from tqdm.auto import tqdm

from scipy.interpolate import interp1d

from math import pi, sqrt, exp
import sklearn, sklearn.model_selection
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.metrics import average_precision_score
from timm.scheduler import CosineLRScheduler

plt.style.use("ggplot")

from pyarrow.parquet import ParquetFile
import pyarrow as pa
import ctypes


class PATHS:
    MAIN_DIR = "/kaggle/input/d/harshshelar/custom/"
    # CSV FILES :
    SUBMISSION = MAIN_DIR + "sample_submission.csv"
    TRAIN_EVENTS = MAIN_DIR + "train_event_data.csv"
    # PARQUET FILES:
    TRAIN_SERIES = (
        "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet"
    )
    TEST_SERIES = MAIN_DIR + "test_series.parquet"


class CFG:
    DEMO_MODE = True


class data_reader:
    def __init__(self, demo_mode):
        super().__init__()
        # MAPPING FOR DATA LOADING :
        self.names_mapping = {
            "submission": {
                "path": PATHS.SUBMISSION,
                "is_parquet": False,
                "has_timestamp": False,
            },
            "train_events": {
                "path": PATHS.TRAIN_EVENTS,
                "is_parquet": False,
                "has_timestamp": True,
            },
            "train_series": {
                "path": PATHS.TRAIN_SERIES,
                "is_parquet": True,
                "has_timestamp": True,
            },
            "test_series": {
                "path": PATHS.TEST_SERIES,
                "is_parquet": True,
                "has_timestamp": True,
            },
        }
        self.valid_names = ["submission", "train_events", "train_series", "test_series"]
        self.demo_mode = demo_mode

    def verify(self, data_name):
        "function for data name verification"
        if data_name not in self.valid_names:
            print("PLEASE ENTER A VALID DATASET NAME, VALID NAMES ARE : ", valid_names)
        return

    def cleaning(self, data):
        "cleaning function : drop na values"
        before_cleaning = len(data)
        print("Number of missing timestamps : ", len(data[data["timestamp"].isna()]))
        data = data.dropna(subset=["timestamp"])
        after_cleaning = len(data)
        print(
            "Percentage of removed rows : {:.1f}%".format(
                100 * (before_cleaning - after_cleaning) / before_cleaning
            )
        )
        #         print(data.isna().any())
        #         data = data.bfill()
        return data

    @staticmethod
    def reduce_memory_usage(data):
        "iterate through all the columns of a dataframe and modify the data type to reduce memory usage."
        start_mem = data.memory_usage().sum() / 1024**2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
        for col in data.columns:
            col_type = data[col].dtype
            if col_type != object:
                c_min = data[col].min()
                c_max = data[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        data[col] = data[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        data[col] = data[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        data[col] = data[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        data[col] = data[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)
            else:
                data[col] = data[col].astype("category")

        end_mem = data.memory_usage().sum() / 1024**2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
        return data

    def load_data(self, data_name):
        "function for data loading"
        self.verify(data_name)
        data_props = self.names_mapping[data_name]
        if data_props["is_parquet"]:
            if self.demo_mode:
                pf = ParquetFile(data_props["path"])
                demo_rows = next(pf.iter_batches(batch_size=20_000))
                data = pa.Table.from_batches([demo_rows]).to_pandas()
            else:
                data = pd.read_parquet(data_props["path"])
        else:
            if self.demo_mode:
                data = pd.read_csv(data_props["path"], nrows=20_000)
            else:
                data = pd.read_csv(data_props["path"])

        gc.collect()
        if data_props["has_timestamp"]:
            print("cleaning")
            data = self.cleaning(data)
            gc.collect()
        data = self.reduce_memory_usage(data)
        return data


reader = data_reader(demo_mode=False)
test_series = reader.load_data(data_name="test_series")
ids = test_series.series_id.unique()
gc.collect()


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(hidden_size * dir_factor, hidden_size * dir_factor * 2)
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h


class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers, bidir=True):
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        if h is None:
            h = [None for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)

        x = self.fc_out(x)
        return x, new_h


class SleepDataset(Dataset):
    def __init__(
        self,
        series_ids,
        series,
    ):
        series_ids = series_ids
        series = series.reset_index()
        self.data = []

        for viz_id in tqdm(series_ids):
            self.data.append(
                series.loc[(series.series_id == viz_id)].copy().reset_index()
            )

    def downsample_seq_generate_features(self, feat, downsample_factor):
        if len(feat) % 12 != 0:
            feat = np.concatenate([feat, np.zeros(12 - ((len(feat)) % 12)) + feat[-1]])
        feat = np.reshape(feat, (-1, 12))
        feat_mean = np.mean(feat, 1)
        feat_std = np.std(feat, 1)
        feat_median = np.median(feat, 1)
        feat_max = np.max(feat, 1)
        feat_min = np.min(feat, 1)

        return np.dstack([feat_mean, feat_std, feat_median, feat_max, feat_min])[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index][["anglez", "enmo"]].values.astype(np.float32)
        X = np.concatenate(
            [
                self.downsample_seq_generate_features(X[:, i], 12)
                for i in range(X.shape[1])
            ],
            -1,
        )
        X = torch.from_numpy(X)
        return X


test_ds = SleepDataset(test_series.series_id.unique(), test_series)
del test_series
gc.collect()

max_chunk_size = 24 * 60 * 100
min_interval = 30

model = (
    MultiResidualBiGRU(input_size=10, hidden_size=64, out_size=2, n_layers=5)
    .to(device)
    .eval()
)
model.load_state_dict(
    torch.load(
        f"/kaggle/input/sleep-critical-point-train/model_best.pth", map_location=device
    )
)
submission = pd.DataFrame()
for i in range(len(test_ds)):
    X = test_ds[i].half()
    seq_len = X.shape[0]
    h = None
    pred = torch.zeros((len(X), 2)).half()
    for j in range(0, seq_len, max_chunk_size):
        y_pred, h = model(X[j : j + max_chunk_size].float(), h)
        h = [hi.detach() for hi in h]
        pred[j : j + max_chunk_size] = y_pred.detach()
        del y_pred
        gc.collect()
    del h, X
    gc.collect()
    pred = pred.numpy()
    series_id = ids[i]

    days = len(pred) / (17280 / 12)
    scores0, scores1 = np.zeros(len(pred), dtype=np.float16), np.zeros(
        len(pred), dtype=np.float16
    )
    for index in range(len(pred)):
        if pred[index, 0] == max(
            pred[max(0, index - min_interval) : index + min_interval, 0]
        ):
            scores0[index] = max(
                pred[max(0, index - min_interval) : index + min_interval, 0]
            )
        if pred[index, 1] == max(
            pred[max(0, index - min_interval) : index + min_interval, 1]
        ):
            scores1[index] = max(
                pred[max(0, index - min_interval) : index + min_interval, 1]
            )
    candidates_onset = np.argsort(scores0)[-max(1, round(days)) :]
    candidates_wakeup = np.argsort(scores1)[-max(1, round(days)) :]

    onset = (
        test_ds.data[i][["step"]]
        .iloc[np.clip(candidates_onset * 12, 0, len(test_ds.data[i]) - 1)]
        .astype(np.int32)
    )
    onset["event"] = "onset"
    onset["series_id"] = series_id
    onset["score"] = scores0[candidates_onset]
    wakeup = (
        test_ds.data[i][["step"]]
        .iloc[np.clip(candidates_wakeup * 12, 0, len(test_ds.data[i]) - 1)]
        .astype(np.int32)
    )
    wakeup["event"] = "wakeup"
    wakeup["series_id"] = series_id
    wakeup["score"] = scores1[candidates_wakeup]
    submission = pd.concat([submission, onset, wakeup], axis=0)
    del (
        onset,
        wakeup,
        candidates_onset,
        candidates_wakeup,
        scores0,
        scores1,
        pred,
        series_id,
    )
    gc.collect()
submission = submission.sort_values(["series_id", "step"]).reset_index(drop=True)
submission["row_id"] = submission.index.astype(int)
submission["score"] = submission["score"].fillna(submission["score"].mean())
submission = submission[["row_id", "series_id", "step", "event", "score"]]
submission.to_csv("submission.csv", index=False)
