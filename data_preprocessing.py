# Data Prepocessing

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
import sklearn,sklearn.model_selection
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.metrics import average_precision_score
from timm.scheduler import CosineLRScheduler
plt.style.use("ggplot")

from pyarrow.parquet import ParquetFile
import pyarrow as pa 
import ctypes


class PATHS:
    MAIN_DIR = "/kaggle/input/child-mind-institute-detect-sleep-states/"
    # CSV FILES : 
    SUBMISSION = MAIN_DIR + "sample_submission.csv"
    TRAIN_EVENTS = MAIN_DIR + "train_events.csv"
    # PARQUET FILES:
    TRAIN_SERIES = MAIN_DIR + "train_series.parquet"
    TEST_SERIES = MAIN_DIR + "test_series.parquet"


class CFG:
    DEMO_MODE = True

class data_reader:
    def __init__(self, demo_mode):
        super().__init__()
        # MAPPING FOR DATA LOADING :
        self.names_mapping = {
            "submission" : {"path" : PATHS.SUBMISSION, "is_parquet" : False, "has_timestamp" : False}, 
            "train_events" : {"path" : PATHS.TRAIN_EVENTS, "is_parquet" : False, "has_timestamp" : True},
            "train_series" : {"path" : PATHS.TRAIN_SERIES, "is_parquet" : True, "has_timestamp" : True},
            "test_series" : {"path" : PATHS.TEST_SERIES, "is_parquet" : True, "has_timestamp" : True}
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
        print("Percentage of removed steps : {:.1f}%".format(100 * (before_cleaning - after_cleaning) / before_cleaning) )
        return data
    
    @staticmethod
    def reduce_memory_usage(data):
        "iterate through all the columns of a dataframe and modify the data type to reduce memory usage."
        start_mem = data.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        for col in data.columns:
            col_type = data[col].dtype    
            if col_type != object:
                c_min = data[col].min()
                c_max = data[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        data[col] = data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)
            else:
                data[col] = data[col].astype('category')

        end_mem = data.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return data
    
    def load_data(self, data_name):
        "function for data loading"
        self.verify(data_name)
        data_props = self.names_mapping[data_name]
        if data_props["is_parquet"]:
            if self.demo_mode:
                pf = ParquetFile(data_props["path"]) 
                demo_steps = next(pf.iter_batches(batch_size=20_000)) 
                data = pa.Table.from_batches([demo_steps]).to_pandas()
            else:
                data = pd.read_parquet(data_props["path"])
        else:
            if self.demo_mode:
                data = pd.read_csv(data_props["path"], nsteps=20_000)
            else:
                data = pd.read_csv(data_props["path"])
                
        gc.collect()
        if data_props["has_timestamp"]:
            print('cleaning')
            data = self.cleaning(data)
            gc.collect()
        data = self.reduce_memory_usage(data)
        return data


reader = data_reader(demo_mode=False)
series = reader.load_data(data_name="train_series")
events = reader.load_data(data_name="train_events")


targets = []
data = []
ids = series.series_id.unique()

for viz_id in tqdm(ids):
    viz_targets = []
    viz_events = events[events.series_id == viz_id]
    viz_series = series.loc[(series.series_id==viz_id)].copy().reset_index()
    viz_series['dt'] = pd.to_datetime(viz_series.timestamp,format = '%Y-%m-%dT%H:%M:%S%z').astype("datetime64[ns, UTC-04:00]")
    viz_series['hour'] = viz_series['dt'].dt.hour

    check = 0
    for i in range(len(viz_events)-1):
        if viz_events.iloc[i].event =='onset' and viz_events.iloc[i+1].event =='wakeup' and viz_events.iloc[i].night==viz_events.iloc[i+1].night:
            start,end = viz_events.timestamp.iloc[i],viz_events.timestamp.iloc[i+1]

            start_id = viz_series.loc[viz_series.timestamp ==start].index.values[0]
            end_id = viz_series.loc[viz_series.timestamp ==end].index.values[0]
            viz_targets.append((start_id,end_id))
            check+=1
    targets.append(viz_targets)
    data.append(viz_series[['anglez','enmo','step']]) #you can include features like hour or minnute or second if you want to
joblib.dump((targets,data,ids), 'train_data.pkl')
len(data)


