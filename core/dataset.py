import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset


def create_time_series_dataset(df, time_steps=60):
    df["N_TS"] = 0
    for i in tqdm.tqdm(range(time_steps, len(df) + 1)):
        df.loc[i - time_steps : i, "N_TS"] = i // time_steps
    return df


def time_diff(df_data):
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"], format="%m/%d/%Y %H:%M")
    df_data["timestamp_diff_minutes"] = df_data["timestamp"].diff().dt.total_seconds() / 60
    return df_data


def create_ts_dataset(df_data, time_steps):
    X = []
    y = []
    vars = [x for x in df_data.keys() if "sensor" in x]
    for i in tqdm.tqdm(range(1, max(df_data.N_TS.unique()))):
        if (
            np.sum(df_data[df_data["N_TS"] == i].timestamp_diff_minutes.values)
            == len(df_data[df_data["N_TS"] == i])
            and len(df_data[df_data["N_TS"] == i]) == time_steps
        ):
            X.append(
                torch.from_numpy(df_data[df_data["N_TS"] == i][vars].to_numpy().astype("float32"))
            )
            y.append(
                torch.tensor(
                    df_data[df_data["N_TS"] == i]["predictive_machine_status_label"].min(),
                    dtype=torch.int,
                )
            )
    return X, y


class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.y[idx]
