import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .classification_utils import mark_danger_zone
from .correlation_utils import (
    find_largest_group_of_similar_ideas,
    select_features_to_keep,
)

MAPPING = {"NORMAL": 0, "DANGER_ZONE": 1, "MAINTENANCE": 0, "BROKEN": 1}


def detect_outliers(desc_stats):
    # Detect points over mean+-3std at outliers
    filtered_columns = desc_stats.columns[
        (desc_stats.loc["max"] > desc_stats.loc["mean"] + 3 * desc_stats.loc["std"])
        | (desc_stats.loc["min"] < desc_stats.loc["mean"] - 3 * desc_stats.loc["std"])
    ]
    return filtered_columns


def temporal_plot(df_data, state_dic, color_dic, var="sensor_29"):
    ts = df_data[var].values
    plt.plot(df_data[var].index, ts, label=var)
    already_labeled = set()
    for key, state in state_dic.items():
        if state in color_dic.keys():
            if state not in already_labeled:
                if (
                    int(key.split("_")[0]) in df_data[var].index
                    and int(key.split("_")[1]) in df_data[var].index
                ):
                    plt.fill_between(
                        np.arange(int(key.split("_")[0]), int(key.split("_")[1])),
                        y1=0,
                        y2=max(ts),
                        color=color_dic[state],
                        label=state,
                    )
                    already_labeled.add(state)
            else:
                if (
                    int(key.split("_")[0]) in df_data[var].index
                    and int(key.split("_")[1]) in df_data[var].index
                ):
                    plt.fill_between(
                        np.arange(int(key.split("_")[0]), int(key.split("_")[1])),
                        y1=0,
                        y2=max(ts),
                        color=color_dic[state],
                    )
    plt.title(f"Regime over time of {var}")
    plt.legend()
    plt.show()


def detect_cycles(df_data):
    y = df_data["machine_status"]
    # create a dictionnary of changing stats indexes
    state_changes_vec = y[y != y.shift(1)].index
    state_changes = {
        f"{state_changes_vec[i]}_{state_changes_vec[i+1]}": df_data.iloc[
            state_changes_vec[i]
        ].machine_status
        for i in range(len(state_changes_vec) - 1)
    }
    state_changes.update(
        {
            f"{state_changes_vec[-1]}_{len(df_data)}": df_data.iloc[
                state_changes_vec[-1]
            ].machine_status
        }
    )
    df_state_change = pd.DataFrame(state_changes.items(), columns=["IndexRange", "State"])
    df_state_change["StartIndex"] = df_state_change["IndexRange"].apply(
        lambda x: int(x.split("_")[0])
    )

    df_state_change["EndIndex"] = df_state_change["IndexRange"].apply(
        lambda x: int(x.split("_")[1])
    )

    df_state_change = df_state_change.drop(columns=["IndexRange"])

    # Compute mono machines status ranges
    df_state_change["Diff"] = df_state_change["StartIndex"].diff() / 60 / 24

    return df_state_change, state_changes


def cycle_train_test_split(df_data, df_state_change, cycles, cycle_idx=None):
    # Select test cycle
    if cycle_idx is None:
        test_cycle = random.choice(cycles)
    else:
        test_cycle = cycles[cycle_idx]
    df_state_change["Split"] = "Train"
    df_state_change.loc[test_cycle[0] : test_cycle[-1], "Split"] = "Test"
    test_range_start = int(df_state_change[df_state_change["Split"] == "Test"].StartIndex.min())
    test_range_end = int(df_state_change[df_state_change["Split"] == "Test"].EndIndex.max())
    # Add split to dataset
    df_data["Split"] = df_data.apply(
        lambda row: (
            "Test"
            if int(row.name) >= test_range_start and int(row.name) < test_range_end
            else "Train"
        ),
        axis=1,
    )
    # Add cycle number for later analysis
    df_data.loc[:, "Cycle"] = -1
    df_state_change.loc[:, "Cycle"] = -1
    for cycle_position, cycle in enumerate(cycles):
        df_state_change.loc[cycle[0] : cycle[-1], "Cycle"] = cycle_position
        start_idx = int(df_state_change.iloc[cycle[0] : cycle[-1]].StartIndex.min())
        end_idx = int(df_state_change.iloc[cycle[0] : cycle[-1] + 1].EndIndex.max())
        df_data.loc[start_idx:end_idx, "Cycle"] = cycle_position
    # Split data
    df_data_test = df_data[df_data["Split"] == "Test"]
    df_data_train = df_data[df_data["Split"] == "Train"]
    return df_data_train, df_data_test


def get_cycles(df_state_change):
    cycles = []
    # Dectect NORMAL-BROKEN-MAINTENANCE as cycle
    for i in range(len(df_state_change) - 2):
        if (
            df_state_change.iloc[i]["State"] == "NORMAL"
            and df_state_change.iloc[i + 1]["State"] == "BROKEN"
            and df_state_change.iloc[i + 2]["State"] == "MAINTENANCE"
        ):
            cycles.append(np.arange(i, i + 3))
    return cycles


def remove_missing_data(df_data, percentage_threshold=30):
    # Check for missing data
    missing_data = df_data.isnull().sum() / len(df_data) * 100
    column_to_drop = [name for name, val in missing_data.items() if val > percentage_threshold] + [
        col for col in df_data.keys() if "Unnamed" in col
    ]
    return df_data.drop(columns=column_to_drop), column_to_drop


def process_data(df_data, ALPHA_CORR=0.9, ALPHA_PREDICTIVE_STRENGH=1):
    df_data = df_data.drop(columns=[col for col in df_data.keys() if "Unnamed" in col])
    desc_stats = df_data[df_data["machine_status"].isin(["NORMAL"])].describe()
    filtered_columns = detect_outliers(desc_stats)

    df_state_change, state_changes = detect_cycles(df_data)
    cycles = get_cycles(df_state_change)

    df_data_train, df_data_test = cycle_train_test_split(
        df_data, df_state_change, cycles, cycle_idx=None
    )

    df_data_train, dropped_columns = remove_missing_data(df_data_train)
    non_feature_columns = [col for col in df_data_train.keys() if "sensor" not in col]
    df_data_train = df_data_train[
        ~df_data_train.drop(columns=non_feature_columns).isna().all(axis=1)
    ]

    corr_matrix = df_data_train.drop(columns=non_feature_columns).dropna().corr()
    g_similar_ideas = nx.from_pandas_adjacency(corr_matrix[corr_matrix >= ALPHA_CORR].fillna(0))
    cor_groups = find_largest_group_of_similar_ideas(g_similar_ideas, [], corr_matrix)
    no_group_ideas = list(g_similar_ideas.nodes)
    df_data_train = select_features_to_keep(df_data_train, cor_groups)

    df_data_train = mark_danger_zone(df_data_train, ALPHA_PREDICTIVE_STRENGH)
    df_data_test = mark_danger_zone(df_data_test, ALPHA_PREDICTIVE_STRENGH)

    df_data_train = df_data_train.copy()
    df_data_train.loc[:, "predictive_machine_status_label"] = df_data_train[
        "predictive_machine_status"
    ].map(MAPPING)
    df_data_test = df_data_test.copy()
    df_data_test.loc[:, "predictive_machine_status_label"] = df_data_test[
        "predictive_machine_status"
    ].map(MAPPING)

    non_feature_columns += ["predictive_machine_status", "predictive_machine_status_label"]

    return df_data_train, df_data_test, non_feature_columns
