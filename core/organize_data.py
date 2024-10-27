import random

import networkx as nx
import numpy as np
import pandas as pd

PCA_N_COMPONENT = 12
GROUP_MIN_SIMILARITY = 0.8
ALPHA_PREDICTIVE_STRENGH = 2

label_mapping = {"DANGER_ZONE": 0, "NORMAL": 1, "MAINTENANCE": 2, "BROKEN": 3}


def get_cycles(df_state_change):
    cycles = []
    for i in range(len(df_state_change) - 2):
        if (
            df_state_change.iloc[i]["State"] == "NORMAL"
            and df_state_change.iloc[i + 1]["State"] == "BROKEN"
            and df_state_change.iloc[i + 2]["State"] == "MAINTENANCE"
        ):
            cycles.append(np.arange(i, i + 3))
    return cycles


def compute_group_mean_similarity(similar_ideas, df_ideas_cos_sim):
    df_tmp = df_ideas_cos_sim.filter(items=similar_ideas, axis=1)
    df_tmp = df_tmp.filter(items=similar_ideas, axis=0)
    return df_tmp.sum().sum() / (df_tmp.shape[0] * (df_tmp.shape[0] - 1))


def find_largest_group_of_similar_ideas(g_similar_ideas, idea_groups, df_cor):
    new_idea_groups = idea_groups.copy()

    similar_ideas_cliques = list(nx.find_cliques(g_similar_ideas))
    cliques_ordered_by_size_desc = sorted(similar_ideas_cliques, key=len, reverse=True)
    clique_number = len(cliques_ordered_by_size_desc[0])
    if clique_number < 2:
        return new_idea_groups

    largest_max_cliques = list(
        filter(lambda c: len(c) == clique_number, cliques_ordered_by_size_desc)
    )
    largest_max_cliques_count = len(largest_max_cliques)

    if largest_max_cliques_count > 1:
        largest_max_cliques_sorted = sorted(
            largest_max_cliques,
            key=lambda c: compute_group_mean_similarity(c, df_cor),
            reverse=True,
        )
        similar_ideas = sorted(largest_max_cliques_sorted[0])
    else:
        similar_ideas = sorted(largest_max_cliques[0])

    new_idea_groups.append(similar_ideas)
    g_similar_ideas.remove_nodes_from(similar_ideas)
    if len(g_similar_ideas) < 1:
        return new_idea_groups

    return find_largest_group_of_similar_ideas(g_similar_ideas, new_idea_groups, df_cor)


def organize_data(df_data):
    y = df_data["machine_status"]
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
    df_state_change["DIff"] = df_state_change["StartIndex"].diff() / 60 / 24
    cycles = get_cycles(df_state_change)

    test_cycle = random.choice(cycles)
    df_state_change["Split"] = "Train"
    df_state_change.loc[test_cycle[0] : test_cycle[-1], "Split"] = "Test"

    test_range_start = int(df_state_change[df_state_change["Split"] == "Test"].StartIndex.min())

    test_range_end = int(df_state_change[df_state_change["Split"] == "Test"].EndIndex.max())

    df_data["Split"] = df_data.apply(
        lambda row: (
            "Test"
            if int(row.name) >= test_range_start and int(row.name) <= test_range_end
            else "Train"
        ),
        axis=1,
    )

    df_data.loc[:, "Cycle"] = -1

    df_state_change.loc[:, "Cycle"] = -1

    for cycle_position, cycle in enumerate(cycles):
        df_state_change.loc[cycle[0] : cycle[-1], "Cycle"] = cycle_position
        start_idx = int(df_state_change.iloc[cycle[0] : cycle[-1]].StartIndex.min())
        end_idx = int(df_state_change.iloc[cycle[0] : cycle[-1] + 1].EndIndex.max())
        df_data.loc[start_idx:end_idx, "Cycle"] = cycle_position

    df_data_test = df_data[df_data["Split"] == "Test"]
    df_data = df_data[df_data["Split"] == "Train"]

    missing_data = df_data.isnull().sum() / len(df_data) * 100

    column_to_drop = [name for name, val in missing_data.items() if val > 30] + [
        col for col in df_data.keys() if "Unnamed" in col
    ]
    df_data = df_data.drop(columns=column_to_drop)

    non_feature_columns = ["timestamp", "machine_status", "Split", "Cycle"]

    corr_matrix = df_data.drop(columns=["timestamp", "machine_status", "Split"]).dropna().corr()

    g_similar_ideas = nx.from_pandas_adjacency(
        corr_matrix[corr_matrix >= GROUP_MIN_SIMILARITY].fillna(0)
    )
    cor_groups = find_largest_group_of_similar_ideas(g_similar_ideas, [], corr_matrix)
    no_group_ideas = list(g_similar_ideas.nodes)

    var_to_keep = []
    var_to_drop = []
    for group in cor_groups:
        sorted_sensors = sorted(
            {x: int(df_data[x].isna().sum()) for x in group}.items(), key=lambda x: x[1]
        )
        var_to_keep.extend([sorted_sensors[0][0]])
        var_to_drop.extend([sorted_sensors[i][0] for i in range(1, len(sorted_sensors))])

    df_data = df_data.drop(columns=var_to_drop)

    broken_idx = df_data[df_data["machine_status"] == "BROKEN"].index.values
    broken_idx_test = df_data_test[df_data_test["machine_status"] == "BROKEN"].index.values

    df_data["predictive_machine_status"] = df_data.apply(
        lambda row: (
            "DANGER_ZONE"
            if any(
                [
                    idx - int(row.name) < ALPHA_PREDICTIVE_STRENGH * 24 * 60
                    and idx - int(row.name) > 0
                    for idx in broken_idx
                ]
            )
            else row["machine_status"]
        ),
        axis=1,
    )

    df_data_test["predictive_machine_status"] = df_data_test.apply(
        lambda row: (
            "DANGER_ZONE"
            if any(
                [
                    idx - int(row.name) < ALPHA_PREDICTIVE_STRENGH * 24 * 60
                    and idx - int(row.name) > 0
                    for idx in broken_idx_test
                ]
            )
            else row["machine_status"]
        ),
        axis=1,
    )

    df_data["predictive_machine_status_label"] = df_data["predictive_machine_status"].map(
        label_mapping
    )

    df_data_test["predictive_machine_status_label"] = df_data_test["predictive_machine_status"].map(
        label_mapping
    )

    non_feature_columns += ["predictive_machine_status", "predictive_machine_status_label"]

    return df_data, df_data_test
