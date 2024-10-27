import networkx as nx

GROUP_MIN_SIMILARITY = 0.8


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


def select_features_to_keep(df_data, cor_groups):
    var_to_keep = []
    var_to_drop = []
    for group in cor_groups:
        sorted_sensors = sorted(
            {x: int(df_data[x].isna().sum()) for x in group}.items(), key=lambda x: x[1]
        )
        var_to_keep.extend([sorted_sensors[0][0]])
        var_to_drop.extend([sorted_sensors[i][0] for i in range(1, len(sorted_sensors))])
    return df_data.drop(columns=var_to_drop)
