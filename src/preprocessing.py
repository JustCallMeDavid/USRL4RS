import os

import pandas as pd
from random import sample


def generate_train_test_splits(data_path, number_splits, min_interactions_user, data_type, train_pct):
    """
    Generates cross-validation train/test-splits from the given datafile.
    :param data_path: data location
    :param number_splits: number of cross validation splits/folds to generate
    :param min_interactions_user: minimum number of items rated per user to be included in the dataset
    :param data_type: the type of data given
    :param train_pct: train/test split percentage
    :return: the generated splits, a dict mapping ids to their original value and the total amount of items in the data
    """

    if data_type == 'ml1m':
        colnames = ['user_id', 'item_id', 'rating', 'timestamp']
        separator = '::'
    elif data_type == 'ymusic':
        raise NotImplementedError()  # tbd.
    else:
        raise ValueError(f'Data type {data_type} not recognized.')

    df = pd.read_csv(data_path, sep=separator, names=colnames, engine='python')

    # transforms the ids according to their order
    unique_items = df.item_id.unique()
    id_mapping_dict = {k: v for v, k in enumerate(unique_items)}
    df.item_id = df.item_id.apply(lambda x: id_mapping_dict[x])
    total_items = len(unique_items)

    unique_users = df.user_id.unique()
    user_mapping_dict = {k:v for v, k in enumerate(unique_users)}
    df.user_id = df.user_id.apply(lambda x : user_mapping_dict[x])
    total_users = len(unique_users)

    # construct a column with transformed ids (sequential)
    df_vc = df.user_id.value_counts()
    df_vc_gtmin = df_vc[df_vc > min_interactions_user]
    users_gtmin = df_vc_gtmin.index.to_list()

    train_test_splits = []
    for i in range(number_splits):
        train_users = sample(users_gtmin, int(len(users_gtmin) * train_pct))
        # test_users = list(set(users_gtmin) - set(train_users))
        train_df, test_df = df[df.user_id.isin(train_users)], df[~df.user_id.isin(train_users)]
        train_test_splits.append((train_df, test_df))
        assert not (set(train_df.user_id.unique()) & set(test_df.user_id.unique()))

    return train_test_splits, (id_mapping_dict, user_mapping_dict), (total_items, total_users)
