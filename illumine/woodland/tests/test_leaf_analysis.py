"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from illumine.woodland import gather_leaf_values
from illumine.woodland import rank_leaves


def test_gather_leaf_values():
    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    clf = GradientBoostingRegressor(max_depth=1, n_estimators=3, random_state=3)
    clf.fit(X_df, y)

    # gather_leaf_values with gather_method='aggregate', gathers all the values
    # for a given unique leaf/terminal-node
    leaf_values = gather_leaf_values(
        clf, X_df,
        feature_names=X_df.columns,
        gather_method='aggregate')

    expected_pairs = {
        'x1<=7.05': -0.31863,
        'x1<=7.15': -0.3451,
        'x1<=3.35': 0.67753,
        'x1>3.35': -0.34903,
        'x1>7.05': 0.78009,
        'x1>7.15': 0.88741
    }
    # Check the lengths are the same
    assert(len(expected_pairs) == len(leaf_values))

    # Check all the mean values are as expected
    for path, val in expected_pairs.items():
        assert(val == round(np.mean(leaf_values[path]), 5))


def test_rank_leaves():
    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    clf = GradientBoostingRegressor(max_depth=1, n_estimators=3, random_state=3)
    clf.fit(X_df, y)

    # gather_leaf_values with gather_method='aggregate', gathers all the values
    # for a given unique leaf/terminal-node
    leaf_values = gather_leaf_values(
        clf, X_df,
        feature_names=X_df.columns,
        gather_method='aggregate')

    top_leaves = rank_leaves(leaf_values, rank_method='absolute-sum')
    expected_pair_rank = {
        'x1>7.15': 24.84748,
        'x1<=7.15': 24.8472,
        'x1<=3.35': 23.03602,
        'x1>3.35': 23.03598,
        'x1<=7.05': 22.62273,
        'x1>7.05': 22.62261
    }
    # Check the lengths are the same
    assert(len(expected_pair_rank) == len(leaf_values))

    # Check all the mean values are as expected
    for path, val in expected_pair_rank.items():
        assert(val == round(np.mean(top_leaves[path]), 5))


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_gather_leaf_values()
        test_rank_leaves()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)