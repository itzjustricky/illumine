"""
    Description:
        Test cases for methods in the
        leaf_validation.py module


    @author: Ricky Chang
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from illumine.woodland import make_LucidSKEnsemble
from illumine.woodland import score_leaves
from illumine.woodland import score_leaf_group


def test_score_leaves():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    gbr_regr = GradientBoostingRegressor(n_estimators=5000, max_depth=3)
    gbr_regr.fit(X_df, y)

    lucid_ensemble = make_LucidSKEnsemble(
        gbr_regr, feature_names=X_df.columns, print_precision=3)
    compressed_ensemble = lucid_ensemble.compress()

    def sign_score_function(y_hat, y):
        return np.sum(np.sign(y_hat) == np.sign(y))

    leaf_scores = score_leaves(
        compressed_ensemble, X_df, y,
        score_function=sign_score_function,
        required_threshold=X_df.shape[0] * 0.2,
        normalize_score=True)

    # Check all the leaf-paths in leaf_scores are in lucid_ensemble paths
    for leaf in leaf_scores.keys():
        assert(leaf in compressed_ensemble.leaves)


def test_score_leaf_groups():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    gbr_regr = GradientBoostingRegressor(n_estimators=5000, max_depth=3)
    gbr_regr.fit(X_df, y)

    lucid_ensemble = make_LucidSKEnsemble(
        gbr_regr, feature_names=X_df.columns, print_precision=3)
    compressed_ensemble = lucid_ensemble.compress()

    def sign_score_function(y_hat, y):
        return np.sum(np.sign(y_hat) == np.sign(y))

    leaf_scores = score_leaves(
        compressed_ensemble, X_df, y,
        score_function=sign_score_function,
        normalize_score=True)

    leaf_group = list(compressed_ensemble.leaves)[0]

    leaf_group_score = score_leaf_group(
        [leaf_group], compressed_ensemble, X_df, y,
        score_function=sign_score_function,
        normalize_score=True)

    assert(leaf_group_score == leaf_scores[leaf_group])


if __name__ == "__main__":
    test_score_leaves()
    test_score_leaf_groups()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_score_leaves()
        test_score_leaf_groups()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
