"""
    Description:
        Test cases for the methods in
        leaf_analysis.py module.

    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from illumine.woodland import gather_leaf_values
from illumine.woodland import make_LucidSKEnsemble


def test_gather_leaf_values():
    # This tests just makes sure nothing breaks
    # when calling gather leaf values
    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    regr = GradientBoostingRegressor(
        max_depth=3, n_estimators=100, random_state=3)
    regr.fit(X_df, y)
    lucid_ensemble = make_LucidSKEnsemble(
        regr, feature_names=X_df.columns, print_precision=3)

    # gather_leaf_values with gather_method='aggregate',
    # gathers all the values for a given unique leaf/terminal-node
    gather_leaf_values(
        lucid_ensemble, X_df,
        feature_names=X_df.columns,
        gather_method='aggregate')


if __name__ == "__main__":
    test_gather_leaf_values()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_gather_leaf_values()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
