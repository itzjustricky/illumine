"""
    Description:



    @author: Ricky Chang
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from illumine.woodland import gather_leaf_values


def test_gather_leaf_values():
    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    clf = GradientBoostingRegressor(
        max_depth=1, n_estimators=3, random_state=3)
    clf.fit(X_df, y)

    # gather_leaf_values with gather_method='aggregate', gathers all the values
    # for a given unique leaf/terminal-node
    leaf_values = gather_leaf_values(
        clf, X_df,
        feature_names=X_df.columns,
        gather_method='aggregate')

    # extract the expected leaf_values from a pickle file
    script_dir = os.path.dirname(__file__)
    expected_leaf_values = \
        pickle.load(open(
            os.path.join(script_dir, 'test_material/expected_leaf_values.pkl'),
            'rb')
        )

    for key, val in expected_leaf_values.items():
        np.testing.assert_almost_equal(leaf_values[key], val)


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
