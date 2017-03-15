
"""
    Description:
        Unit tests for methods in factory_methods and
        testing the methods of the objects in leaf_objects

    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from illumine.tree import make_LucidTree
from sklearn.tree import DecisionTreeRegressor
from illumine.utils import StopWatch


def test_LucidTree():

    X1 = np.arange(0, 10, 0.1) + np.random.rand(100)
    X2 = np.arange(10, 20, 0.1) + np.random.rand(100)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    # try a range of depths
    max_depths = [1, 2, 3, 10]

    for max_depth in max_depths:
        regr = DecisionTreeRegressor(max_depth=max_depth)
        regr.fit(X_df, y)

        lucid_tree = make_LucidTree(
            regr, feature_names=X_df.columns, print_precision=3)
        with StopWatch("Lucid prediction for depth {}".format(max_depth)):
            lucid_pred = lucid_tree.predict(X_df)
        sk_pred = regr.predict(X_df)

        # test prediction outputted from LucidTree
        np.testing.assert_almost_equal(lucid_pred, sk_pred)


"""
if __name__ == "__main__":
    test_LucidTree()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_LucidTree()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
