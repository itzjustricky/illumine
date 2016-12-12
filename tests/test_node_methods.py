"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from illumine.woodland import make_LucidSKTree
from illumine.woodland import make_LucidSKEnsemble


def test_LucidSKTree():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    clf = DecisionTreeRegressor()
    clf.fit(X_df, y)

    lucid_tree = make_LucidSKTree(
        clf, feature_names=X_df.columns, float_precision=8)
    lucid_pred = lucid_tree.predict(X_df)
    sk_pred = clf.predict(X_df)

    np.testing.assert_almost_equal(lucid_pred.values, sk_pred)


def test_LucidSKEnsemble():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    clf = GradientBoostingRegressor()
    clf.fit(X_df, y)

    lucid_ensemble = make_LucidSKEnsemble(
        clf, feature_names=X_df.columns, float_precision=8)
    lucid_pred = lucid_ensemble.predict(X_df)
    sk_pred = clf.predict(X_df)
    np.testing.assert_almost_equal(lucid_pred.values, sk_pred)


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_LucidSKTree()
        test_LucidSKEnsemble()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
if __name__ == "__main__":
    test_LucidSKTree()
    test_LucidSKEnsemble()
"""
