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
from illumine.utils import StopWatch


def test_LucidSKTree():

    X1 = np.arange(0, 10, 0.1) + np.random.rand(100)
    X2 = np.arange(10, 20, 0.1) + np.random.rand(100)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    # try a range of depths
    max_depths = [1, 2, 3, 10]

    for max_depth in max_depths:
        clf = DecisionTreeRegressor(max_depth=max_depth)
        clf.fit(X_df, y)

        lucid_tree = make_LucidSKTree(
            clf, feature_names=X_df.columns, float_precision=8)
        lucid_pred = lucid_tree.predict(X_df)
        sk_pred = clf.predict(X_df)

        # test prediction outputted from LucidSKTree
        np.testing.assert_almost_equal(lucid_pred, sk_pred)


def test_LucidSKEnsemble():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    clf = GradientBoostingRegressor(n_estimators=5000, max_depth=10)
    clf.fit(X_df, y)

    with StopWatch("LucidSKEnsemble construction"):
        lucid_ensemble = make_LucidSKEnsemble(
            clf, feature_names=X_df.columns, float_precision=8)

    with StopWatch("Scikit-learn"):
        sk_pred = clf.predict(X_df)
    with StopWatch("Lucid (non-compressed)"):
        lucid_pred = lucid_ensemble.predict(X_df)
    # test prediction outputted from LucidSKEnsemble
    np.testing.assert_almost_equal(lucid_pred, sk_pred)

    lucid_ensemble.compress()
    print("{} unique nodes and {} tree estimators"
          .format(len(lucid_ensemble._compressed_ensemble),
                  len(lucid_ensemble)))

    with StopWatch("Lucid (compressed)"):
        lucid_pred = lucid_ensemble.predict(X_df)
    # test the compressed prediction
    np.testing.assert_almost_equal(lucid_pred, sk_pred)


if __name__ == "__main__":
    # test_LucidSKTree()
    test_LucidSKEnsemble()

"""
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
