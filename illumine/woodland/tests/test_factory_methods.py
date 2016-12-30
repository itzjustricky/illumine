"""
    Description:
        Unit tests for methods in factory_methods and
        testing the methods of the objects in leaf_objects

    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

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
        regr = DecisionTreeRegressor(max_depth=max_depth)
        regr.fit(X_df, y)

        lucid_tree = make_LucidSKTree(
            regr, feature_names=X_df.columns, print_precision=3)
        lucid_pred = lucid_tree.predict(X_df)
        sk_pred = regr.predict(X_df)

        # test prediction outputted from LucidSKTree
        np.testing.assert_almost_equal(lucid_pred, sk_pred)


def test_LucidGBR():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    gbr_regr = GradientBoostingRegressor(n_estimators=5000, max_depth=3)
    gbr_regr.fit(X_df, y)
    with StopWatch("LucidSKEnsemble Gradient Boost construction"):
        lucid_gbr = make_LucidSKEnsemble(
            gbr_regr, feature_names=X_df.columns, print_precision=3)

    with StopWatch("Scikit-learn Gradient Boost prediction"):
        gbr_pred = gbr_regr.predict(X_df)

    with StopWatch("Lucid Gradient Boost (non-compressed)"):
        lucid_gbr_pred = lucid_gbr.predict(X_df)

    # test prediction outputted from LucidSKEnsemble
    np.testing.assert_almost_equal(lucid_gbr_pred, gbr_pred)

    with StopWatch("Compression of Lucid Gradient Boost"):
        lucid_gbr.compress()
    print("{} unique nodes, {} total nodes, and {} tree estimators"
          .format(lucid_gbr.unique_leaves_count,
                  lucid_gbr.total_leaves_count,
                  len(lucid_gbr)))

    with StopWatch("Lucid Gradient Boost (compressed)"):
        cgbr_pred = lucid_gbr.predict(X_df)

    # test the compressed prediction
    np.testing.assert_almost_equal(cgbr_pred, gbr_pred)


def test_LucidRF():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    rf_regr = RandomForestRegressor(n_estimators=1000, max_depth=5, bootstrap=False)
    rf_regr.fit(X_df, y)
    with StopWatch("LucidSKEnsemble Random Forest construction"):
        lucid_rf = make_LucidSKEnsemble(
            rf_regr, feature_names=X_df.columns, print_precision=5)

    # If this is not float32 there are precision errors
    # apparently DecisionTreeRegressor within RandomForestRegressor
    # requires that the matrix be of type float32 so there is a
    # type conversion from types to float32
    X_df = X_df.astype(np.float32)

    with StopWatch("Scikit-learn Random Forest prediction"):
        rf_pred = rf_regr.predict(X_df)
    with StopWatch("Lucid Random Forest (non-compressed)"):
        lucid_rf_pred = lucid_rf.predict(X_df)
    np.testing.assert_almost_equal(lucid_rf_pred, rf_pred)
    with StopWatch("Compression of Lucid Random Forest"):
        lucid_rf.compress()
    print("{} unique nodes, {} total nodes, and {} tree estimators"
          .format(lucid_rf.unique_leaves_count,
                  lucid_rf.total_leaves_count,
                  len(lucid_rf)))

    with StopWatch("Lucid Random Forest (compressed)"):
        crf_pred = lucid_rf.predict(X_df)
    np.testing.assert_almost_equal(crf_pred, rf_pred)


if __name__ == "__main__":
    test_LucidSKTree()
    test_LucidGBR()
    test_LucidRF()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_LucidSKTree()
        test_LucidGBR()
        test_LucidRF()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
