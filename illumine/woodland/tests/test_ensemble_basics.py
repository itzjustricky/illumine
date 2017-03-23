"""
    Unit tests for factory methods for creating the
    ensemble objects. Only test the construction and
    predictions of the objects.

    @author: Ricky Chang
"""

# import os
# import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from illumine.woodland import make_LucidEnsemble
from illumine.utils import StopWatch


def test_GradientBoost_basics():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    gbr_regr = GradientBoostingRegressor(n_estimators=5000, max_depth=3)
    gbr_regr.fit(X_df, y)

    with StopWatch("LucidEnsemble Gradient Boost construction"):
        lucid_gbr = make_LucidEnsemble(
            gbr_regr, feature_names=X_df.columns, print_precision=3)
    with StopWatch("Scikit-learn Gradient Boost prediction"):
        gbr_pred = gbr_regr.predict(X_df)
    with StopWatch("Lucid Gradient Boost prediction"):
        lucid_gbr_pred = lucid_gbr.predict(X_df)

    # test prediction outputted from LucidEnsemble
    np.testing.assert_almost_equal(lucid_gbr_pred, gbr_pred)
    # just test that this runs
    lucid_gbr.apply(X_df)


def test_RandomForest_basics():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    rf_regr = RandomForestRegressor(n_estimators=1000, max_depth=5, bootstrap=False)
    rf_regr.fit(X_df, y)

    with StopWatch("LucidEnsemble Random Forest construction"):
        lucid_rf = make_LucidEnsemble(
            rf_regr, feature_names=X_df.columns, print_precision=5)
    # If this is not float32 there are precision errors
    # apparently DecisionTreeRegressor within RandomForestRegressor
    # requires that the matrix be of type float32 so there is a
    # type conversion from types to float32
    X_df = X_df.astype(np.float32)

    with StopWatch("Scikit-learn Random Forest prediction"):
        rf_pred = rf_regr.predict(X_df)
    with StopWatch("Lucid Random Forest (non-compressed) prediction"):
        lucid_rf_pred = lucid_rf.predict(X_df)

    ######################################################
    # test prediction outputted from LucidEnsemble
    np.testing.assert_almost_equal(lucid_rf_pred, rf_pred)
    # just test that this runs
    lucid_rf.apply(X_df)


if __name__ == "__main__":
    test_GradientBoost_basics()
    test_RandomForest_basics()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_GradientBoost_basics()
        test_RandomForest_basics()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
