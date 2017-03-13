"""
    Description:
        Unit tests for methods in factory_methods and
        testing the methods of the objects in leaf_objects

    @author: Ricky Chang
"""

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from illumine.woodland import make_LucidEnsemble
from illumine.utils import StopWatch


def test_GradientBoost():

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

    with StopWatch("Lucid Gradient Boost (non-compressed) prediction"):
        lucid_gbr_pred = lucid_gbr.predict(X_df)

    ######################################################
    # test prediction outputted from LucidEnsemble
    np.testing.assert_almost_equal(lucid_gbr_pred, gbr_pred)
    assert(np.all(gbr_regr.apply(X_df) == lucid_gbr.apply(X_df)))

    with StopWatch("Compression of Lucid Gradient Boost"):
        compressed_lucid_gbr = lucid_gbr.compress()
    print("{} unique nodes and {} # of estimators"
          .format(compressed_lucid_gbr.n_leaves,
                  len(lucid_gbr)))

    with StopWatch("Lucid Gradient Boost (compressed) prediction"):
        cgbr_pred = compressed_lucid_gbr.predict(X_df)

    ######################################################
    # test the compressed prediction
    np.testing.assert_almost_equal(cgbr_pred, gbr_pred)

    # test comparison, compare the leaves of two
    # LucidEnsembles made from the the same arguments
    lucid_gbr2 = make_LucidEnsemble(
        gbr_regr, feature_names=X_df.columns, print_precision=3)
    compressed_lucid_gbr2 = lucid_gbr2.compress()

    assert(set(compressed_lucid_gbr.leaves) == set(compressed_lucid_gbr2.leaves))

    script_dir = os.path.dirname(__name__)
    ######################################################
    # test pickling functionality
    pickle_path = os.path.join(script_dir, 'lucid_gbr.pkl')
    with open(pickle_path, 'wb') as fh:
        pickle.dump(lucid_gbr, fh)
    with open(pickle_path, 'rb') as fh:
        lucid_gbr_pickle = pickle.load(fh)
        np.testing.assert_almost_equal(
            lucid_gbr_pickle.predict(X_df),
            lucid_gbr_pred)
    os.remove(pickle_path)

    pickle_path = os.path.join(script_dir, 'compressed_lucid_gbr.pkl')
    with open(pickle_path, 'wb') as fh:
        pickle.dump(compressed_lucid_gbr, fh)
    with open(pickle_path, 'rb') as fh:
        compressed_lucid_gbr_pickle = pickle.load(fh)
        np.testing.assert_almost_equal(
            compressed_lucid_gbr_pickle.predict(X_df),
            cgbr_pred)
    os.remove(pickle_path)


def test_RandomForest():

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
    assert(np.all(rf_regr.apply(X_df) == lucid_rf.apply(X_df)))

    with StopWatch("Compression of Lucid Random Forest"):
        compressed_lucid_rf = lucid_rf.compress()
    print("{} unique nodes and {} # of estimators"
          .format(compressed_lucid_rf.n_leaves,
                  len(lucid_rf)))

    with StopWatch("Lucid Random Forest (compressed) prediction"):
        crf_pred = compressed_lucid_rf.predict(X_df)
    np.testing.assert_almost_equal(crf_pred, rf_pred)

    ######################################################
    # test comparison, compare the leaves of two
    # LucidEnsembles made from the the same arguments
    lucid_rf2 = make_LucidEnsemble(
        rf_regr, feature_names=X_df.columns, print_precision=3)
    compressed_lucid_rf2 = lucid_rf2.compress()

    assert(set(compressed_lucid_rf.leaves) == set(compressed_lucid_rf2.leaves))

    script_dir = os.path.dirname(__name__)
    ######################################################
    # test pickling functionality
    pickle_path = os.path.join(script_dir, 'lucid_rf.pkl')
    with open(pickle_path, 'wb') as fh:
        pickle.dump(lucid_rf, fh)
    with open(pickle_path, 'rb') as fh:
        lucid_rf_pickle = pickle.load(fh)
        np.testing.assert_almost_equal(
            lucid_rf_pickle.predict(X_df),
            lucid_rf_pred)
    os.remove(pickle_path)

    pickle_path = os.path.join(script_dir, 'compressed_lucid_rf.pkl')
    with open(pickle_path, 'wb') as fh:
        pickle.dump(compressed_lucid_rf, fh)
    with open(pickle_path, 'rb') as fh:
        compressed_lucid_rf_pickle = pickle.load(fh)
        np.testing.assert_almost_equal(
            compressed_lucid_rf_pickle.predict(X_df),
            crf_pred)
    os.remove(pickle_path)


"""
if __name__ == "__main__":
    test_GradientBoost()
    test_RandomForest()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_GradientBoost()
        test_RandomForest()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
