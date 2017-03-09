"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.datasets.california_housing import fetch_california_housing

from illumine.utils import StopWatch
from illumine.woodland import weighted_nurturing

__test__ = False


def test_weighted_nurturing():
    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)
    y_train = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_train = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])
    X_test, y_test = X_train, y_train

    # cal_housing = fetch_california_housing()
    # X_train, X_test, y_train, y_test = \
    #     train_test_split(cal_housing.data, cal_housing.target,
    #                      test_size=0.3, random_state=5)
    # X_train = pd.DataFrame(X_train, columns=cal_housing.feature_names)
    # X_test = pd.DataFrame(X_test, columns=cal_housing.feature_names)

    model_params = {
        'max_depth': 10, 'n_estimators': 200, 'random_state': 5,
        'n_jobs': -1, 'bootstrap': True,
    }
    with StopWatch('weighted nurturing'):
        nurtured_ensemble = weighted_nurturing(
            RandomForestRegressor, X_train, y_train,
            feature_names=X_train.columns,
            n_iterations=20, metric_function='mse',
            n_prunes=50, update_weight=0.3,
            model_params=model_params)

    print("The mean-squared error for nurtured ensemble prediction: {}".format(
          mean_squared_error(nurtured_ensemble.predict(X_test), y_test)))
    # plain_rf = RandomForestRegressor(
    #     min_samples_leaf=100, max_depth=8,
    #     n_estimators=1000, random_state=5, n_jobs=-1)
    plain_params = {'random_state': 5, }
    plain_rf = RandomForestRegressor(n_estimators=3000, max_depth=8, n_jobs=-1, **plain_params)
    plain_rf.fit(X_train, y_train)
    plain_gbr = GradientBoostingRegressor(
        n_estimators=1000, max_depth=5, **plain_params)
    plain_gbr.fit(X_train, y_train)

    print("The mean-squared error for plain RandomForest prediction: {}".format(
          mean_squared_error(plain_rf.predict(X_test), y_test)))
    print("The mean-squared error for plain GradientBoost prediction: {}".format(
          mean_squared_error(plain_gbr.predict(X_test), y_test)))
    # make sure the loss over which the prune happened is <= previous loss
    # assert(mean_squared_error(y, ypred_after_eprune) <= mean_squared_error(y, ypred))
    # assert(mean_squared_error(y, ypred_after_lprune) <= mean_squared_error(y, ypred))


if __name__ == "__main__":
    test_weighted_nurturing()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_weighted_nurturing()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
