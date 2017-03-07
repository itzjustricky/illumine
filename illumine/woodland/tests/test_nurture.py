"""
    Description:



    @author: Ricky Chang
"""

# import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets.california_housing import fetch_california_housing

from illumine.utils import StopWatch
from illumine.woodland import weighted_nurturing


def test_weighted_nurturing():
    """ Tests the pruning methods on both the objects
        LucidSKEnsemble and the CompressedEnsemble
    """
    # X1 = np.arange(0, 10, 0.1)
    # X2 = np.arange(10, 20, 0.1)
    # y = np.sin(X1).ravel() + np.cos(X2).ravel()
    # X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])
    cal_housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = \
        train_test_split(cal_housing.data, cal_housing.target,
                         test_size=0.3, random_state=5)

    X_train = pd.DataFrame(X_train, columns=cal_housing.feature_names)
    X_test = pd.DataFrame(X_test, columns=cal_housing.feature_names)

    model_params = {
        'max_depth': 8, 'n_estimators': 200,
        'min_samples_leaf': 100, 'n_jobs': -1,
    }
    with StopWatch('weighted nurturing'):
        nurtured_ensemble = weighted_nurturing(
            RandomForestRegressor, X_train, y_train,
            feature_names=cal_housing.feature_names,
            n_iterations=5, metric_function='mse',
            n_prunes=0, update_weight=0.5,
            model_params=model_params)

    plain_rf = RandomForestRegressor()
    plain_rf.fit(X_train, y_train)

    print(mean_squared_error(nurtured_ensemble.predict(X_test), y_test))
    print(mean_squared_error(plain_rf.predict(X_test), y_test))
    # make sure the loss over which the prune happened is <= previous loss
    # assert(mean_squared_error(y, ypred_after_eprune) <= mean_squared_error(y, ypred))
    # assert(mean_squared_error(y, ypred_after_lprune) <= mean_squared_error(y, ypred))


"""
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
