"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from illumine.utils import StopWatch
from illumine.woodland import make_LucidEnsemble


def test_pruning():
    """ Tests the pruning methods on both the objects
        LucidEnsemble and the CompressedEnsemble
    """
    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    regr = GradientBoostingRegressor(
        max_depth=3, n_estimators=100, random_state=3)
    regr.fit(X_df, y)
    lucid_ensemble = make_LucidEnsemble(
        regr, feature_names=X_df.columns, print_precision=3)
    compressed_ensemble = lucid_ensemble.compress()

    print("There are {} estimators in the ensemble".format(len(lucid_ensemble)))
    print("There are {} leaves in the ensemble".format(len(compressed_ensemble)))

    with StopWatch('Prune by leaf'):
        compressed_ensemble.prune_by_leaves(X_df, y, metric_function='mse')
    print("There are {} leaves left after pruning".format(len(compressed_ensemble)))

    ypred = regr.predict(X_df)
    ypred_after_eprune = lucid_ensemble.predict(X_df)
    ypred_after_lprune = compressed_ensemble.predict(X_df)

    # make sure the loss over which the prune happened is <= previous loss
    assert(mean_squared_error(y, ypred_after_eprune) <= mean_squared_error(y, ypred))
    assert(mean_squared_error(y, ypred_after_lprune) <= mean_squared_error(y, ypred))


if __name__ == "__main__":
    test_pruning()

"""
# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_pruning()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
