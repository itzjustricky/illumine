import bpdb
"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from illumine.utils import StopWatch
from illumine.woodland import make_LucidSKEnsemble


def test_pruning():
    """ Tests the pruning methods on both the LucidSKEnsemble
        and the CompressedEnsemble object
    """
    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    regr = GradientBoostingRegressor(
        max_depth=3, n_estimators=100, random_state=3)
    regr.fit(X_df, y)
    lucid_ensemble = make_LucidSKEnsemble(
        regr, feature_names=X_df.columns, print_precision=3)
    compressed_ensemble = lucid_ensemble.compress()

    with StopWatch('Prune by estimator'):
        lucid_ensemble.prune_by_estimators(X_df, y)
    with StopWatch('Prune by leaf'):
        compressed_ensemble.prune_by_leaves(X_df, y)

    ypred = regr.predict(X_df)
    ypred_after_eprune = lucid_ensemble.predict(X_df)
    ypred_after_lprune = compressed_ensemble.predict(X_df)

    # make sure the loss over which the prune happened is <= previous loss
    assert(lucid_ensemble._loss(y, ypred_after_eprune) <= lucid_ensemble._loss(y, ypred))
    assert(compressed_ensemble._loss(y, ypred_after_lprune) <= lucid_ensemble._loss(y, ypred))


"""
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
