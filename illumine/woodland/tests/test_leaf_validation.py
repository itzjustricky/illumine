"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from illumine.woodland import make_LucidSKEnsemble
from illumine.woodland import score_leaves


def test_score_leaves():

    X1 = np.arange(0, 10, 0.1)
    X2 = np.arange(10, 20, 0.1)

    y = np.sin(X1).ravel() + np.cos(X2).ravel()
    X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

    gbr_regr = GradientBoostingRegressor(n_estimators=5000, max_depth=3)
    gbr_regr.fit(X_df, y)

    lucid_ensemble = make_LucidSKEnsemble(
        gbr_regr, feature_names=X_df.columns, print_precision=3)
    lucid_ensemble.compress()

    print(score_leaves(
        lucid_ensemble, X_df, y,
        score_function=lambda y_hat, y: -np.linalg.norm(y_hat - y),
        required_threshold=X_df.shape[0] * 0.1,
        normalize_score=True)
    )


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        test_score_leaves()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
