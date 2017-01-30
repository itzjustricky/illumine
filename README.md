# illumine

This package is used to analyze Scikit-learn models. Currently, only modules
inside the illumine.woodland package have been developed.

The functions inside the woodland/factory_methods.py parses the decision trees trained
in the Scikit-learn library. Currently support is created only for regression trees.
Call all the unit tests via nosetests.


## Example Use (taken from unit tests):

```
from sklearn.ensemble import GradientBoostingRegressor
from illumine.woodland import make_LucidSKEnsemble

X1 = np.arange(0, 10, 0.1)
X2 = np.arange(10, 20, 0.1)

y = np.sin(X1).ravel() + np.cos(X2).ravel()
X_df = pd.DataFrame(np.array([X1, X2]).T, columns=['x1', 'x2'])

gbr_regr = GradientBoostingRegressor(n_estimators=5000, max_depth=3)
gbr_regr.fit(X_df, y)

lucid_gbr = make_LucidSKEnsemble(
    gbr_regr, feature_names=X_df.columns, print_precision=3)

gbr_pred = gbr_regr.predict(X_df)
lucid_gbr_pred = lucid_gbr.predict(X_df)

# test prediction outputted from LucidSKEnsemble
np.testing.assert_almost_equal(lucid_gbr_pred, gbr_pred)

```
