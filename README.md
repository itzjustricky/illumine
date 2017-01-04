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

# Compress the ensemble object so internally leaves/terminal-nodes
# with the same split paths are compressed into one object
lucid_gbr.compress()
print("{} unique nodes, {} total nodes, and {} tree estimators"
      .format(lucid_gbr.unique_leaves_count,
              lucid_gbr.total_leaves_count,
              len(lucid_gbr)))
```

## Continuing from Last Example ...

```
# Using some analysis tools
from illumine.woodland import score_leaves

def sign_score_function(y_hat, y):
    return np.sum(np.sign(y_hat) == np.sign(y))

# This is a dictionary of leaf-paths mapped to their
# scores over the passed data X_df, y
leaf_scores = score_leaves(
    lucid_ensemble, X_df, y,
    score_function=sign_score_function,
    normalize_score=True)
```
