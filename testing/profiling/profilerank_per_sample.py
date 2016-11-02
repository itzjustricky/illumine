"""
    Description:
        Profile the rank_per_sample method.


    @author: Ricky Chang
"""

import profile
import numpy as np

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle

from illumine import woodland


boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.1)
X_train, y_train = X[:offset], y[:offset]

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

top_leaf_samples = woodland.rank_per_sample(clf, X_train, boston.feature_names, n_top=3)
profile.run('woodland.rank_per_sample(clf, X_train, boston.feature_names, n_top=3)')
